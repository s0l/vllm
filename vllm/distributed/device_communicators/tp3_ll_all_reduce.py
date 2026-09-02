# SPDX-License-Identifier: Apache-2.0
"""One-shot pinned-host LL all-reduce for small TP3 rows (AG2 lever P1-LL).

Latency-bound small collectives (decode qlen1..3: 10-30 KB payloads) pay
~46 us each through host-staged NCCL on this no-P2P topology; the one-shot
pinned-host protocol measured 23.8 us median at 30 KB (S1/S2, bit-exact,
deterministic sum order; S3 graph-replay proven).

Protocol per op (mirrors vLLM's CustomAllreduce contract, which this
originally lacked): zero-copy write into the rank's pinned slot, system
fence, START barrier on pairwise counters, zero-copy read of every peer
slot with fp32 accumulation, END barrier, then the epoch is advanced. The
end barrier is what makes slot reuse safe; the exact-equality wait makes
any epoch divergence a detectable hang rather than a silent cross-epoch
read (the failure mode of the first version, acceptance 0.0000).

Scope: TP world 3, hidden 5120, rows <= VLLM_TP3_LL_MAX_ROWS (default 16),
bf16/fp32 tensors. Anything else falls back to NCCL. Enabled by
VLLM_TP3_LL_REDUCE=1. Sum order is fixed by construction (batch-invariant),
fp32 accumulation: numerics differ from NCCL bf16 ring at ULP level.
"""

from __future__ import annotations

import ctypes
import mmap
import os
import time
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

from vllm.logger import init_logger

logger = init_logger(__name__)

MAX_ROWS_DEFAULT = 16
HIDDEN = 5120
WORLD = 3
SLOT_ELEMS = 32 * HIDDEN  # max rows padded, bf16/fp32 stored as fp32
SLOT_BYTES = SLOT_ELEMS * 4
SIG_BYTES = 4096  # WORLD x Sig, generously padded
SHM_BYTES = WORLD * SLOT_BYTES + SIG_BYTES
SHM_PATH = "/dev/shm/ag2_tp3_ll_reduce"

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define WORLD 3

// Per-rank synchronization record, mirroring vLLM's Signal contract:
// pairwise counters for a start and an end barrier plus the rank's own
// epoch. Two barriers are mandatory: without the end barrier a peer that
// reaches the next operation can overwrite its slot while this rank is
// still reading it, and an exact-equality wait (not >=) turns any epoch
// divergence into a detectable hang instead of silent cross-epoch reads.
struct Sig {
    int start[8];
    int end[8];
    int epoch;
    int _pad[7];
};

__device__ __forceinline__ float ag2_to_f(const __nv_bfloat16 v) {
    return __bfloat162float(v);
}
__device__ __forceinline__ float ag2_to_f(const float v) { return v; }
__device__ __forceinline__ void ag2_from_f(__nv_bfloat16* p, float v) {
    *p = __float2bfloat16(v);
}
__device__ __forceinline__ void ag2_from_f(float* p, float v) { *p = v; }

__device__ __forceinline__ void st_sys(int* addr, int v) {
    asm volatile("st.volatile.global.u32 [%1], %0;" ::"r"(v), "l"(addr));
}
__device__ __forceinline__ int ld_sys(int* addr) {
    int v;
    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(addr));
    return v;
}

template <typename T>
__global__ void ll_ar_kernel(
    const T* __restrict__ inp, T* __restrict__ out,
    float* data_base, Sig* sigs, int rank, int n, long slot_elems)
{
    float* my_slot = data_base + (long)rank * slot_elems;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        my_slot[i] = ag2_to_f(inp[i]);
    }
    __threadfence_system();
    __syncthreads();

    int epoch = sigs[rank].epoch + 1;
    if (threadIdx.x < WORLD) {
        st_sys(&sigs[threadIdx.x].start[rank], epoch);
        while (ld_sys(&sigs[rank].start[threadIdx.x]) != epoch) {}
    }
    __syncthreads();

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float acc = 0.f;
        #pragma unroll
        for (int r = 0; r < WORLD; r++) {
            acc += ((volatile float*)(data_base + (long)r * slot_elems))[i];
        }
        ag2_from_f(&out[i], acc);
    }
    __threadfence_system();
    __syncthreads();

    if (threadIdx.x < WORLD) {
        st_sys(&sigs[threadIdx.x].end[rank], epoch);
        while (ld_sys(&sigs[rank].end[threadIdx.x]) != epoch) {}
    }
    __syncthreads();
    if (threadIdx.x == 0) sigs[rank].epoch = epoch;
}

void ll_all_reduce(torch::Tensor inp, torch::Tensor out, long data_base,
                   long sig_base, int rank, long slot_elems) {
    int n = inp.numel();
    // Single block: the barriers order threads within a block only.
    dim3 grid(1), block(1024);
    if (inp.scalar_type() == at::kBFloat16) {
        ll_ar_kernel<__nv_bfloat16><<<grid, block>>>(
            (const __nv_bfloat16*)inp.data_ptr(), (__nv_bfloat16*)out.data_ptr(),
            (float*)data_base, (Sig*)sig_base, rank, n, slot_elems);
    } else {
        ll_ar_kernel<float><<<grid, block>>>(
            inp.data_ptr<float>(), out.data_ptr<float>(),
            (float*)data_base, (Sig*)sig_base, rank, n, slot_elems);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ll_all_reduce", &ll_all_reduce);
}
"""


class _LLState:
    def __init__(self, rank: int):
        self.rank = rank
        build_dir = Path(f"/tmp/ag2_ll_reduce_build_{rank}")
        build_dir.mkdir(parents=True, exist_ok=True)
        self.mod = load_inline(
            name=f"ag2_tp3_ll_{rank}",
            cpp_sources="",
            cuda_sources=_CUDA_SRC,
            with_cuda=True,
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_120,code=sm_120"],
            build_directory=str(build_dir),
        )
        fd = os.open(SHM_PATH, os.O_RDWR)
        self.buf = mmap.mmap(fd, SHM_BYTES)
        os.close(fd)
        addr = ctypes.addressof(ctypes.c_char.from_buffer(self.buf))
        cudart = ctypes.CDLL("libcudart.so")
        rc = cudart.cudaHostRegister(
            ctypes.c_void_p(addr), ctypes.c_size_t(SHM_BYTES), 0
        )
        if rc not in (0, 712):  # 712 = already registered
            raise RuntimeError(f"cudaHostRegister failed: {rc}")
        dev_ptr = ctypes.c_void_p()
        if (
            cudart.cudaHostGetDevicePointer(
                ctypes.byref(dev_ptr), ctypes.c_void_p(addr), 0
            )
            != 0
        ):
            raise RuntimeError("cudaHostGetDevicePointer failed")
        self.base = dev_ptr.value
        self.sigs = dev_ptr.value + WORLD * SLOT_BYTES
        logger.info("AG2 TP3 LL all-reduce initialized (rank %d)", rank)


_state: _LLState | None = None
_failed = False


def _ensure_shm() -> None:
    # Create exclusively; peers wait for the creator instead of racing to
    # rewrite a file another rank may already be spinning on.
    if not os.path.exists(SHM_PATH):
        try:
            fd = os.open(SHM_PATH, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            try:
                os.ftruncate(fd, SHM_BYTES)
            finally:
                os.close(fd)
        except FileExistsError:
            pass
    for _ in range(200):
        if os.path.exists(SHM_PATH) and os.path.getsize(SHM_PATH) == SHM_BYTES:
            return
        time.sleep(0.05)
    raise RuntimeError("shared LL buffer was not initialized by any rank")


def max_rows() -> int:
    return int(os.environ.get("VLLM_TP3_LL_MAX_ROWS", str(MAX_ROWS_DEFAULT)))


def should_use_tp3_ll(tensor_dim: int, rows: int, hidden: int, tp_world: int) -> bool:
    return (
        not _failed
        and tensor_dim == 2
        and hidden == HIDDEN
        and tp_world == WORLD
        and 0 < rows <= max_rows()
    )


def tp3_ll_all_reduce(input_: torch.Tensor, rank: int) -> torch.Tensor:
    global _state, _failed
    if _state is None:
        try:
            _ensure_shm()
            _state = _LLState(rank)
        except Exception:
            _failed = True
            logger.exception("AG2 TP3 LL all-reduce init failed; NCCL fallback")
            raise
    st = _state
    out = torch.empty_like(input_)
    st.mod.ll_all_reduce(
        input_.contiguous(), out, st.base, st.sigs, st.rank, SLOT_ELEMS
    )
    return out
