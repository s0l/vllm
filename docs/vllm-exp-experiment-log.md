
### 2026-07-28 — P1-LL root cause found in vLLM's own source; approach replaced (Claude)

Owner pointed at the startup line listing SIX all-reduce backends
(NCCL_SYMM_MEM, QUICK_REDUCE, FLASHINFER, AITER_CUSTOM, CUSTOM, SYMM_MEM,
PYNCCL) of which only PYNCCL is selected. Reading `csrc/custom_all_reduce.cuh`
found my exact defect described in a comment above the `Signal` struct:

  "Two sets of peer counters are needed for two syncs: starting and ending
   an operation. The reason is that it's possible for peer GPU block to
   arrive at the second sync point while the current GPU block haven't
   passed the first sync point. Thus, peer GPU may write counter+1 while
   current GPU is busy waiting for counter."

My protocol has ONE barrier (start) and no end barrier, and compares with
`flags[peer] < seq`. A peer that races one op ahead therefore never blocks
me and I read its NEXT epoch's payload as if it were this epoch's - silent
corruption, permanent once the ranks drift, which matches the observed
acceptance 0.0000 exactly. Every standalone harness kept the ranks in
lockstep (barrier per op, or identical free-running loops with equal work),
so none of them could produce the drift; that is why five controls were
clean while the server failed identically every time. Hypothesis-by-
hypothesis debugging cost four restarts and did not find this; ten minutes
of reading the in-tree implementation did.

Corrected approach for P1-LL (replaces the hand-rolled protocol):

- vLLM's CustomAllreduce protocol is memory-agnostic: `barrier_at_start` /
  `barrier_at_end`, per-block per-rank-pair counters, epoch stored in the
  shared Signal struct (not a local Python tensor), release/acquire flag
  semantics, alternating counter arrays. It is battle-tested and already
  wired into the dispatch chain.
- Its only P2P dependency is the ALLOCATION path:
  `allocate_shared_buffer_and_handle` does cudaMalloc + cudaIpcGetMemHandle
  (csrc/libtorch_stable/custom_all_reduce.cu:157-190), and
  `_can_p2p`/`fully_connected` gate it off for >2 PCIe GPUs
  (custom_all_reduce.py:150-167).
- Plan: allocate ONE shared host region (/dev/shm) per node, register it
  with cudaHostRegister on each rank, take cudaHostGetDevicePointer, and
  hand per-rank offsets into that region to `init_custom_ar` /
  `register_buffer` in place of the IPC pointers. All three GPUs then have
  valid UVA device pointers to the same physical memory; the kernel needs
  no change. Expected regime: latency-bound small rows only (measured
  host-path all-reduce ~24 us vs NCCL ~46 us at 30 KB), i.e. the same P1-LL
  scope, but on a protocol that already handles the drift my version did
  not.
- Open risks to check before enabling: the one-shot kernel's vectorized
  16-byte peer loads over uncached host memory; `fully_connected=false`
  code paths inside the kernel; graph-capture registration
  (`register_graph_buffers`) with host pointers.

External confirmation that the stock path is unavailable on this topology
(not a local misconfiguration): vLLM disables custom all-reduce for >2
PCIe-only GPUs by design, and Blackwell PCIe boxes hang without
--disable-custom-all-reduce.
