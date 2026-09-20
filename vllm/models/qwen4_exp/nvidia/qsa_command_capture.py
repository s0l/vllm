# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Selective QSA command capture for admitted single-token metadata."""

import dataclasses
import inspect
from typing import Any, cast

import torch

from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.models.qwen4_exp.nvidia.qsa import _qsa_fp8_owner


def tensor_signature(value):
    return tuple(value.shape), tuple(value.stride()), value.dtype, value.device


class MetadataBinding:
    def __init__(self, layer, args, shared):
        self.layer = layer
        self.names = (
            layer.layer_name,
            layer.indexer.raw_key_cache.prefix,
            layer.indexer.compressed_key_cache.prefix,
        )
        self.operands = args
        self.backings = self.backing_signature()
        self.skip_topk = layer.indexer.skip_topk
        self.geometry = {}
        self.static = {}
        current = get_forward_context().attn_metadata
        if not isinstance(current, dict):
            raise ValueError("QSA requires dictionary metadata")
        for name in self.names:
            value = current[name]
            self.geometry[name] = (
                getattr(value, "num_decodes", None),
                getattr(value, "num_prefills", None),
                getattr(value, "decode_query_len", None),
                getattr(value, "num_actual_tokens", None),
            )
            if id(value) in shared:
                previous, frozen = shared[id(value)]
                assert previous is value
                self.static[name] = frozen
                continue
            fields = {}
            for field in dataclasses.fields(cast(Any, value)):
                v = getattr(value, field.name)
                fields[field.name] = (
                    None
                    if field.name == "draft_common"
                    else v.clone()
                    if isinstance(v, torch.Tensor)
                    else v
                )
            self.static[name] = type(value)(**fields)
            shared[id(value)] = (value, self.static[name])
        self.pairs(current)

    def backing_signature(self):
        return tuple(
            (t.data_ptr(), t.numel(), tensor_signature(t))
            for t in (
                self.layer.kv_cache_storage,
                self.layer.indexer.raw_key_cache.kv_cache_storage,
                self.layer.indexer.compressed_key_cache.kv_cache_storage,
            )
        )

    def pairs(self, current, seen=None):
        if self.backing_signature() != self.backings:
            raise ValueError("QSA backing changed; invalidate capture before replay")
        if self.layer.indexer.skip_topk != self.skip_topk:
            raise ValueError("QSA selection policy changed; invalidate capture")
        seen = {} if seen is None else seen
        pairs = []
        for name in self.names:
            src, dst = current[name], self.static[name]
            if id(dst) in seen:
                if seen[id(dst)] is not src:
                    raise ValueError("QSA metadata alias grouping changed")
                continue
            seen[id(dst)] = src
            if type(src) is not type(dst):
                raise ValueError("QSA metadata type changed")
            geometry = (
                getattr(src, "num_decodes", None),
                getattr(src, "num_prefills", None),
                getattr(src, "decode_query_len", None),
                getattr(src, "num_actual_tokens", None),
            )
            if geometry != self.geometry[name]:
                raise ValueError("QSA capture geometry changed: " + name)
            for field in dataclasses.fields(dst):
                key = field.name
                a, b = getattr(dst, key), getattr(src, key)
                if key in ("draft_common", "max_seq_len"):
                    continue
                if isinstance(a, torch.Tensor):
                    if not isinstance(b, torch.Tensor) or tensor_signature(
                        a
                    ) != tensor_signature(b):
                        raise ValueError(
                            "QSA metadata tensor signature changed: " + key
                        )
                    pairs.append((a, b))
                elif a != b:
                    raise ValueError("QSA capture scalar changed: " + key)
        return pairs

    def replay(self, graph):
        pairs = self.pairs(get_forward_context().attn_metadata)
        for dst, src in pairs:
            dst.copy_(src)
        graph.replay()


class QSACommandCapture(BreakableCUDAGraphCapture):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qsa_bindings = []
        self.shared_metadata = {}
        self.last_staging = {}

    def add_eager(self, callback):
        closure = inspect.getclosurevars(callback).nonlocals
        if closure.get("fn") is not getattr(_qsa_fp8_owner, "__wrapped__", None):
            return super().add_eager(callback)
        operands = closure["weak_args"]
        context = get_forward_context()
        owner = context.no_compile_layers[operands[-1]]
        metadata = context.attn_metadata
        names = (
            owner.layer_name,
            owner.indexer.raw_key_cache.prefix,
            owner.indexer.compressed_key_cache.prefix,
        )
        if not isinstance(metadata, dict) or any(
            name not in metadata for name in names
        ):
            return super().add_eager(callback)
        # Capture only the measured single-request target q1/q2 contracts.
        # MTP keeps its existing eager boundary and larger target geometries
        # fail closed until they have their own response-surface proof.
        main = metadata[names[0]]
        actual_tokens = getattr(main, "num_actual_tokens", None)

        def admitted(value):
            return (
                getattr(value, "num_actual_tokens", actual_tokens) == actual_tokens
                and (not hasattr(value, "num_decodes") or value.num_decodes == 1)
                and (not hasattr(value, "num_prefills") or value.num_prefills == 0)
                and (
                    not hasattr(value, "decode_query_len")
                    or value.decode_query_len == actual_tokens
                )
            )

        if (
            ".mtp." in "." + owner.layer_name
            or actual_tokens not in (1, 2)
            or any(not admitted(metadata[n]) for n in names)
        ):
            return super().add_eager(callback)
        self._end_segment()
        binding = MetadataBinding(owner, operands, self.shared_metadata)
        original = context.attn_metadata
        if not isinstance(original, dict):
            return super().add_eager(callback)
        captured = dict(original)
        captured.update(binding.static)
        graph = torch.cuda.CUDAGraph()
        context.attn_metadata = captured
        try:
            graph.capture_begin(
                pool=self.pool
            ) if self.pool is not None else graph.capture_begin()
            callback()
            graph.capture_end()
        finally:
            context.attn_metadata = original
        self.graphs.append(graph)
        self._num_graphs += 1
        self.qsa_bindings.append(binding)
        self.segments.append(graph.replay)
        self._begin_segment()
        from vllm.logger import init_logger

        init_logger(__name__).info_once(
            "FlashNext QSA command capture active: x1/q%d; mutable metadata staged",
            actual_tokens,
        )
        return None

    def replay(self):
        # Whole-DAG preflight must finish before any GPU/KV mutation. Sharing
        # follows the actual builder's object aliases, never equal-looking data.
        if not is_forward_context_available():
            raise RuntimeError(
                "QSA replay requires request or manager-restored ForwardContext"
            )
        current = get_forward_context().attn_metadata
        seen: dict[int, Any] = {}
        pairs: list[Any] = []
        for binding in self.qsa_bindings:
            pairs.extend(binding.pairs(current, seen))
        self.last_staging = dict(
            mode="forward_context",
            groups=len(seen),
            copies=len(pairs),
            bytes=sum(a.numel() * a.element_size() for a, _ in pairs),
        )
        for dst, src in pairs:
            dst.copy_(src)
        super().replay()

    def reset(self):
        super().reset()
        self.qsa_bindings.clear()
        self.shared_metadata.clear()
