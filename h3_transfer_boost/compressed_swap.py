import contextvars
import inspect
import logging
import threading
import zlib

import torch

from .entropy import entropy_bits_per_byte, estimated_ratio
from .nvcomp import NvcompANS, availability


_ACTIVE = contextvars.ContextVar("h3_transfer_boost_active_swap", default=None)
_PATCH_LOCK = threading.Lock()
_ORIGINAL_CAST_TO_GATHERED = None


def _stream_context(stream):
    if hasattr(stream, "as_context"):
        return stream.as_context(stream)
    return torch.cuda.stream(stream)


def install_transfer_hook():
    global _ORIGINAL_CAST_TO_GATHERED
    with _PATCH_LOCK:
        import comfy.model_management as mm

        if getattr(mm.cast_to_gathered, "_h3_transfer_boost_hook", False):
            _ORIGINAL_CAST_TO_GATHERED = mm.cast_to_gathered._h3_transfer_boost_original
            return
        _ORIGINAL_CAST_TO_GATHERED = mm.cast_to_gathered

        def hooked(tensors, result, non_blocking=False, stream=None, r2=None):
            manager = _ACTIVE.get()
            if manager is None:
                return _ORIGINAL_CAST_TO_GATHERED(
                    tensors, result, non_blocking=non_blocking, stream=stream, r2=r2
                )
            return manager.cast_to_gathered(
                _ORIGINAL_CAST_TO_GATHERED,
                tensors,
                result,
                non_blocking=non_blocking,
                stream=stream,
                r2=r2,
            )

        hooked._h3_transfer_boost_hook = True
        hooked._h3_transfer_boost_original = _ORIGINAL_CAST_TO_GATHERED
        mm.cast_to_gathered = hooked


class CacheEntry:
    def __init__(self, host, uncompressed_bytes, ratio, ready_event):
        self.host = host
        self.uncompressed_bytes = uncompressed_bytes
        self.ratio = ratio
        self.ready_event = ready_event


class CompressedSwapManager:
    def __init__(self, min_tensor_mb, max_ratio, cache_limit_gb, entropy_sample_kib=256, fallback=True):
        self.minimum_bytes = int(min_tensor_mb * 1024 * 1024)
        self.max_ratio = float(max_ratio)
        self.cache_limit_bytes = int(cache_limit_gb * 1024 * 1024 * 1024)
        self.entropy_sample_bytes = int(entropy_sample_kib * 1024)
        self.fallback = bool(fallback)
        self.entries = {}
        self.rejected = set()
        self.codecs = {}
        self.staging = {}
        self.cache_bytes = 0
        self.raw_bytes = 0
        self.hits = 0
        self.misses = 0
        self.errors = 0
        self.disabled = False
        self._reported = False

    @staticmethod
    def _source_key(source):
        try:
            version = source._version
        except RuntimeError as error:
            if "do not track version counter" not in str(error):
                raise
            sample_count = min(256, source.numel())
            stride = max(1, source.numel() // sample_count)
            sample = bytes(source[::stride][:sample_count].tolist())
            version = ("crc32", zlib.crc32(sample))
        return (id(source), source.data_ptr(), source.numel(), version)

    def _eligible(self, tensors, result, stream, r2):
        if self.disabled:
            return None
        if stream is None or r2 is not None or result is None or len(tensors) != 1:
            return None
        source = tensors[0]
        if not isinstance(source, torch.Tensor) or source.device.type != "cpu":
            return None
        if source.dim() != 1 or source.element_size() != 1 or source.numel() < self.minimum_bytes:
            return None
        if not isinstance(result, torch.Tensor) or result.device.type != "cuda":
            return None
        if result.dim() != 1 or result.element_size() != 1 or result.numel() != source.numel():
            return None
        return source

    def _codec(self, stream):
        key = (str(stream.device), stream.cuda_stream)
        codec = self.codecs.get(key)
        if codec is None:
            codec = NvcompANS(stream)
            self.codecs[key] = codec
        return codec

    def _stage(self, stream, size, device):
        key = (str(device), stream.cuda_stream)
        stage = self.staging.get(key)
        if stage is None or stage.numel() < size:
            stage = torch.empty(size, dtype=torch.uint8, device=device)
            self.staging[key] = stage
        return stage[:size]

    def _entropy_ratio(self, source):
        stride = max(1, source.numel() // self.entropy_sample_bytes)
        sample = source[::stride][:self.entropy_sample_bytes].tolist()
        return estimated_ratio(entropy_bits_per_byte(sample))

    def _drop_stale(self, key):
        identity = key[:3]
        stale = [old for old in self.entries if old[:3] == identity and old != key]
        for old in stale:
            entry = self.entries.pop(old)
            entry.ready_event.synchronize()
            self.cache_bytes -= entry.host.numel()
            self.raw_bytes -= entry.uncompressed_bytes
        self.rejected = {old for old in self.rejected if old[:3] != identity or old == key}

    def _build(self, key, source, result, stream):
        if self._entropy_ratio(source) > self.max_ratio:
            self.rejected.add(key)
            return
        codec = self._codec(stream)
        with _stream_context(stream):
            compressed_gpu = codec.compress(result)
            compressed_bytes = compressed_gpu.numel()
            ratio = compressed_bytes / result.numel()
            if ratio > self.max_ratio or self.cache_bytes + compressed_bytes > self.cache_limit_bytes:
                self.rejected.add(key)
                return
            compressed_host = torch.empty(compressed_bytes, dtype=torch.uint8, pin_memory=True)
            compressed_host.copy_(compressed_gpu, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(stream)
        self.entries[key] = CacheEntry(compressed_host, result.numel(), ratio, ready)
        self.cache_bytes += compressed_bytes
        self.raw_bytes += result.numel()
        self.misses += 1

    def _decode(self, entry, result, stream):
        stream.wait_event(entry.ready_event)
        stage = self._stage(stream, entry.host.numel(), result.device)
        codec = self._codec(stream)
        with _stream_context(stream):
            stage.copy_(entry.host, non_blocking=True)
            codec.decompress_into(stage, result)
        self.hits += 1

    def cast_to_gathered(self, original, tensors, result, non_blocking=False, stream=None, r2=None):
        source = self._eligible(tensors, result, stream, r2)
        if source is None:
            return original(tensors, result, non_blocking=non_blocking, stream=stream, r2=r2)
        key = self._source_key(source)
        self._drop_stale(key)
        entry = self.entries.get(key)
        if entry is not None:
            try:
                self._decode(entry, result, stream)
                return None
            except Exception:
                self.errors += 1
                if self.errors >= 3:
                    self.disabled = True
                if not self.fallback:
                    raise
                logging.exception("H3 compressed swap decode failed; using normal transfer")
                try:
                    stream.synchronize()
                except Exception:
                    pass
                old = self.entries.pop(key)
                self.cache_bytes -= old.host.numel()
                self.raw_bytes -= old.uncompressed_bytes
                self.rejected.add(key)

        output = original(tensors, result, non_blocking=non_blocking, stream=stream, r2=r2)
        if key not in self.rejected:
            try:
                self._build(key, source, result, stream)
            except Exception:
                self.errors += 1
                if self.errors >= 3:
                    self.disabled = True
                self.rejected.add(key)
                if not self.fallback:
                    raise
                logging.exception("H3 compressed swap cache build failed; keeping normal transfer")
        return output

    def activate(self):
        return _ACTIVE.set(self)

    @staticmethod
    def deactivate(token):
        _ACTIVE.reset(token)

    def report(self):
        ratio = self.cache_bytes / self.raw_bytes if self.raw_bytes else None
        return {
            "cached_tensors": len(self.entries),
            "rejected_tensors": len(self.rejected),
            "raw_cache_gib": round(self.raw_bytes / 1073741824, 3),
            "compressed_cache_gib": round(self.cache_bytes / 1073741824, 3),
            "measured_ratio": round(ratio, 3) if ratio is not None else None,
            "compressed_transfer_hits": self.hits,
            "cache_builds": self.misses,
            "errors": self.errors,
            "disabled_after_errors": self.disabled,
        }

    def log_report(self):
        report = self.report()
        if report["cached_tensors"] or (not self._reported and report["rejected_tensors"]):
            logging.info("H3 compressed swap: %s", report)
            self._reported = True

    def cleanup(self):
        for entry in self.entries.values():
            entry.ready_event.synchronize()
        self.entries.clear()
        self.staging.clear()
        self.codecs.clear()
        self.rejected.clear()
        self.cache_bytes = 0
        self.raw_bytes = 0


def check_runtime():
    ok, message = availability()
    if not ok:
        raise RuntimeError(f"nvCOMP compressed swap unavailable: {message}")
    import comfy.model_management as mm

    target = getattr(mm.cast_to_gathered, "_h3_transfer_boost_original", mm.cast_to_gathered)
    parameters = inspect.signature(target).parameters
    if not {"tensors", "r", "non_blocking", "stream", "r2"}.issubset(parameters):
        raise RuntimeError("当前 ComfyUI 的 cast_to_gathered 接口不兼容；请更新 ComfyUI 或关闭压缩交换")
    install_transfer_hook()
