import json
import torch

from .h3_transfer_boost.analysis import analyze_model
from .h3_transfer_boost.compressed_swap import CompressedSwapManager, check_runtime
from .h3_transfer_boost.runtime import AsyncOffloadWrapper, CompressedSwapWrapper


class H3TransferAnalyze:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "min_tensor_mb": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1024.0, "step": 0.1}),
            "sample_kib": ("INT", {"default": 256, "min": 4, "max": 4096}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report_json",)
    FUNCTION = "run"
    CATEGORY = "H3/Transfer Boost"
    OUTPUT_NODE = True

    def run(self, model, min_tensor_mb, sample_kib):
        report = analyze_model(model, min_tensor_mb, sample_kib)
        text = json.dumps(report, ensure_ascii=False, indent=2)
        return {"ui": {"text": (text,)}, "result": (text,)}


class H3AsyncOffloadTune:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "streams": ("INT", {"default": 3, "min": 1, "max": 8}),
        }}

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "status")
    FUNCTION = "run"
    CATEGORY = "H3/Transfer Boost"

    def run(self, model, streams):
        tuned = model.clone()
        previous = tuned.model_options.get("model_function_wrapper")
        tuned.set_model_unet_function_wrapper(AsyncOffloadWrapper(streams, previous))
        status = (
            f"本模型推理期间使用 {streams} 个异步卸载流。"
            "建议分别测试 2/3/4；更多流会占用更多显存，且不一定更快。"
        )
        return (tuned, status)


class H3CompressedSwap:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "streams": ("INT", {"default": 3, "min": 1, "max": 8}),
            "min_tensor_mb": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 1024.0, "step": 1.0}),
            "max_ratio": ("FLOAT", {"default": 0.80, "min": 0.10, "max": 1.0, "step": 0.01}),
            "cache_limit_gb": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 128.0, "step": 0.25}),
            "max_entry_mb": ("FLOAT", {"default": 256.0, "min": 16.0, "max": 4096.0, "step": 16.0}),
            "warmup_budget_mb": ("FLOAT", {"default": 256.0, "min": 16.0, "max": 4096.0, "step": 16.0}),
            "safe_mode": ("BOOLEAN", {"default": True}),
            "fallback_on_error": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "status")
    FUNCTION = "run"
    CATEGORY = "H3/Transfer Boost"

    def run(self, model, streams, min_tensor_mb, max_ratio, cache_limit_gb, max_entry_mb, warmup_budget_mb, safe_mode, fallback_on_error):
        check_runtime()
        tuned = model.clone()
        previous = tuned.model_options.get("model_function_wrapper")
        if isinstance(previous, AsyncOffloadWrapper):
            previous = previous.previous
        manager = CompressedSwapManager(
            min_tensor_mb=min_tensor_mb,
            max_ratio=max_ratio,
            cache_limit_gb=cache_limit_gb,
            max_entry_mb=max_entry_mb,
            warmup_budget_mb=warmup_budget_mb,
            safe_mode=safe_mode,
            fallback=fallback_on_error,
        )
        tuned.set_model_unet_function_wrapper(CompressedSwapWrapper(streams, manager, previous))
        status = (
            "实验性 nvCOMP ANS 压缩交换已启用。安全模式会限制额外缓存至 1 GiB、单次压缩至 256 MiB，"
            "并逐步预热；统计信息输出到 ComfyUI 日志。不要再串联 H3 Async Offload Tuner。"
        )
        return (tuned, status)


class H3CompressedSwapStats:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return float("nan")

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report_json",)
    FUNCTION = "run"
    CATEGORY = "H3/Transfer Boost"
    OUTPUT_NODE = True

    def run(self, model):
        wrapper = model.model_options.get("model_function_wrapper")
        while wrapper is not None and not isinstance(wrapper, CompressedSwapWrapper):
            wrapper = getattr(wrapper, "previous", None)
        if wrapper is None:
            report = {"active": False, "error": "MODEL 未经过 H3 Compressed Swap"}
        else:
            report = {"active": True, **wrapper.stats()}
        text = json.dumps(report, ensure_ascii=False, indent=2)
        return {"ui": {"text": (text,)}, "result": (text,)}


class H3TransferBenchmark:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "size_mb": ("INT", {"default": 256, "min": 16, "max": 2048}),
            "iterations": ("INT", {"default": 8, "min": 2, "max": 100}),
            "data_pattern": (["h3_like", "bf16_like", "low_entropy", "random"],),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report_json",)
    FUNCTION = "run"
    CATEGORY = "H3/Transfer Boost"
    OUTPUT_NODE = True

    def run(self, size_mb, iterations, data_pattern):
        if not torch.cuda.is_available():
            raise RuntimeError("此基准需要 NVIDIA CUDA GPU")
        count = size_mb * 1024 * 1024
        if data_pattern == "random":
            host = torch.randint(0, 256, (count,), dtype=torch.uint8, pin_memory=True)
        elif data_pattern == "low_entropy":
            host = torch.randint(0, 16, (count,), dtype=torch.uint8, pin_memory=True)
        elif data_pattern == "h3_like":
            host = torch.empty(count, dtype=torch.uint8, pin_memory=True)
            chunk = 16 * 1024 * 1024
            for start in range(0, count, chunk):
                length = min(chunk, count - start)
                values = torch.randn(length).mul_(28).add_(128).clamp_(0, 255)
                host[start:start + length].copy_(values.to(torch.uint8))
        else:
            host = torch.empty(count, dtype=torch.uint8, pin_memory=True)
            chunk = 16 * 1024 * 1024
            for start in range(0, count, chunk):
                length = min(chunk, count - start)
                values = torch.randn(length // 2, dtype=torch.bfloat16)
                host[start:start + length].copy_(values.view(torch.uint8))
        device = torch.empty_like(host, device="cuda")
        stream = torch.cuda.Stream()
        event_start = torch.cuda.Event(enable_timing=True)
        event_end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            device.copy_(host, non_blocking=True)
        stream.synchronize()
        with torch.cuda.stream(stream):
            event_start.record(stream)
            for _ in range(iterations):
                device.copy_(host, non_blocking=True)
            event_end.record(stream)
        event_end.synchronize()
        milliseconds = event_start.elapsed_time(event_end)
        gbps = (size_mb / 1024 * iterations) / (milliseconds / 1000)
        sample = host[::max(1, host.numel() // 262144)][:262144].tolist()
        from .h3_transfer_boost.entropy import entropy_bits_per_byte, estimated_ratio
        entropy = entropy_bits_per_byte(sample)
        ratio = estimated_ratio(entropy)
        report = {
            "pinned_h2d_gib_s": round(gbps, 2),
            "entropy_bits": round(entropy, 3),
            "estimated_ANS_ratio": round(ratio, 3),
            "estimated_compressed_h2d_gib_s": round(gbps / ratio, 2) if ratio else None,
            "note": "压缩吞吐是基于熵下界的上限估计，不包含 nvCOMP 解压开销。",
        }
        from .h3_transfer_boost.nvcomp import NvcompANS, availability
        nvcomp_ok, nvcomp_status = availability()
        report["nvcomp"] = nvcomp_status
        if nvcomp_ok:
            with torch.cuda.stream(stream):
                codec = NvcompANS(stream)
                compressed_gpu = codec.compress(device)
            stream.synchronize()
            compressed_size = compressed_gpu.numel()
            compressed_host = torch.empty(compressed_size, dtype=torch.uint8, pin_memory=True)
            compressed_host.copy_(compressed_gpu, non_blocking=False)
            compressed_stage = torch.empty_like(compressed_gpu)
            decoded = torch.empty_like(device)
            with torch.cuda.stream(stream):
                compressed_stage.copy_(compressed_host, non_blocking=True)
                codec.decompress_into(compressed_stage, decoded)
            stream.synchronize()
            if not torch.equal(decoded, device):
                raise RuntimeError("nvCOMP ANS 往返校验失败")
            with torch.cuda.stream(stream):
                event_start.record(stream)
                for _ in range(iterations):
                    compressed_stage.copy_(compressed_host, non_blocking=True)
                    codec.decompress_into(compressed_stage, decoded)
                event_end.record(stream)
            event_end.synchronize()
            compressed_ms = event_start.elapsed_time(event_end)
            effective = (size_mb / 1024 * iterations) / (compressed_ms / 1000)
            report.update({
                "measured_ANS_ratio": round(compressed_size / count, 3),
                "compressed_h2d_plus_decode_effective_gib_s": round(effective, 2),
                "projected_transfer_speedup": round(effective / gbps, 3),
            })
        text = json.dumps(report, ensure_ascii=False, indent=2)
        return {"ui": {"text": (text,)}, "result": (text,)}


NODE_CLASS_MAPPINGS = {
    "H3TransferAnalyze": H3TransferAnalyze,
    "H3AsyncOffloadTune": H3AsyncOffloadTune,
    "H3CompressedSwap": H3CompressedSwap,
    "H3CompressedSwapStats": H3CompressedSwapStats,
    "H3TransferBenchmark": H3TransferBenchmark,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3TransferAnalyze": "H3 Weight Compressibility Analyzer",
    "H3AsyncOffloadTune": "H3 Async Offload Tuner",
    "H3CompressedSwap": "H3 Compressed Swap (Experimental)",
    "H3CompressedSwapStats": "H3 Compressed Swap Stats",
    "H3TransferBenchmark": "H3 PCIe Transfer Benchmark",
}
