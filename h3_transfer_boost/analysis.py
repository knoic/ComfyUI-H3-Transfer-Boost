import torch

from .entropy import entropy_bits_per_byte, estimated_ratio


SUPPORTED_DTYPES = {
    torch.int8,
    torch.uint8,
    torch.bfloat16,
    torch.float16,
    getattr(torch, "float8_e4m3fn", torch.int8),
    getattr(torch, "float8_e5m2", torch.int8),
}


def _bytes_view(tensor):
    return tensor.detach().contiguous().view(torch.uint8).reshape(-1)


def _diffusion_model(model):
    current = model
    for name in ("model", "diffusion_model"):
        current = getattr(current, name, None)
        if current is None:
            raise ValueError("输入不是可识别的 ComfyUI MODEL（缺少 model.diffusion_model）")
    return current


def analyze_model(model, min_tensor_mb=1.0, sample_kib=256, max_tensors=512):
    diffusion = _diffusion_model(model)
    minimum = int(min_tensor_mb * 1024 * 1024)
    sample_bytes = max(4096, int(sample_kib * 1024))
    rows = []
    total_bytes = 0
    estimated_bytes = 0
    by_dtype = {}

    for name, parameter in diffusion.named_parameters():
        if len(rows) >= max_tensors:
            break
        if not isinstance(parameter, torch.Tensor):
            continue
        size = parameter.numel() * parameter.element_size()
        if size < minimum or parameter.dtype not in SUPPORTED_DTYPES:
            continue
        try:
            flat = _bytes_view(parameter)
            stride = max(1, flat.numel() // sample_bytes)
            sample = flat[::stride][:sample_bytes].cpu().tolist()
        except (RuntimeError, TypeError, NotImplementedError):
            continue
        entropy = entropy_bits_per_byte(sample)
        ratio = estimated_ratio(entropy)
        rows.append({
            "name": name,
            "dtype": str(parameter.dtype).replace("torch.", ""),
            "mib": round(size / 1048576, 2),
            "entropy_bits": round(entropy, 3),
            "estimated_ratio": round(ratio, 3),
        })
        total_bytes += size
        estimated_bytes += int(size * ratio)
        dtype_name = str(parameter.dtype).replace("torch.", "")
        dtype_stats = by_dtype.setdefault(dtype_name, {"tensors": 0, "bytes": 0, "estimated_bytes": 0})
        dtype_stats["tensors"] += 1
        dtype_stats["bytes"] += size
        dtype_stats["estimated_bytes"] += int(size * ratio)

    dtype_report = {}
    for dtype_name, stats in by_dtype.items():
        dtype_report[dtype_name] = {
            "tensors": stats["tensors"],
            "gib": round(stats["bytes"] / 1073741824, 3),
            "estimated_ratio": round(stats["estimated_bytes"] / stats["bytes"], 3),
        }

    return {
        "eligible_tensors": len(rows),
        "eligible_gib": round(total_bytes / 1073741824, 3),
        "estimated_compressed_gib": round(estimated_bytes / 1073741824, 3),
        "estimated_saved_percent": round(100 * (1 - estimated_bytes / total_bytes), 1) if total_bytes else 0.0,
        "by_dtype": dtype_report,
        "method": "Raw-byte Shannon entropy estimate for INT8/FP8/BF16/FP16; runtime uses measured ANS ratio",
        "tensors": rows,
    }
