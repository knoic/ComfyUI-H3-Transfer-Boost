import torch

from .entropy import entropy_bits_per_byte, estimated_ratio


ONE_BYTE_DTYPES = {
    torch.int8,
    torch.uint8,
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

    for name, parameter in diffusion.named_parameters():
        if len(rows) >= max_tensors:
            break
        if not isinstance(parameter, torch.Tensor):
            continue
        size = parameter.numel() * parameter.element_size()
        if size < minimum or parameter.element_size() != 1 or parameter.dtype not in ONE_BYTE_DTYPES:
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

    return {
        "eligible_tensors": len(rows),
        "eligible_gib": round(total_bytes / 1073741824, 3),
        "estimated_compressed_gib": round(estimated_bytes / 1073741824, 3),
        "estimated_saved_percent": round(100 * (1 - estimated_bytes / total_bytes), 1) if total_bytes else 0.0,
        "method": "Shannon entropy estimate; run NVCOMP Benchmark for measured device throughput",
        "tensors": rows,
    }
