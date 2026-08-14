"""Small adapter around NVIDIA nvCOMP's public Python Codec API."""

import torch


class NvcompANS:
    def __init__(self, stream):
        from nvidia import nvcomp

        self.nvcomp = nvcomp
        self.stream = stream
        self.codec = nvcomp.Codec(algorithm="ANS", cuda_stream=stream.cuda_stream)
        self._decode_configs = {}

    def compress(self, tensor):
        source = tensor.contiguous().view(torch.uint8).reshape(-1)
        config = self.codec.compression_config(source.numel())
        array = self.nvcomp.as_array(source, cuda_stream=self.stream.cuda_stream)
        encoded = self.codec.encode(array, compression_config=config)
        size = encoded.buffer_size
        return torch.from_dlpack(encoded.to_dlpack(cuda_stream=self.stream.cuda_stream))[:size].clone()

    def decompress_into(self, compressed, output):
        size = output.numel()
        config = self._decode_configs.get(size)
        if config is None:
            config = self.codec.decompression_config(self.codec.compression_config(size))
            self._decode_configs[size] = config
        source = self.nvcomp.as_array(compressed, cuda_stream=self.stream.cuda_stream)
        destination = self.nvcomp.as_array(output, cuda_stream=self.stream.cuda_stream)
        self.codec.decode(source, decompression_config=config, out=destination)


def availability():
    if not torch.cuda.is_available():
        return False, "CUDA unavailable"
    try:
        from nvidia import nvcomp  # noqa: F401
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"
    return True, "available"
