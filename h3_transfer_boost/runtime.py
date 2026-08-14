import threading

import torch


_LOCK = threading.RLock()


def _call(wrapper, apply_model, args):
    if wrapper is not None:
        return wrapper(apply_model, args)
    return apply_model(args["input"], args["timestep"], **args["c"])


class AsyncOffloadWrapper:
    """Apply a ComfyUI async-offload stream count only while this model runs."""

    def __init__(self, streams, previous=None):
        self.streams = int(streams)
        self.previous = previous
        self._pools = {}

    def _pool(self, device):
        pool = self._pools.get(device)
        if pool is None:
            pool = []
            for _ in range(self.streams):
                stream = torch.cuda.Stream(device=device, priority=0)
                stream.as_context = torch.cuda.stream
                pool.append(stream)
            self._pools[device] = pool
        return pool

    def __call__(self, apply_model, args):
        import comfy.model_management as mm

        with _LOCK:
            old_streams = mm.NUM_STREAMS
            mm.NUM_STREAMS = self.streams
            device = args["input"].device
            previous_pool = mm.STREAMS.get(device)
            previous_counter = mm.stream_counters.get(device)
            try:
                if torch.cuda.is_available() and device.type == "cuda":
                    mm.STREAMS[device] = self._pool(device)
                    mm.stream_counters[device] = 0
                return _call(self.previous, apply_model, args)
            finally:
                if previous_pool is None:
                    mm.STREAMS.pop(device, None)
                else:
                    mm.STREAMS[device] = previous_pool
                if previous_counter is None:
                    mm.stream_counters.pop(device, None)
                else:
                    mm.stream_counters[device] = previous_counter
                mm.NUM_STREAMS = old_streams

    def to(self, _device):
        return self


class CompressedSwapWrapper(AsyncOffloadWrapper):
    def __init__(self, streams, manager, previous=None):
        super().__init__(streams, previous)
        self.manager = manager

    def __call__(self, apply_model, args):
        self.manager.begin_call()
        token = self.manager.activate()
        try:
            return super().__call__(apply_model, args)
        finally:
            self.manager.deactivate(token)
            self.manager.log_report()

    def cleanup(self):
        self.manager.cleanup()

    def stats(self):
        return self.manager.report()
