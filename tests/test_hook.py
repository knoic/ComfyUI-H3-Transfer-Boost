import importlib
import sys
import types
import unittest


if "torch" not in sys.modules:
    try:
        import torch  # noqa: F401
    except ImportError:
        sys.modules["torch"] = types.ModuleType("torch")


class TransferHookTests(unittest.TestCase):
    def test_safe_mode_caps_requested_memory_budget(self):
        module = importlib.import_module("h3_transfer_boost.compressed_swap")
        manager = module.CompressedSwapManager(
            min_tensor_mb=8,
            max_ratio=0.8,
            cache_limit_gb=8,
            max_entry_mb=1024,
            warmup_budget_mb=1024,
            safe_mode=True,
        )
        self.assertEqual(manager.cache_limit_bytes, 1024 * 1024 * 1024)
        self.assertEqual(manager.maximum_entry_bytes, 256 * 1024 * 1024)
        self.assertEqual(manager.warmup_budget_bytes, 256 * 1024 * 1024)

    def test_inference_tensor_key_uses_content_fingerprint(self):
        module = importlib.import_module("h3_transfer_boost.compressed_swap")

        class Sample:
            def __getitem__(self, _item):
                return self

            def tolist(self):
                return [1, 2, 3, 4]

        class InferenceTensor:
            @property
            def _version(self):
                raise RuntimeError("Inference tensors do not track version counter.")

            def numel(self):
                return 4

            def data_ptr(self):
                return 1234

            def __getitem__(self, _item):
                return Sample()

        tensor = InferenceTensor()
        key = module.CompressedSwapManager._source_key(tensor)
        self.assertEqual(key[:3], (id(tensor), 1234, 4))
        self.assertEqual(key[3][0], "crc32")

    def test_hook_dispatches_only_while_active(self):
        calls = []

        def original(*args, **kwargs):
            calls.append((args, kwargs))
            return "original"

        model_management = types.ModuleType("comfy.model_management")
        model_management.cast_to_gathered = original
        comfy = types.ModuleType("comfy")
        comfy.model_management = model_management
        old_comfy = sys.modules.get("comfy")
        old_mm = sys.modules.get("comfy.model_management")
        sys.modules["comfy"] = comfy
        sys.modules["comfy.model_management"] = model_management
        try:
            module = importlib.import_module("h3_transfer_boost.compressed_swap")
            module._ORIGINAL_CAST_TO_GATHERED = None
            module.install_transfer_hook()
            self.assertEqual(model_management.cast_to_gathered([], None), "original")

            class Manager:
                def cast_to_gathered(self, original_fn, *args, **kwargs):
                    self.original = original_fn
                    return "compressed"

            manager = Manager()
            token = module._ACTIVE.set(manager)
            try:
                self.assertEqual(model_management.cast_to_gathered([], None), "compressed")
                self.assertIs(manager.original, original)
            finally:
                module._ACTIVE.reset(token)

            module.install_transfer_hook()
            self.assertIs(module._ORIGINAL_CAST_TO_GATHERED, original)
            self.assertEqual(len(calls), 1)
        finally:
            if old_comfy is None:
                sys.modules.pop("comfy", None)
            else:
                sys.modules["comfy"] = old_comfy
            if old_mm is None:
                sys.modules.pop("comfy.model_management", None)
            else:
                sys.modules["comfy.model_management"] = old_mm


if __name__ == "__main__":
    unittest.main()
