"""Control-flow regressions; no model download or GPU is required."""
import os
from pathlib import Path
import runpy
import unittest
from unittest.mock import MagicMock, patch


class AppTests(unittest.TestCase):
    def setUp(self):
        self.torch = MagicMock()
        self.torch.cuda.is_available.return_value = True
        self.gradio = MagicMock()
        self.gradio.Error = type("GradioError", (Exception,), {})
        self.diffusers = MagicMock()
        self.factory = self.diffusers.ZImagePipeline.from_pretrained
        self.pipeline = self.factory.return_value
        self.modules = patch.dict("sys.modules", {
            "torch": self.torch, "gradio": self.gradio,
            "diffusers": self.diffusers,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.env = patch.dict(os.environ, {"Z_IMAGE_OFFLOAD": "sequential"})
        self.env.start()
        self.addCleanup(self.env.stop)
        with patch("os.makedirs"):
            self.app = runpy.run_path(str(Path(__file__).parents[1] / "app.py"))

    def test_ui_import_does_not_load_model(self):
        self.factory.assert_not_called()
        self.gradio.Blocks.assert_called_once()

    def test_default_offload_never_moves_entire_pipeline_to_gpu(self):
        load = self.app["load_pipeline"]
        self.assertIs(load(), self.pipeline)
        self.assertIs(load(), self.pipeline)
        self.factory.assert_called_once()
        self.assertTrue(self.factory.call_args.kwargs["low_cpu_mem_usage"])
        self.pipeline.enable_sequential_cpu_offload.assert_called_once_with(device="cuda")
        self.pipeline.to.assert_not_called()

    def test_model_offload(self):
        os.environ["Z_IMAGE_OFFLOAD"] = "model"
        self.app["load_pipeline"]()
        self.pipeline.enable_model_cpu_offload.assert_called_once_with(device="cuda")
        self.pipeline.to.assert_not_called()

    def test_failed_transfer_is_retried(self):
        os.environ["Z_IMAGE_OFFLOAD"] = "none"
        self.pipeline.to.side_effect = [RuntimeError("out of memory"), None]
        load = self.app["load_pipeline"]
        with self.assertRaises(RuntimeError):
            load()
        self.assertIs(load(), self.pipeline)
        self.assertEqual(self.factory.call_count, 2)

    def test_invalid_offload_fails_before_download(self):
        os.environ["Z_IMAGE_OFFLOAD"] = "typo"
        with self.assertRaises(ValueError):
            self.app["load_pipeline"]()
        self.factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
