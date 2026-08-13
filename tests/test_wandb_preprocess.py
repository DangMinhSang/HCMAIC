from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run


CODE = Path(__file__).resolve().parents[1] / "Code"
if str(CODE) not in sys.path:
    sys.path.append(str(CODE))

from preprocess_direct_video import VideoWindow, WandbPreprocessTracker


def tracker_arguments() -> SimpleNamespace:
    return SimpleNamespace(
        sample_fps=0.0,
        max_side=0,
        gpus="0,1",
        workers=2,
        clip_model="ViT-B/32",
        clip_batch=64,
        object_model="yolo11m.pt",
        object_batch=16,
        object_confidence=0.2,
        mask_clip_overlays=True,
        ocr_language="vi",
        ocr_device="gpu:0",
        ocr_min_confidence=0.45,
        force=False,
    )


class FakeRun:
    def __init__(self) -> None:
        self.id = "fixed-run"
        self.url = "https://wandb.ai/example/run/fixed-run"
        self.logs: list[dict] = []
        self.summary: dict = {}
        self.exit_code = None

    def log(self, metrics: dict) -> None:
        self.logs.append(metrics)

    def finish(self, *, exit_code: int) -> None:
        self.exit_code = exit_code


class FakeWandb:
    def __init__(self, fake_run: FakeRun | None = None, error: Exception | None = None) -> None:
        self.run = fake_run or FakeRun()
        self.error = error
        self.init_arguments: dict = {}
        self.settings_arguments: dict = {}

    def Settings(self, **arguments):
        self.settings_arguments = arguments
        return SimpleNamespace(**arguments)

    def init(self, **arguments):
        self.init_arguments = arguments
        if self.error is not None:
            raise self.error
        return self.run


class WandbPreprocessTests(unittest.TestCase):
    def test_launcher_accepts_wandb_api_key_option(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["run.py", "--pre-direct-video", "1", "--wandb-api-key", "secret-api-key"],
        ):
            arguments = run.parse_arguments()
        self.assertEqual(arguments.wandb_api_key, "secret-api-key")

    def test_launcher_builds_shared_secret_safe_stage_environment(self) -> None:
        arguments = SimpleNamespace(
            wandb_api_key="secret-api-key",
            start_pre_video=1,
            end_pre_video=25,
        )
        with patch.dict(
            os.environ,
            {"AIC_WANDB_PROJECT": "aic-test", "AIC_WANDB_RUN_ID": "shared-run"},
            clear=True,
        ):
            environment = run.build_wandb_preprocess_environment(arguments)

        self.assertEqual(environment["AIC_WANDB_ENABLED"], "1")
        self.assertEqual(environment["AIC_WANDB_RUN_ID"], "shared-run")
        self.assertEqual(environment["AIC_WANDB_PROJECT"], "aic-test")
        self.assertEqual(environment["WANDB_API_KEY"], "secret-api-key")
        self.assertEqual(environment["WANDB_CONSOLE"], "off")

        arguments.wandb_api_key = ""
        self.assertEqual(
            run.build_wandb_preprocess_environment(arguments),
            {"AIC_WANDB_ENABLED": "0"},
        )

    def test_parent_tracker_logs_video_manifest_and_resumes_run(self) -> None:
        fake_wandb = FakeWandb()
        window = VideoWindow((Path("L21_V001.mp4"),), 1, 1, 1)
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ,
            {
                "AIC_WANDB_ENABLED": "1",
                "AIC_WANDB_RUN_ID": "fixed-run",
                "AIC_WANDB_PROJECT": "aic-test",
                "AIC_WANDB_DIR": temporary,
                "WANDB_API_KEY": "secret-api-key",
            },
            clear=True,
        ), patch.dict(sys.modules, {"wandb": fake_wandb}):
            tracker = WandbPreprocessTracker(
                stage="visual",
                window=window,
                output_root=Path(temporary) / "output",
                arguments=tracker_arguments(),
                dataset_root=Path("/kaggle/input/video-aic"),
                source_kind="mounted",
            )
            tracker.log_video(
                {
                    "stage": "visual",
                    "ordinal": 1,
                    "video_id": "L21_V001",
                    "gpu_id": "1",
                    "frames": 120,
                    "seconds": 12.5,
                    "skipped": False,
                    "timing_seconds": {
                        "decode": 2.0,
                        "prepare": 1.0,
                        "clip": 4.0,
                        "object": 5.0,
                    },
                }
            )
            tracker.log_manifest(
                {
                    "visual_videos": 1,
                    "all_frame_videos": 1,
                    "complete_videos": 1,
                    "ocr_records": 10,
                    "object_records": 120,
                }
            )
            tracker.finish(0)

        self.assertEqual(fake_wandb.init_arguments["id"], "fixed-run")
        self.assertEqual(fake_wandb.init_arguments["resume"], "allow")
        self.assertNotIn("secret-api-key", repr(fake_wandb.init_arguments))
        self.assertEqual(fake_wandb.settings_arguments["console"], "off")
        self.assertEqual(fake_wandb.run.logs[0]["visual/decode_seconds"], 2.0)
        self.assertEqual(fake_wandb.run.logs[0]["visual/clip_seconds"], 4.0)
        self.assertEqual(fake_wandb.run.logs[0]["video/gpu_id"], "1")
        self.assertEqual(fake_wandb.run.logs[1]["finalize/complete_videos"], 1)
        self.assertEqual(fake_wandb.run.exit_code, 0)

    def test_wandb_failure_is_redacted_and_does_not_abort(self) -> None:
        fake_wandb = FakeWandb(error=RuntimeError("bad secret-api-key"))
        window = VideoWindow((Path("L21_V001.mp4"),), 1, 1, 1)
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ,
            {
                "AIC_WANDB_ENABLED": "1",
                "AIC_WANDB_DIR": temporary,
                "WANDB_API_KEY": "secret-api-key",
            },
            clear=True,
        ), patch.dict(sys.modules, {"wandb": fake_wandb}), redirect_stderr(stderr):
            tracker = WandbPreprocessTracker(
                stage="visual",
                window=window,
                output_root=Path(temporary) / "output",
                arguments=tracker_arguments(),
                dataset_root=Path("/kaggle/input/video-aic"),
                source_kind="mounted",
            )

        self.assertFalse(tracker.enabled)
        self.assertNotIn("secret-api-key", stderr.getvalue())
        self.assertIn("<hidden>", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
