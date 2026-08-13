from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


CODE = Path(__file__).resolve().parents[1] / "Code"
if str(CODE) not in sys.path:
    sys.path.append(str(CODE))

import numpy as np

from direct_video_retrieval import (
    DirectFrame,
    DirectVideoRetrievalEngine,
    discover_video_files,
    resolve_video_dataset,
)
import dashboard


class DirectVideoTests(unittest.TestCase):
    def test_direct_mode_is_opt_in(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(dashboard.direct_video_enabled())
        with patch.dict(os.environ, {"AIC_DIRECT_VIDEO": "1"}, clear=True):
            self.assertTrue(dashboard.direct_video_enabled())

    def _make_dataset(self, root: Path) -> Path:
        video_dir = root / "Videos_L21_a" / "video"
        video_dir.mkdir(parents=True)
        (video_dir / "L21_V001.mp4").write_bytes(b"not-decoded-in-this-test")
        (video_dir / "L21_V002.mp4").write_bytes(b"not-decoded-in-this-test")
        (video_dir / "README.txt").write_text("ignore", encoding="utf-8")
        return root

    def test_discovery_matches_video_layout_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._make_dataset(Path(temporary))
            files = discover_video_files(root)
            self.assertEqual([path.stem for path in files], ["L21_V001", "L21_V002"])

    def test_mount_is_preferred_over_kagglehub(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._make_dataset(Path(temporary))
            with patch.dict(
                os.environ,
                {"AIC_DIRECT_VIDEO_ROOT": str(root), "AIC_DIRECT_VIDEO_DATASET": "custom/id"},
                clear=False,
            ):
                with patch.dict(
                    sys.modules,
                    {"kagglehub": SimpleNamespace(dataset_download=lambda _dataset: self.fail("downloaded"))},
                ):
                    resolved, files, source = resolve_video_dataset()
            self.assertEqual(resolved, root.resolve())
            self.assertEqual(len(files), 2)
            self.assertEqual(source, "mounted")

    def test_kagglehub_is_only_fallback_when_mount_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._make_dataset(Path(temporary) / "downloaded")
            missing = Path(temporary) / "missing"
            fake_kagglehub = SimpleNamespace(dataset_download=lambda dataset: self.assertEqual(dataset, "custom/id") or root)
            with patch.dict(
                os.environ,
                {
                    "AIC_DIRECT_VIDEO_ROOT": str(missing),
                    "AIC_DATA_ROOT": str(missing / "input"),
                    "AIC_DIRECT_VIDEO_DATASET": "custom/id",
                },
                clear=False,
            ):
                with patch.dict(sys.modules, {"kagglehub": fake_kagglehub}):
                    resolved, files, source = resolve_video_dataset()
            self.assertEqual(resolved, root.resolve())
            self.assertEqual(len(files), 2)
            self.assertEqual(source, "kagglehub")

    def test_engine_has_no_btc_feature_dependency_before_prepare(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._make_dataset(Path(temporary))
            engine = DirectVideoRetrievalEngine(root)
            self.assertEqual(engine.source_mode, "direct-video")
            self.assertEqual(engine.video_count, 2)
            self.assertIn("không dùng feature/mapping BTC", engine.source_description)
            self.assertEqual(engine.vector_count, 0)

    def test_direct_search_and_trake_use_local_index(self) -> None:
        class FakeEncoder:
            model_name = "fake"
            last_query = None

            @staticmethod
            def encode(*_arguments):
                vector = np.zeros(512, dtype="float32")
                vector[0] = 1.0
                return vector

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "L21_V001.mp4"
            video.write_bytes(b"not-decoded-in-this-test")
            engine = DirectVideoRetrievalEngine(root, [video], encoder=FakeEncoder())
            engine._embeddings = np.zeros((5, 512), dtype="float32")
            engine._embeddings[:, 0] = [0.1, 0.9, 0.2, 0.8, 0.7]
            engine._records = [DirectFrame(index, index * 15, index * 0.5, 30.0, video) for index in range(5)]
            engine._set_records(engine._records)
            engine.ensure_result_image = lambda _result: None
            engine.ensure_result_images = lambda _results: None

            results = engine.search("scene", top_k=3, min_frame_gap=0)
            self.assertEqual([result.frame_id for result in results], [15, 45, 60])
            sequences = engine.search_trake(["prepare", "airborne"], top_videos=1)
            self.assertEqual(len(sequences), 1)
            self.assertEqual([frame.frame_id for frame in sequences[0].frames], [15, 45])

    def test_lazy_video_frames_use_a_bounded_runtime_cache(self) -> None:
        class FakeEncoder:
            model_name = "fake"

        class FakeCapture:
            def __init__(self, _path: str) -> None:
                self.frame_id = 0

            @staticmethod
            def isOpened() -> bool:
                return True

            def set(self, _property: int, value: int) -> None:
                self.frame_id = int(value)

            def read(self):
                return True, np.full((2, 2, 3), self.frame_id, dtype=np.uint8)

            @staticmethod
            def release() -> None:
                return None

        class FakeCV2:
            CAP_PROP_POS_FRAMES = 1
            IMWRITE_JPEG_QUALITY = 2
            VideoCapture = FakeCapture

            @staticmethod
            def imwrite(path: str, image, _options) -> bool:
                Path(path).write_bytes(bytes([int(image[0, 0, 0])]))
                return True

        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ,
            {
                "AIC_RUNTIME_DIR": temporary,
                "AIC_DIRECT_FRAME_CACHE_MAX": "2",
            },
        ):
            root = Path(temporary)
            video = root / "L21_V001.mp4"
            video.write_bytes(b"video-source")
            engine = DirectVideoRetrievalEngine(root, [video], encoder=FakeEncoder())
            engine._cv2 = FakeCV2
            paths = [
                engine._materialize_frame(DirectFrame(index, index, index / 30, 30, video))
                for index in range(3)
            ]

            self.assertTrue(all(path is not None for path in paths))
            self.assertEqual(len(list(engine._frame_cache_dir.glob("*/*.jpg"))), 2)
            self.assertEqual(sum(path.exists() for path in paths), 2)
            self.assertTrue(paths[-1].exists())

    def test_batch_materialization_opens_each_video_once(self) -> None:
        opened = []

        class FakeEncoder:
            model_name = "fake"

        class FakeCapture:
            def __init__(self, path: str) -> None:
                opened.append(path)
                self.position = 0

            @staticmethod
            def isOpened() -> bool:
                return True

            def set(self, _property: int, value: int) -> None:
                self.position = int(value)

            def read(self):
                frame = np.full((2, 2, 3), self.position, dtype=np.uint8)
                self.position += 1
                return True, frame

            @staticmethod
            def release() -> None:
                return None

        class FakeCV2:
            CAP_PROP_POS_FRAMES = 1
            IMWRITE_JPEG_QUALITY = 2
            VideoCapture = FakeCapture

            @staticmethod
            def imwrite(path: str, image, _options) -> bool:
                Path(path).write_bytes(bytes([int(image[0, 0, 0])]))
                return True

        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ,
            {"AIC_RUNTIME_DIR": temporary},
        ):
            root = Path(temporary)
            video = root / "L21_V001.mp4"
            video.write_bytes(b"video-source")
            engine = DirectVideoRetrievalEngine(root, [video], encoder=FakeEncoder())
            engine._cv2 = FakeCV2
            engine._frame_for_result = lambda result: DirectFrame(
                result.keyframe_number,
                result.keyframe_number,
                result.keyframe_number / 30,
                30,
                video,
            )
            results = [
                SimpleNamespace(video_id=video.stem, keyframe_number=index, image_path=None)
                for index in (1, 2, 5)
            ]

            engine.ensure_result_images(results)

            self.assertEqual(opened, [str(video)])
            self.assertTrue(all(result.image_path for result in results))
            self.assertEqual(
                [Path(result.image_path).read_bytes()[0] for result in results],
                [1, 2, 5],
            )


if __name__ == "__main__":
    unittest.main()
