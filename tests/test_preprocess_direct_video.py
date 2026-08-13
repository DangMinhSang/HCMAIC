from __future__ import annotations

import gzip
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


CODE = Path(__file__).resolve().parents[1] / "Code"
if str(CODE) not in sys.path:
    sys.path.append(str(CODE))

import numpy as np

import dashboard
from direct_video_retrieval import DirectVideoRetrievalEngine
from ocr_regions import OCR_INDEX_SCHEMA_VERSION
from preprocess_direct_video import (
    DIRECT_PREPROCESS_SCHEMA,
    TemporalFrameSampler,
    VideoWindow,
    artifact_paths,
    finalize_artifacts,
    select_video_window,
    write_json_atomic,
)


class DirectVideoPreprocessTests(unittest.TestCase):
    def test_video_window_is_sorted_and_one_based_inclusive(self) -> None:
        files = [Path("L22_V003.mp4"), Path("L21_V002.mp4"), Path("L21_V001.mp4")]
        window = select_video_window(files, 2, 3)
        self.assertEqual([path.stem for path in window.videos], ["L21_V002", "L22_V003"])
        self.assertEqual((window.start, window.end, window.total), (2, 3, 3))

        through_end = select_video_window(files, 2, 0)
        self.assertEqual([path.stem for path in through_end.videos], ["L21_V002", "L22_V003"])

    def test_video_window_rejects_out_of_range_values(self) -> None:
        files = [Path("L21_V001.mp4"), Path("L21_V002.mp4")]
        for start, end in ((0, 1), (2, 1), (3, 0), (1, 3)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                select_video_window(files, start, end)

        with self.assertRaisesRegex(ValueError, "Trùng video_id"):
            select_video_window(
                [Path("batch-a/L21_V001.mp4"), Path("batch-b/L21_V001.mp4")],
                1,
                0,
            )

    def test_temporal_sampler_uses_requested_rate(self) -> None:
        sampler = TemporalFrameSampler(source_fps=30.0, sample_fps=2.0)
        selected = [frame for frame in range(61) if sampler.accept(frame)]
        self.assertEqual(selected, [0, 15, 30, 45, 60])

        every_frame = TemporalFrameSampler(source_fps=25.0, sample_fps=0.0)
        self.assertEqual([frame for frame in range(5) if every_frame.accept(frame)], list(range(5)))

    @staticmethod
    def _write_completed_video(output: Path, video_id: str, clip_value: float = 1.0) -> None:
        artifacts = artifact_paths(output, video_id)
        artifacts.frames_dir.mkdir(parents=True)
        mapping = {
            "schema": DIRECT_PREPROCESS_SCHEMA,
            "video_id": video_id,
            "keyframe_number": 0,
            "sample_index": 0,
            "frame_id": 30,
            "pts_time": 1.0,
            "fps": 30.0,
            "image": "frames/000000030.png",
            "width": 640,
            "height": 360,
        }
        (artifacts.frames_dir / "000000030.png").write_bytes(b"png-placeholder")
        artifacts.mapping.write_text(json.dumps(mapping) + "\n", encoding="utf-8")
        with artifacts.clip.open("wb") as stream:
            vector = np.zeros((1, 512), dtype=np.float32)
            vector[0, 0] = clip_value
            np.save(stream, vector, allow_pickle=False)
        with artifacts.object_scores.open("wb") as stream:
            np.save(stream, np.asarray([[0.9]], dtype=np.float16), allow_pickle=False)
        artifacts.object_classes.write_text('{"classes":{"0":"person"}}\n', encoding="utf-8")
        with gzip.open(artifacts.objects, "wt", encoding="utf-8") as stream:
            stream.write(json.dumps({**mapping, "objects": [{"label": "person", "confidence": 0.9}]}) + "\n")
        write_json_atomic(
            artifacts.visual_marker,
            {
                "schema": DIRECT_PREPROCESS_SCHEMA,
                "video_id": video_id,
                "clip_model": "fake",
                "sampled_frames": 1,
            },
        )
        with gzip.open(artifacts.ocr, "wt", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "ocr_schema": OCR_INDEX_SCHEMA_VERSION,
                        "video_id": video_id,
                        "keyframe_number": 0,
                        "text": "bien canh bao sat lo",
                        "text_quality": 1.0,
                    }
                )
                + "\n"
            )
        write_json_atomic(
            artifacts.ocr_marker,
            {
                "schema": DIRECT_PREPROCESS_SCHEMA,
                "video_id": video_id,
                "records": 1,
                "scene_text_records": 1,
            },
        )

    def test_finalize_creates_global_indexes_and_resume_markers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "preprocessed"
            self._write_completed_video(output, "L21_V001")
            source = root / "L21_V001.mp4"
            window = VideoWindow((source,), 1, 1, 1)

            manifest = finalize_artifacts(window, output)

            self.assertEqual(manifest["complete_videos"], 1)
            self.assertEqual(manifest["ocr_records"], 1)
            self.assertEqual(manifest["object_records"], 1)
            self.assertTrue((output / "ocr_index.jsonl.gz").is_file())
            self.assertTrue(artifact_paths(output, "L21_V001").complete_marker.is_file())
            self.assertTrue((output / "shards" / "pre_0001_0001.json").is_file())

    def test_engine_loads_preprocessed_clip_png_and_objects(self) -> None:
        class FakeEncoder:
            model_name = "fake"
            last_query = None

            @staticmethod
            def encode(*_arguments):
                vector = np.zeros(512, dtype=np.float32)
                vector[0] = 1.0
                return vector

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "preprocessed"
            self._write_completed_video(output, "L21_V001")
            video = root / "L21_V001.mp4"
            video.write_bytes(b"not-decoded")
            with patch.dict(os.environ, {"AIC_DIRECT_PREPROCESSED_ROOT": str(output)}):
                engine = DirectVideoRetrievalEngine(root, [video], encoder=FakeEncoder())
                engine.prepare_runtime()
                results = engine.search("person", top_k=1, min_frame_gap=0)

            self.assertEqual(engine.vector_count, 1)
            self.assertEqual(results[0].frame_id, 30)
            self.assertEqual(results[0].object_labels, ("person",))
            self.assertTrue(results[0].image_path.endswith("000000030.png"))

    def test_dashboard_loads_only_direct_ocr_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            self._write_completed_video(output, "L21_V001")
            window = VideoWindow((Path("L21_V001.mp4"),), 1, 1, 1)
            finalize_artifacts(window, output)
            old_index = dashboard.OCR_INDEX
            old_loaded = dashboard.OCR_INDEX_LOADED
            try:
                dashboard.OCR_INDEX = None
                dashboard.OCR_INDEX_LOADED = False
                with patch.dict(
                    os.environ,
                    {
                        "AIC_DIRECT_VIDEO": "1",
                        "AIC_DIRECT_PREPROCESSED_ROOT": str(output),
                        "AIC_OCR_INDEX": str(output / "must-not-be-used.jsonl.gz"),
                    },
                ):
                    index = dashboard.get_ocr_index()
                self.assertIsNotNone(index)
                self.assertEqual(index.record_count, 1)
                self.assertEqual(index.records[0].keyframe_number, 0)
            finally:
                dashboard.OCR_INDEX = old_index
                dashboard.OCR_INDEX_LOADED = old_loaded


if __name__ == "__main__":
    unittest.main()
