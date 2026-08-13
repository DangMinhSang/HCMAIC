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
from direct_video_retrieval import (
    DirectVideoRetrievalEngine,
    parse_frame_steps,
    temporal_modulo_indices,
)
from ocr_regions import OCR_INDEX_SCHEMA_VERSION
from preprocess_direct_video import (
    DIRECT_PREPROCESS_SCHEMA,
    TemporalFrameSampler,
    VideoWindow,
    artifact_paths,
    build_parser,
    finalize_artifacts,
    select_video_window,
    write_json_atomic,
)


class DirectVideoPreprocessTests(unittest.TestCase):
    def test_preprocess_defaults_to_every_frame(self) -> None:
        arguments = build_parser().parse_args(["--stage", "visual"])
        self.assertEqual(arguments.sample_fps, 0.0)

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

    def test_modulo_schedule_reaches_every_frame(self) -> None:
        self.assertEqual(parse_frame_steps("4,2,1"), (4, 2, 1))
        for invalid in ("", "4,2", "2,4,1", "4,3,1", "0,1"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_frame_steps(invalid)

        frame_ids = np.arange(12, dtype=np.int64)
        even = temporal_modulo_indices(frame_ids, [4, 8], 4, 2)
        every = temporal_modulo_indices(frame_ids, even.tolist(), 2, 1)
        self.assertTrue(all(int(frame_ids[index]) % 2 == 0 for index in even))
        self.assertEqual(every.tolist(), list(range(12)))

    @staticmethod
    def _write_completed_video(output: Path, video_id: str) -> None:
        artifacts = artifact_paths(output, video_id)
        artifacts.frames_dir.mkdir(parents=True)
        mappings = []
        for frame_id in range(8):
            mapping = {
                "schema": DIRECT_PREPROCESS_SCHEMA,
                "video_id": video_id,
                "keyframe_number": frame_id,
                "sample_index": frame_id,
                "frame_id": frame_id,
                "pts_time": frame_id / 30.0,
                "fps": 30.0,
                "image": f"frames/{frame_id:09d}.png",
                "width": 640,
                "height": 360,
            }
            mappings.append(mapping)
            (artifacts.frames_dir / f"{frame_id:09d}.png").write_bytes(b"png-placeholder")
        artifacts.mapping.write_text(
            "".join(json.dumps(mapping) + "\n" for mapping in mappings),
            encoding="utf-8",
        )
        frame_ids = np.arange(8, dtype=np.int64)
        pts_times = frame_ids.astype(np.float32) / 30.0
        with artifacts.frame_ids.open("wb") as stream:
            np.save(stream, frame_ids, allow_pickle=False)
        with artifacts.pts_times.open("wb") as stream:
            np.save(stream, pts_times, allow_pickle=False)
        with artifacts.clip.open("wb") as stream:
            vectors = np.zeros((8, 512), dtype=np.float32)
            vectors[:, 0] = [0.40, 0.20, 0.60, 1.00, 0.80, 0.20, 0.40, 0.50]
            vectors[:, 1] = [0.80, 1.00, 0.90, 0.10, 0.10, 0.00, 0.00, 0.00]
            vectors[:, 2] = [0.00, 0.00, 0.10, 0.20, 0.70, 1.00, 0.80, 0.70]
            vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
            np.save(stream, vectors, allow_pickle=False)
        with artifacts.object_scores.open("wb") as stream:
            scores = np.zeros((8, 1), dtype=np.float16)
            scores[3, 0] = 0.9
            np.save(stream, scores, allow_pickle=False)
        artifacts.object_classes.write_text('{"classes":{"0":"person"}}\n', encoding="utf-8")
        with gzip.open(artifacts.objects, "wt", encoding="utf-8") as stream:
            for mapping in mappings:
                objects = (
                    [{"label": "person", "confidence": 0.9}]
                    if mapping["frame_id"] == 3
                    else []
                )
                stream.write(json.dumps({**mapping, "objects": objects}) + "\n")
        write_json_atomic(
            artifacts.visual_marker,
            {
                "schema": DIRECT_PREPROCESS_SCHEMA,
                "video_id": video_id,
                "clip_model": "fake",
                "source_fps": 30.0,
                "source_frames": 8,
                "sample_fps": 0.0,
                "all_frames": True,
                "sampled_frames": 8,
            },
        )
        with gzip.open(artifacts.ocr, "wt", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "ocr_schema": OCR_INDEX_SCHEMA_VERSION,
                        "video_id": video_id,
                        "keyframe_number": 3,
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
            self.assertEqual(manifest["all_frame_videos"], 1)
            self.assertEqual(manifest["ocr_records"], 1)
            self.assertEqual(manifest["object_records"], 8)
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

            self.assertEqual(engine.vector_count, 8)
            self.assertIsNone(engine._embeddings)
            self.assertEqual(engine._coarse_embeddings.shape, (2, 512))
            self.assertIsInstance(engine._preprocessed_shards["L21_V001"].embeddings, np.memmap)
            self.assertEqual(results[0].frame_id, 3)
            self.assertEqual(results[0].object_labels, ("person",))
            self.assertTrue(results[0].image_path.endswith("000000003.png"))

    def test_trake_refines_coarse_centers_to_exact_frames(self) -> None:
        class FakeEncoder:
            model_name = "fake"
            last_query = None

            @staticmethod
            def encode(query, *_arguments):
                vector = np.zeros(512, dtype=np.float32)
                vector[1 if query == "prepare" else 2] = 1.0
                return vector

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "preprocessed"
            self._write_completed_video(output, "L21_V001")
            video = root / "L21_V001.mp4"
            video.write_bytes(b"not-decoded")
            with patch.dict(
                os.environ,
                {
                    "AIC_DIRECT_PREPROCESSED_ROOT": str(output),
                    "AIC_DIRECT_FRAME_STEPS": "4,2,1",
                },
            ):
                engine = DirectVideoRetrievalEngine(root, [video], encoder=FakeEncoder())
                sequences = engine.search_trake(["prepare", "land"], top_videos=1)

            self.assertEqual(len(sequences), 1)
            self.assertEqual([frame.frame_id for frame in sequences[0].frames], [1, 5])

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
                self.assertEqual(index.records[0].keyframe_number, 3)
            finally:
                dashboard.OCR_INDEX = old_index
                dashboard.OCR_INDEX_LOADED = old_loaded


if __name__ == "__main__":
    unittest.main()
