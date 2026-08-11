from __future__ import annotations

import sys
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace


CODE = Path(__file__).resolve().parents[1] / "Code"
if str(CODE) not in sys.path:
    sys.path.append(str(CODE))

import numpy as np

from multimodal_reranker import QwenVLQueryReranker
from ocr_index import OCRMemoryIndex, OCRRecord
from ranking import select_diverse_results, select_multisource_candidates
from retrieval import AICRetrievalEngine


class PipelineTests(unittest.TestCase):
    def test_trake_alignment_is_strictly_ordered(self) -> None:
        scores = np.array(
            [
                [0.90, 0.10, 0.05],
                [0.80, 0.70, 0.10],
                [0.10, 0.95, 0.20],
                [0.05, 0.60, 0.98],
            ],
            dtype=np.float32,
        )
        alignment = AICRetrievalEngine._ordered_alignment(scores)
        self.assertIsNotNone(alignment)
        indices, values = alignment
        self.assertEqual(indices.tolist(), [0, 2, 3])
        self.assertTrue(np.all(np.diff(indices) > 0))
        self.assertGreater(float(values.mean()), 0.9)

    def test_trake_rejects_too_few_frames(self) -> None:
        scores = np.zeros((2, 3), dtype=np.float32)
        self.assertIsNone(AICRetrievalEngine._ordered_alignment(scores))

    def test_ocr_bilingual_aliases_rank_exact_sign_first(self) -> None:
        index = OCRMemoryIndex(
            [
                OCRRecord("correct", 9, "CẢNH BÁO SẠT LỞ NGUY HIỂM TẠM DỪNG LƯU THÔNG"),
                OCRRecord("weather", 2, "Cảnh báo thời tiết có mưa lớn"),
                OCRRecord("news", 3, "Khu vực xảy ra sạt lở đất"),
            ]
        )
        for query in ("cảnh báo sạt lở nguy hiểm", "dangerous landslide warning"):
            with self.subTest(query=query):
                hits = index.search(query)
                self.assertTrue(hits)
                self.assertEqual(hits[0].video_id, "correct")

    def test_multisource_pool_reserves_ocr_candidate(self) -> None:
        def result(video: str, keyframe: int, score: float, metadata: float = 0.0):
            return SimpleNamespace(
                video_id=video,
                keyframe_number=keyframe,
                frame_id=keyframe * 30,
                score=score,
                metadata_score=metadata,
            )

        visual = [result("v1", 1, 0.9), result("v2", 1, 0.8), result("v3", 1, 0.7)]
        ocr_result = result("v4", 1, 0.6, 0.9)
        combined = {(item.video_id, item.keyframe_number): item for item in [*visual, ocr_result]}
        hits = [SimpleNamespace(video_id="v4", keyframe_number=1)]
        selected = select_multisource_candidates(visual, hits, combined, 4)
        self.assertEqual(len(selected), 4)
        self.assertIn("v4", {item.video_id for item in selected})
        diverse = select_diverse_results(selected, limit=4, min_frame_gap=0, max_per_video=1)
        self.assertEqual(len({item.video_id for item in diverse}), 4)

    def test_qwen_pair_scores_are_cached(self) -> None:
        model = QwenVLQueryReranker.__new__(QwenVLQueryReranker)
        model._score_cache = OrderedDict()
        model._cache_size = 8
        calls: list[int] = []

        def fake_predict(pairs, _prompt):
            calls.append(len(pairs))
            return [0.9] * len(pairs)

        model._predict_pairs = fake_predict
        result = SimpleNamespace(image_path="/tmp/frame.jpg", ocr_text="warning")
        self.assertEqual(model.score_pairs(["query"], [result]), [0.9])
        self.assertEqual(model.score_pairs(["query"], [result]), [0.9])
        self.assertEqual(calls, [1])


if __name__ == "__main__":
    unittest.main()
