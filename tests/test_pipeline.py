from __future__ import annotations

import sys
import unittest
from collections import Counter, OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


CODE = Path(__file__).resolve().parents[1] / "Code"
if str(CODE) not in sys.path:
    sys.path.append(str(CODE))

import numpy as np

from multimodal_reranker import QwenVLQueryReranker
from evaluation import (
    Interval,
    KISGroundTruth,
    QAGroundTruth,
    TrakeGroundTruth,
    final_score,
    score_kis,
    score_qa,
    score_trake,
)
from ocr_index import OCRMemoryIndex, OCRRecord
from qa import VQABaseline
from query_router import QueryProfile, build_query_profile
from ranking import fuse_adaptive_retrieval_scores, select_diverse_results, select_multisource_candidates
from retrieval import AICRetrievalEngine, VideoMetadata


class PipelineTests(unittest.TestCase):
    def test_vqa_uses_qwen_on_kaggle_and_can_fallback_to_vilt(self) -> None:
        with patch.dict("os.environ", {"KAGGLE_KERNEL_RUN_TYPE": "Interactive"}, clear=False):
            vqa = VQABaseline()
        self.assertEqual(vqa.requested_backend, "qwen")
        self.assertEqual(vqa.model_name, "Qwen/Qwen3-VL-2B-Instruct")

        def fake_vilt_load():
            vqa._model = object()
            vqa.backend_name = "ViLT fallback"

        with (
            patch.object(vqa, "_load_qwen", side_effect=RuntimeError("simulated VRAM limit")),
            patch.object(vqa, "_load_vilt", side_effect=fake_vilt_load),
        ):
            vqa._load()
        self.assertEqual(vqa.backend_name, "ViLT fallback")
        self.assertIn("simulated VRAM limit", vqa.load_error)

    def test_pdf_final_score_example(self) -> None:
        scores = [0.5, 0.0, 0.8, *([0.0] * 97)]
        self.assertAlmostEqual(final_score(scores), 0.74)

    def test_pdf_kis_and_qa_rules(self) -> None:
        rows = [
            {"video_id": "L01_V001", "frame_id": "505", "answer": "Năm"},
            {"video_id": "L01_V001", "frame_id": "600", "answer": "5"},
        ]
        kis = KISGroundTruth("L01_V001", Interval(500, 510))
        qa = QAGroundTruth("L01_V001", Interval(500, 510), ("5", "five"))
        self.assertEqual(score_kis(rows, kis), [1.0, 0.0])
        self.assertEqual(score_qa(rows, qa), [1.0, 0.0])

    def test_pdf_trake_partial_credit_example(self) -> None:
        rows = [{
            "video_id": "L10_V010",
            "frame_id_1": "101",
            "frame_id_2": "156",
            "frame_id_3": "203",
            "frame_id_4": "251",
        }]
        ground_truth = TrakeGroundTruth(
            "L10_V010",
            (Interval(95, 105), Interval(145, 155), Interval(195, 205), Interval(245, 255)),
        )
        self.assertEqual(score_trake(rows, ground_truth), [0.75])

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

    def test_qwen_candidate_pool_uses_soft_video_diversity(self) -> None:
        candidates = [
            SimpleNamespace(
                video_id=video_id,
                keyframe_number=index,
                frame_id=index * 60,
                score=1.0 - index * 0.01,
                metadata_score=0.0,
            )
            for video_id in ("v1", "v2", "v3")
            for index in range(1, 5)
        ]
        combined = {(item.video_id, item.keyframe_number): item for item in candidates}
        selected = select_multisource_candidates(
            candidates,
            [],
            combined,
            6,
            source_weights={"visual": 0.8, "ocr": 0.05, "metadata": 0.05, "object": 0.1},
            max_per_video=2,
            min_frame_gap=30,
        )
        counts = Counter(item.video_id for item in selected)
        self.assertEqual(len(selected), 6)
        self.assertLessEqual(max(counts.values()), 2)

    def test_unknown_video_filter_returns_no_ram_candidates(self) -> None:
        engine = AICRetrievalEngine.__new__(AICRetrievalEngine)
        engine._ram_features = np.zeros((2, 512), dtype=np.float32)
        engine._ram_offsets = np.array([0, 2], dtype=np.int64)
        engine._ram_video_ids = ("known",)
        engine._ram_video_positions = {"known": 0}
        candidates = engine._ram_candidates(np.zeros(512, dtype=np.float32), 10, {"missing"})
        self.assertEqual(candidates, [])

    def test_metadata_bm25_precompute_rewards_rare_terms(self) -> None:
        engine = AICRetrievalEngine.__new__(AICRetrievalEngine)
        engine._features = {"generic": None, "specific": None, "other": None}
        engine._metadata_cache = {
            "generic": VideoMetadata(title="daily news bulletin"),
            "specific": VideoMetadata(title="daily news aurora expedition"),
            "other": VideoMetadata(title="daily weather update"),
        }
        engine._metadata_tokens = {}
        engine._metadata_document_lengths = {}
        engine._metadata_average_length = 1.0
        engine._metadata_idf = {}
        engine._metadata_ready = False
        engine.paths = SimpleNamespace(metadata_dir=Path("/metadata"))
        scores = engine._metadata_scores("daily aurora", engine._features)
        self.assertEqual(scores["specific"], 1.0)
        self.assertGreater(scores["specific"], scores["generic"])

    def test_qwen_query_router_returns_normalized_ocr_priority(self) -> None:
        class FakeReranker:
            def score_documents(self, _query, _documents, *, prompt):
                self.prompt = prompt
                return [0.15, 0.96, 0.10, 0.25]

        profile = build_query_profile("Biển có nội dung cảnh báo sạt lở", FakeReranker())
        self.assertEqual(profile.source, "Qwen+lexical")
        self.assertGreater(profile.ocr, max(profile.visual, profile.metadata, profile.object))
        self.assertAlmostEqual(sum(profile.values().values()), 1.0)

    def test_adaptive_ocr_fusion_can_recover_exact_text_candidate(self) -> None:
        visual = SimpleNamespace(
            visual_score=0.95,
            ocr_score=0.0,
            metadata_score=0.0,
            object_score=0.0,
            retrieval_score=0.9,
            score=0.9,
        )
        exact_text = SimpleNamespace(
            visual_score=0.35,
            ocr_score=1.0,
            metadata_score=0.0,
            object_score=0.0,
            retrieval_score=0.4,
            score=0.4,
        )
        profile = QueryProfile(visual=0.15, ocr=0.75, metadata=0.05, object=0.05, source="test")
        fuse_adaptive_retrieval_scores([visual, exact_text], profile)
        self.assertGreater(exact_text.score, visual.score)

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
        self.assertEqual(model.score_documents("route", ["OCR evidence", "visual evidence"]), [0.9, 0.9])
        self.assertEqual(model.score_documents("route", ["OCR evidence", "visual evidence"]), [0.9, 0.9])
        self.assertEqual(calls, [1, 2])


if __name__ == "__main__":
    unittest.main()
