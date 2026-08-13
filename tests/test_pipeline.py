from __future__ import annotations

import sys
import tempfile
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
from build_ocr_index import read_text
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
from ocr_regions import (
    OCR_INDEX_SCHEMA_VERSION,
    is_broadcast_overlay_box,
    legacy_text_quality,
    prepare_reranker_image,
)
from query_analyzer import LightweightQueryAnalyzer, split_query_clauses
from qa import VQABaseline, normalize_answer, question_kind
from query_language import parse_trake_events
from query_router import (
    QueryProfile,
    build_query_profile,
    has_explicit_ocr_intent,
    is_ambiguous_warning_query,
)
from ranking import (
    fuse_adaptive_retrieval_scores,
    fuse_multimodal_rerank_score,
    select_diverse_results,
    select_multisource_candidates,
)
from retrieval import AICRetrievalEngine, VideoMetadata


class PipelineTests(unittest.TestCase):
    def test_query_analyzer_splits_visual_and_ocr_clauses(self) -> None:
        analyzer = LightweightQueryAnalyzer(model_name="offline-test-model")
        analysis = analyzer.analyze(
            "Biển cảnh báo màu vàng có nội dung là cảnh báo sạt lở nguy hiểm"
        )
        self.assertEqual(
            [clause.text for clause in analysis.clauses],
            ["Biển cảnh báo màu vàng", "cảnh báo sạt lở nguy hiểm"],
        )
        self.assertEqual(analysis.clauses[0].dominant, "visual")
        self.assertEqual(analysis.clauses[1].dominant, "ocr")
        self.assertEqual(analysis.query_for("visual"), "Biển cảnh báo màu vàng")
        self.assertEqual(analysis.query_for("ocr"), "cảnh báo sạt lở nguy hiểm")
        self.assertGreater(analysis.weights["visual"], 0.25)
        self.assertGreater(analysis.weights["ocr"], 0.25)
        self.assertAlmostEqual(sum(analysis.weights.values()), 1.0)

    def test_query_analyzer_keeps_plain_query_usable_without_model(self) -> None:
        seeds = split_query_clauses("Một vận động viên nhảy lên trong sân thi đấu")
        self.assertEqual(len(seeds), 1)
        analysis = LightweightQueryAnalyzer(model_name="offline-test-model").analyze(seeds[0].text)
        self.assertTrue(analysis.query_for("visual"))

    def test_short_warning_query_keeps_secondary_ocr_recall(self) -> None:
        analysis = LightweightQueryAnalyzer(model_name="offline-test-model").analyze(
            "Cảnh báo sạt lở nguy hiểm"
        )
        self.assertTrue(analysis.query_for("visual"))
        self.assertEqual(analysis.query_for("ocr"), "Cảnh báo sạt lở nguy hiểm")

    def test_warning_phrase_prefers_visual_context_without_disabling_ocr_recall(self) -> None:
        self.assertTrue(is_ambiguous_warning_query("Cảnh báo sạt lở nguy hiểm"))
        self.assertFalse(has_explicit_ocr_intent("Cảnh báo sạt lở nguy hiểm"))
        profile = build_query_profile("Cảnh báo sạt lở nguy hiểm")
        self.assertGreater(profile.visual, profile.ocr)

        explicit = build_query_profile("Biển báo có chữ CẢNH BÁO SẠT LỞ")
        self.assertGreater(explicit.ocr, explicit.visual)

    def test_vqa_prompt_normalizes_count_and_holding_answers(self) -> None:
        self.assertEqual(question_kind("Có bao nhiêu người trong ảnh?"), "count")
        self.assertEqual(normalize_answer("There are two people.", "count"), "2")
        self.assertEqual(
            normalize_answer("The red-shirted person is holding a microphone.", "holding"),
            "microphone",
        )

    def test_visual_context_beats_ocr_only_slide_for_scene_query(self) -> None:
        slide = fuse_multimodal_rerank_score(0.96, 0.20, 0.45, explicit_ocr=False)
        physical_scene = fuse_multimodal_rerank_score(0.86, 0.84, 0.62, explicit_ocr=False)
        self.assertGreater(physical_scene, slide)
        self.assertGreater(
            fuse_multimodal_rerank_score(0.90, 0.30, 0.70, explicit_ocr=True),
            fuse_multimodal_rerank_score(0.80, 0.85, 0.25, explicit_ocr=True),
        )

    def test_trake_parser_ignores_demo_introduction_and_keeps_four_events(self) -> None:
        prompt = """Hãy tìm video ghi lại một chuỗi hành động.
Video cần chứa đầy đủ các sự kiện sau theo đúng thứ tự:

Sự kiện 1 — Chuẩn bị: Vận động viên bắt đầu di chuyển.
Sự kiện 2 — Thực hiện động tác: Cơ thể bắt đầu rời vị trí.
Sự kiện 3 — Ở trên không: Vận động viên đạt vị trí cao nhất.
Sự kiện 4 — Tiếp đất: Bắt đầu tiếp xúc trở lại với mặt đất."""
        events = parse_trake_events(prompt)
        self.assertEqual(len(events), 4)
        self.assertTrue(events[0].startswith("Chuẩn bị:"))
        self.assertTrue(events[-1].startswith("Tiếp đất:"))
        self.assertNotIn("Hãy tìm video", " ".join(events))

    def test_trake_parser_supports_plain_numbered_lines(self) -> None:
        events = parse_trake_events("1. Chuẩn bị\n2. Nhảy\n3. Tiếp đất")
        self.assertEqual(events, ["Chuẩn bị", "Nhảy", "Tiếp đất"])

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

    def test_trake_neighbor_refinement_stays_strictly_ordered(self) -> None:
        choices = AICRetrievalEngine._ordered_candidate_alignment(
            [[2, 3], [2, 3], [3, 4]],
            [[0.90, 0.80], [0.95, 0.70], [0.99, 0.60]],
        )
        self.assertEqual(choices, [0, 1, 1])
        selected = [indices[choice] for indices, choice in zip([[2, 3], [2, 3], [3, 4]], choices)]
        self.assertTrue(all(left < right for left, right in zip(selected, selected[1:])))

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

    def test_bottom_news_ticker_is_not_scene_ocr(self) -> None:
        ticker_box = [[0, 650], [1250, 650], [1250, 700], [0, 700]]
        sign_box = [[250, 180], [980, 180], [980, 270], [250, 270]]
        self.assertTrue(
            is_broadcast_overlay_box(
                ticker_box,
                "Cảnh báo nguy cơ lũ quét sạt lở đất vùng núi và trung du Bắc Bộ",
                1280,
                720,
            )
        )
        self.assertFalse(
            is_broadcast_overlay_box(
                sign_box,
                "CẢNH BÁO SẠT LỞ NGUY HIỂM",
                1280,
                720,
            )
        )

    def test_pre_ocr_splits_ticker_from_physical_sign(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is not installed in the local test runtime")

        class FakeReader:
            @staticmethod
            def ocr(_path, cls=False):
                self.assertFalse(cls)
                return [[
                    [
                        [[250, 180], [980, 180], [980, 270], [250, 270]],
                        ("CẢNH BÁO SẠT LỞ NGUY HIỂM", 0.99),
                    ],
                    [
                        [[0, 650], [1250, 650], [1250, 700], [0, 700]],
                        ("Cảnh báo nguy cơ lũ quét sạt lở đất", 0.98),
                    ],
                ]]

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "frame.jpg"
            Image.new("RGB", (1280, 720), "white").save(image_path)
            scene, overlay, scene_count, overlay_count = read_text(FakeReader(), image_path, 0.45)
        self.assertEqual(scene, "CẢNH BÁO SẠT LỞ NGUY HIỂM")
        self.assertIn("lũ quét", overlay)
        self.assertEqual((scene_count, overlay_count), (1, 1))

    def test_pre_ocr_accepts_in_memory_video_frame(self) -> None:
        observed = {}

        class FakeReader:
            @staticmethod
            def ocr(source, cls=False):
                observed["source"] = source
                observed["cls"] = cls
                return [[
                    [
                        [[150, 100], [490, 100], [490, 180], [150, 180]],
                        ("BIỂN CẢNH BÁO", 0.99),
                    ]
                ]]

        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        scene, overlay, scene_count, overlay_count = read_text(FakeReader(), frame, 0.45)

        self.assertIs(observed["source"], frame)
        self.assertFalse(observed["cls"])
        self.assertEqual(scene, "BIỂN CẢNH BÁO")
        self.assertEqual(overlay, "")
        self.assertEqual((scene_count, overlay_count), (1, 0))

    def test_legacy_weather_ticker_is_suppressed(self) -> None:
        # Exact noisy PaddleOCR text from the unrelated tango frame L22_V021/204.
        ticker = "Cänh báo nguy ca lo quét sat l& dát vng núi vä trung du Bäc Bö"
        sign = "CẢNH BÁO SẠT LỞ NGUY HIỂM TẠM DỪNG LƯU THÔNG"
        self.assertLess(legacy_text_quality(ticker), 0.20)
        self.assertEqual(legacy_text_quality(sign), 1.0)
        index = OCRMemoryIndex(
            [
                OCRRecord("unrelated-dance", 1, ticker),
                OCRRecord("physical-sign", 2, sign, OCR_INDEX_SCHEMA_VERSION),
            ]
        )
        hits = index.search("cảnh báo sạt lở nguy hiểm")
        self.assertTrue(hits)
        self.assertEqual(hits[0].video_id, "physical-sign")
        self.assertNotIn("unrelated-dance", {hit.video_id for hit in hits})

    def test_qwen_receives_an_in_memory_blurred_lower_third(self) -> None:
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            self.skipTest("Pillow is not installed in the local test runtime")
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "ticker.jpg"
            source = Image.new("RGB", (320, 180), "white")
            draw = ImageDraw.Draw(source)
            for x in range(0, 320, 8):
                draw.rectangle((x, 150, x + 3, 179), fill="black")
            source.save(image_path)
            masked = prepare_reranker_image(image_path)
            self.assertEqual(masked.getpixel((20, 20)), source.getpixel((20, 20)))
            self.assertNotEqual(masked.crop((0, 150, 320, 180)).tobytes(), source.crop((0, 150, 320, 180)).tobytes())
            masked.close()
            source.close()

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
        profile = build_query_profile("Biển có nội dung cảnh báo sạt lở")
        self.assertIn("structure", profile.source)
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

    def test_qwen_frame_score_runs_joint_and_visual_only_passes(self) -> None:
        model = QwenVLQueryReranker.__new__(QwenVLQueryReranker)
        calls: list[bool] = []

        def fake_score_pairs(_queries, _results, *, prompt, visual_only=False):
            calls.append(visual_only)
            return [0.8]

        model.score_pairs = fake_score_pairs
        result = SimpleNamespace(visual_score=0.2, ocr_score=0.9)
        scores = model.score("scene", [result])
        self.assertEqual(calls, [False, True])
        self.assertEqual(scores[0].visual, 0.8)


if __name__ == "__main__":
    unittest.main()
