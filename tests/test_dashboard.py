from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


CODE = Path(__file__).resolve().parents[1] / "Code"
if str(CODE) not in sys.path:
    sys.path.append(str(CODE))

import dashboard
from dashboard import SearchSession, StoredResult
from retrieval import SearchResult, TrakeVideoResult
from query_analyzer import LightweightQueryAnalyzer
from query_router import build_query_analysis, build_query_profile


def make_result(video_id: str, frame_id: int, *, video_path: str | None = None) -> SearchResult:
    return SearchResult(
        rank=1,
        video_id=video_id,
        frame_id=frame_id,
        keyframe_number=1,
        pts_time=3.5,
        visual_score=0.7,
        metadata_score=0.1,
        score=0.8,
        video_path=video_path,
    )


class DashboardTests(unittest.TestCase):
    session_id = "dashboard-regression-test"

    def setUp(self) -> None:
        self.client = dashboard.app.test_client()
        with self.client.session_transaction() as browser_session:
            browser_session["aic_session"] = self.session_id

    def tearDown(self) -> None:
        dashboard.SESSIONS.pop(self.session_id, None)
        with dashboard.SEARCH_JOBS_LOCK:
            stale = [
                identifier
                for identifier, job in dashboard.SEARCH_JOBS.items()
                if job.session_id == self.session_id
            ]
            for identifier in stale:
                dashboard.SEARCH_JOBS.pop(identifier, None)

    def test_mounted_dashboard_uses_prefixed_api_urls(self) -> None:
        response = self.client.get("/", environ_overrides={"SCRIPT_NAME": "/dashboard"})
        page = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn('"search": "/dashboard/api/search"', page)
        self.assertIn('"export": "/dashboard/api/export"', page)

    def test_api_errors_are_json_not_html(self) -> None:
        missing = self.client.get("/api/does-not-exist")
        self.assertEqual(missing.status_code, 404)
        self.assertTrue(missing.is_json)
        malformed = self.client.post("/api/search", json=["not", "an", "object"])
        self.assertEqual(malformed.status_code, 400)
        self.assertTrue(malformed.is_json)

    def test_frame_and_answer_overrides_reach_csv(self) -> None:
        kis = make_result("L01_V001", 100)
        dashboard.SESSIONS[self.session_id] = SearchSession(
            task="kis",
            results={"card": StoredResult(kis)},
        )
        response = self.client.post(
            "/api/export",
            json={"selected": ["card"], "frame_overrides": {"card": 123}},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("L01_V001,123", response.get_data(as_text=True))

        qa = make_result("L01_V002", 150)
        qa.answer = "old answer"
        dashboard.SESSIONS[self.session_id] = SearchSession(
            task="qa",
            results={"qa-card": StoredResult(qa)},
        )
        response = self.client.post(
            "/api/export",
            json={"selected": ["qa-card"], "answer_overrides": {"qa-card": "five"}},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("L01_V002,150,five", response.get_data(as_text=True))

        first = make_result("L02_V002", 200)
        second = make_result("L02_V002", 300)
        sequence = TrakeVideoResult(rank=1, video_id="L02_V002", score=0.8, frames=[first, second])
        dashboard.SESSIONS[self.session_id] = SearchSession(
            task="trake",
            results={
                "event1": StoredResult(first, group="sequence", event_index=1),
                "event2": StoredResult(second, group="sequence", event_index=2),
            },
            trake_sequences={"sequence": sequence},
        )
        response = self.client.post(
            "/api/export",
            json={
                "selected": ["event1", "event2"],
                "frame_overrides": {"event2": 333},
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("L02_V002,200,333", response.get_data(as_text=True))

    def test_search_runs_as_short_polled_job(self) -> None:
        result = make_result("L04_V004", 500)
        with (
            patch.object(dashboard, "get_engine", return_value=object()),
            patch.object(
                dashboard,
                "make_kis_results",
                return_value=([StoredResult(result)], "pipeline complete"),
            ),
        ):
            accepted = self.client.post(
                "/api/search",
                json={"task": "kis", "query": "a test scene"},
                environ_overrides={"SCRIPT_NAME": "/dashboard"},
            )
            self.assertEqual(accepted.status_code, 202)
            status_url = accepted.get_json()["status_url"]
            self.assertTrue(status_url.startswith("/dashboard/api/search/"))
            local_status_url = status_url.removeprefix("/dashboard")
            for _attempt in range(100):
                status = self.client.get(local_status_url)
                payload = status.get_json()
                if payload["status"] in {"complete", "error"}:
                    break
                time.sleep(0.005)
        self.assertEqual(payload["status"], "complete")
        self.assertEqual(payload["notice"], "pipeline complete")
        self.assertEqual(payload["results"][0]["frame_id"], 500)
        self.assertAlmostEqual(
            sum(payload["query_profile"][name] for name in ("visual", "ocr", "metadata", "object")),
            1.0,
            places=3,
        )

    def test_kis_uses_source_specific_visual_and_ocr_phrases(self) -> None:
        query = "Biển cảnh báo màu vàng có nội dung là cảnh báo sạt lở nguy hiểm"
        analyzer = LightweightQueryAnalyzer(model_name="offline-test-model")
        analysis = build_query_analysis(query, analyzer)
        profile = build_query_profile(query, analyzer=analyzer, analysis=analysis)

        class FakeEngine:
            def __init__(self) -> None:
                self.queries: list[str] = []
                self.encoder = SimpleNamespace(last_query=None)

            def search(self, source_query: str, **_kwargs):
                self.queries.append(source_query)
                return [make_result(f"L{len(self.queries):02d}_V001", len(self.queries) * 100)]

        fake_engine = FakeEngine()
        ocr_queries: list[str] = []

        def fake_ocr_search(source_query: str, **_kwargs):
            ocr_queries.append(source_query)
            return []

        fake_ocr = SimpleNamespace(
            schema_version=2,
            legacy_record_count=0,
            search=fake_ocr_search,
        )
        with patch.object(dashboard, "get_ocr_index", return_value=fake_ocr):
            _stored, _note = dashboard.make_kis_results(
                fake_engine,
                query,
                {"options": {"top_k": 5, "dedupe": 0, "max_per_video": 0}},
                profile=profile,
                analysis=analysis,
                reranker=None,
            )
        self.assertEqual(fake_engine.queries, ["Biển cảnh báo màu vàng"])
        self.assertEqual(ocr_queries, ["cảnh báo sạt lở nguy hiểm"])

    def test_video_route_supports_browser_byte_ranges(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".mp4") as video:
            video.write(b"0123456789")
            video.flush()
            result = make_result("L03_V003", 400, video_path=video.name)
            dashboard.SESSIONS[self.session_id] = SearchSession(
                task="kis",
                results={"video": StoredResult(result)},
            )
            response = self.client.get("/video/video", headers={"Range": "bytes=2-5"})
            status_code = response.status_code
            payload = response.data
            response.close()
        self.assertEqual(status_code, 206)
        self.assertEqual(payload, b"2345")


if __name__ == "__main__":
    unittest.main()
