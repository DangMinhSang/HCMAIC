from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch


CODE = Path(__file__).resolve().parents[1] / "Code"
if str(CODE) not in sys.path:
    sys.path.append(str(CODE))

import dashboard
from dashboard import SearchSession, StoredResult
from retrieval import SearchResult, TrakeVideoResult


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

    def test_kis_and_trake_frame_overrides_reach_csv(self) -> None:
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
