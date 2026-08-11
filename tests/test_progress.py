from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


CODE = Path(__file__).resolve().parents[1] / "Code"
if str(CODE) not in sys.path:
    sys.path.append(str(CODE))

import progress


class ProgressTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calls: list[dict[str, object]] = []

        def fake_tqdm(iterable, **kwargs):
            self.calls.append(kwargs)
            return iterable

        self.tqdm_patch = patch.object(progress, "_tqdm", fake_tqdm)
        self.tqdm_patch.start()

    def tearDown(self) -> None:
        self.tqdm_patch.stop()

    def test_tiny_loop_is_quiet_but_force_is_visible(self) -> None:
        with patch.dict(
            "os.environ",
            {"AIC_PROGRESS": "1", "AIC_PROGRESS_ALL": "0", "AIC_PROGRESS_MIN_ITEMS": "20"},
            clear=False,
        ):
            self.assertEqual(list(progress.track(range(3), desc="tiny")), [0, 1, 2])
            self.assertEqual(self.calls, [])
            self.assertEqual(
                list(progress.track(range(3), desc="expensive", force=True)),
                [0, 1, 2],
            )
        self.assertEqual(self.calls[0]["desc"], "expensive")

    def test_all_exposes_nested_loops(self) -> None:
        with patch.dict(
            "os.environ",
            {"AIC_PROGRESS": "1", "AIC_PROGRESS_ALL": "1"},
            clear=False,
        ):
            self.assertEqual(
                list(progress.track(range(2), desc="nested", nested=True)),
                [0, 1],
            )
        self.assertEqual(self.calls[0]["desc"], "nested")

    def test_global_switch_disables_progress(self) -> None:
        with patch.dict("os.environ", {"AIC_PROGRESS": "0"}, clear=False):
            self.assertEqual(list(progress.track(range(30), desc="off", force=True)), list(range(30)))
            self.assertFalse(progress.progress_enabled())
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
