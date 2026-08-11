from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

import run
from Code.ocr_regions import OCR_INDEX_SCHEMA_VERSION


class OCRSchemaTests(unittest.TestCase):
    def test_schema_falls_back_to_first_record_for_old_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_path = root / "ocr.jsonl.gz"
            marker_path = Path(f"{index_path}.complete")
            with gzip.open(index_path, "wt", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "ocr_schema": OCR_INDEX_SCHEMA_VERSION,
                            "video_id": "L01_V001",
                            "keyframe_number": 1,
                            "text": "physical sign",
                        }
                    )
                    + "\n"
                )
            marker_path.write_text('{"records": 1}\n', encoding="utf-8")
            self.assertEqual(
                run.ocr_index_schema(index_path, marker_path),
                OCR_INDEX_SCHEMA_VERSION,
            )

    def test_import_marks_legacy_index_for_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "legacy.jsonl.gz"
            destination = root / "runtime.jsonl.gz"
            with gzip.open(source, "wt", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "video_id": "L22_V021",
                            "keyframe_number": 204,
                            "text": "legacy ticker",
                        }
                    )
                    + "\n"
                )
            self.assertEqual(run.import_ocr_index(source, destination), 1)
            marker = Path(f"{destination}.complete")
            self.assertEqual(run.ocr_index_schema(destination, marker), 1)


if __name__ == "__main__":
    unittest.main()
