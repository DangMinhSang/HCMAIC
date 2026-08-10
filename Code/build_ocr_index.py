"""Pre-OCR mounted keyframes once, then serve OCR retrieval from RAM."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

from data_paths import AICPaths


def create_reader(language: str):
    try:
        from paddleocr import PaddleOCR  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "Thiếu PaddleOCR. Chạy `pip install -r Code/requirements-ocr.txt` trước khi build OCR index."
        ) from error
    try:
        return PaddleOCR(lang=language, use_angle_cls=False, show_log=False)
    except TypeError:
        # PaddleOCR 3.x removed several 2.x constructor options.
        return PaddleOCR(lang=language)


def read_text(reader, image_path: Path, minimum_confidence: float) -> str:
    """Extract recognized lines across PaddleOCR 2.x/3.x result shapes."""
    try:
        output = reader.ocr(str(image_path), cls=False)
    except AttributeError:
        output = list(reader.predict(str(image_path)))
    lines: list[str] = []

    def walk(node) -> None:
        if isinstance(node, (list, tuple)):
            if len(node) >= 2 and isinstance(node[1], (list, tuple)) and node[1]:
                candidate = node[1]
                if isinstance(candidate[0], str):
                    confidence = float(candidate[1]) if len(candidate) > 1 else 1.0
                    if confidence >= minimum_confidence:
                        lines.append(candidate[0].strip())
                    return
            for child in node:
                walk(child)
        elif isinstance(node, dict):
            for key in ("rec_texts", "rec_text", "text"):
                value = node.get(key)
                if isinstance(value, str) and value.strip():
                    lines.append(value.strip())
                elif isinstance(value, list):
                    lines.extend(str(item).strip() for item in value if str(item).strip())

    walk(output)
    return " ".join(dict.fromkeys(line for line in lines if line))


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute a text-only OCR index for mounted AIC keyframes")
    parser.add_argument("--output", type=Path, required=True, help=".jsonl or .jsonl.gz OCR index path")
    parser.add_argument("--language", default="vi", help="PaddleOCR language, default: vi")
    parser.add_argument("--min-confidence", type=float, default=0.45)
    parser.add_argument("--limit", type=int, default=0, help="For smoke tests; 0 means all keyframes")
    parser.add_argument("--video", default="", help="Only OCR one video id")
    arguments = parser.parse_args()

    paths = AICPaths.from_environment()
    reader = create_reader(arguments.language)
    output = arguments.output
    output.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if output.suffix == ".gz" else open
    processed = written = 0
    with opener(output, "wt", encoding="utf-8") as stream:
        for keyframe_root in paths.keyframe_roots:
            for video_dir in sorted((keyframe_root / "keyframes").iterdir()):
                if not video_dir.is_dir() or (arguments.video and video_dir.name != arguments.video):
                    continue
                for image_path in sorted(video_dir.glob("*.jpg")):
                    processed += 1
                    try:
                        keyframe_number = int(image_path.stem)
                        text = read_text(reader, image_path, arguments.min_confidence)
                    except Exception as error:
                        print(f"[skip] {image_path}: {error}", file=sys.stderr)
                        continue
                    if text:
                        stream.write(
                            json.dumps(
                                {"video_id": video_dir.name, "keyframe_number": keyframe_number, "text": text},
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        written += 1
                    if processed % 250 == 0:
                        stream.flush()
                        print(f"OCR {processed:,} keyframes · {written:,} có text", flush=True)
                    if arguments.limit and processed >= arguments.limit:
                        print(f"Hoàn thành smoke test: {processed:,} keyframes · {written:,} records")
                        return
    print(f"Hoàn thành: {processed:,} keyframes · {written:,} records → {output}")


if __name__ == "__main__":
    main()
