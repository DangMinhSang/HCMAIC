"""Pre-OCR mounted keyframes once, then serve OCR retrieval from RAM."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

from data_paths import AICPaths


def resolve_device(requested: str) -> str:
    """Require the existing Kaggle PyTorch GPU for full-dataset OCR."""
    try:
        import torch  # type: ignore
    except ImportError as error:
        raise RuntimeError("Không thể import PyTorch CUDA của Kaggle.") from error
    device = "gpu:0" if requested == "auto" else requested
    if device.startswith("gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Không có PyTorch GPU. Bật Accelerator = GPU trong Kaggle; "
                "không chạy pre-OCR full dataset bằng CPU. "
                "Chỉ dùng `--device cpu` cho smoke test nhỏ."
            )
    return device


def create_reader(language: str, device: str):
    try:
        import easyocr  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "Thiếu EasyOCR. Chạy `pip install -r Code/requirements-ocr.txt` trước khi build OCR index. "
            "Lỗi gốc: "
            f"{error}"
        ) from error
    # EasyOCR's Vietnamese model can be paired with English, which is useful
    # for road signs, TV overlays, and imported footage in the same keyframe.
    languages = ["vi", "en"] if language == "vi" else [language, "en"]
    return easyocr.Reader(languages, gpu=device.startswith("gpu"), verbose=False)


def read_text(reader, image_path: Path, minimum_confidence: float) -> str:
    """Extract EasyOCR lines as (box, text, confidence) tuples."""
    lines = [
        str(text).strip()
        for _box, text, confidence in reader.readtext(str(image_path), detail=1, paragraph=False)
        if str(text).strip() and float(confidence) >= minimum_confidence
    ]
    return " ".join(dict.fromkeys(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute a text-only OCR index for mounted AIC keyframes")
    parser.add_argument("--output", type=Path, required=True, help=".jsonl or .jsonl.gz OCR index path")
    parser.add_argument("--language", default="vi", help="EasyOCR language, default: vi")
    parser.add_argument("--min-confidence", type=float, default=0.45)
    parser.add_argument("--limit", type=int, default=0, help="For smoke tests; 0 means all keyframes")
    parser.add_argument("--video", default="", help="Only OCR one video id")
    parser.add_argument(
        "--device",
        default=os.environ.get("AIC_OCR_DEVICE", "auto"),
        help="OCR device; default gpu:0. Use cpu only for a small smoke test.",
    )
    arguments = parser.parse_args()
    try:
        from tqdm.auto import tqdm
    except ImportError as error:
        raise RuntimeError(
            "Thiếu tqdm. Chạy `pip install -r Code/requirements-ocr.txt` trước khi build OCR index."
        ) from error

    device = resolve_device(arguments.device)
    print(f"OCR device: {device}", flush=True)
    paths = AICPaths.from_environment()
    reader = create_reader(arguments.language, device)
    output = arguments.output
    output.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if output.suffix == ".gz" else open
    processed = written = consecutive_failures = 0
    video_dirs = [
        video_dir
        for keyframe_root in paths.keyframe_roots
        for video_dir in sorted((keyframe_root / "keyframes").iterdir())
        if video_dir.is_dir() and (not arguments.video or video_dir.name == arguments.video)
    ]
    total_frames = sum(sum(1 for _ in video_dir.glob("*.jpg")) for video_dir in video_dirs)
    print(f"Sẽ OCR {total_frames:,} keyframe trong {len(video_dirs):,} video.", flush=True)
    with opener(output, "wt", encoding="utf-8") as stream:
        with tqdm(
            total=total_frames,
            desc="OCR keyframes",
            unit="frame",
            dynamic_ncols=True,
        ) as frame_progress:
            for video_dir in tqdm(video_dirs, desc="Video", unit="video", leave=False, dynamic_ncols=True):
                for image_path in sorted(video_dir.glob("*.jpg")):
                    processed += 1
                    frame_progress.update(1)
                    try:
                        keyframe_number = int(image_path.stem)
                        text = read_text(reader, image_path, arguments.min_confidence)
                    except Exception as error:
                        consecutive_failures += 1
                        print(f"[skip] {image_path}: {error}", file=sys.stderr)
                        if consecutive_failures >= 5 and written == 0:
                            raise RuntimeError(
                                "OCR thất bại liên tiếp từ frame đầu. "
                                "Đã dừng để không tạo OCR index rỗng. Lỗi cuối: "
                                f"{error}"
                            ) from error
                        continue
                    consecutive_failures = 0
                    if text:
                        stream.write(
                            json.dumps(
                                {"video_id": video_dir.name, "keyframe_number": keyframe_number, "text": text},
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        written += 1
                    if processed % 25 == 0:
                        frame_progress.set_postfix(records=written, refresh=False)
                    if processed % 250 == 0:
                        stream.flush()
                    if arguments.limit and processed >= arguments.limit:
                        print(f"Hoàn thành smoke test: {processed:,} keyframes · {written:,} records")
                        return
    print(f"Hoàn thành: {processed:,} keyframes · {written:,} records → {output}")


if __name__ == "__main__":
    main()
