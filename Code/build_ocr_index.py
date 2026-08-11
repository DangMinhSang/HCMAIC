"""Pre-OCR mounted keyframes once, then serve OCR retrieval from RAM."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

from data_paths import AICPaths
from ocr_regions import (
    OCR_INDEX_SCHEMA_VERSION,
    bottom_overlay_start,
    is_broadcast_overlay_box,
    join_unique_lines,
)
from progress import track


def resolve_device(requested: str) -> str:
    """Require the isolated Paddle GPU runtime for full-dataset OCR."""
    try:
        import paddle  # type: ignore
    except ImportError as error:
        raise RuntimeError("Không thể import PaddlePaddle trong OCR virtualenv.") from error
    device = "gpu:0" if requested == "auto" else requested
    if device.startswith("gpu"):
        if not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() < 1:
            raise RuntimeError(
                "Không có Paddle GPU. Bật Accelerator = GPU trong Kaggle; "
                "không chạy pre-OCR full dataset bằng CPU. "
                "Chỉ dùng `--device cpu` cho smoke test nhỏ."
            )
    return device


def create_reader(language: str, device: str):
    try:
        from paddleocr import PaddleOCR  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "Thiếu PaddleOCR trong OCR virtualenv. "
            "Lỗi gốc: "
            f"{error}"
        ) from error
    # PaddleOCR 2.9's compact API does not import PaddleX/ModelScope/Torch.
    # A larger recognition batch keeps GPU busy when a frame contains many
    # subtitle/sign text regions, while preserving Vietnamese accuracy.
    return PaddleOCR(
        lang=language,
        use_gpu=device.startswith("gpu"),
        use_angle_cls=False,
        rec_batch_num=int(os.environ.get("AIC_OCR_REC_BATCH", "32")),
        show_log=False,
    )


def read_text(reader, image_path: Path, minimum_confidence: float) -> tuple[str, str, int, int]:
    """Return scene and broadcast-overlay text as separate values."""
    try:
        from PIL import Image  # type: ignore
    except ImportError as error:
        raise RuntimeError("Thiếu Pillow để phân vùng OCR theo kích thước ảnh.") from error
    with Image.open(image_path) as image:
        image_width, image_height = image.size
    output = reader.ocr(str(image_path), cls=False)
    scene_lines: list[str] = []
    overlay_lines: list[str] = []
    pages = output or []
    for page in track(
        pages,
        desc="OCR pages",
        total=len(pages),
        unit="page",
        nested=True,
    ):
        items = page or []
        for item in track(
            items,
            desc="OCR text boxes",
            total=len(items),
            unit="box",
            nested=True,
        ):
            try:
                polygon = item[0]
                text, confidence = item[1]
            except (IndexError, TypeError, ValueError):
                continue
            clean_text = str(text).strip()
            if not clean_text or float(confidence) < minimum_confidence:
                continue
            if is_broadcast_overlay_box(
                polygon,
                clean_text,
                image_width,
                image_height,
            ):
                overlay_lines.append(clean_text)
            else:
                scene_lines.append(clean_text)
    return (
        join_unique_lines(scene_lines),
        join_unique_lines(overlay_lines),
        len(scene_lines),
        len(overlay_lines),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute a text-only OCR index for mounted AIC keyframes")
    parser.add_argument("--output", type=Path, required=True, help=".jsonl or .jsonl.gz OCR index path")
    parser.add_argument("--language", default="vi", help="PaddleOCR language, default: vi")
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
    complete_marker = Path(f"{output}.complete")
    # A stopped notebook leaves a partial JSONL behind. It must never be
    # mistaken for a complete OCR index by the next Run all execution.
    complete_marker.unlink(missing_ok=True)
    opener = gzip.open if output.suffix == ".gz" else open
    processed = written = consecutive_failures = 0
    suppressed_overlay_lines = overlay_only_frames = 0
    video_dirs = [
        video_dir
        for keyframe_root in paths.keyframe_roots
        for video_dir in sorted((keyframe_root / "keyframes").iterdir())
        if video_dir.is_dir() and (not arguments.video or video_dir.name == arguments.video)
    ]
    total_frames = sum(sum(1 for _ in video_dir.glob("*.jpg")) for video_dir in video_dirs)
    print(f"Sẽ OCR {total_frames:,} keyframe trong {len(video_dirs):,} video.", flush=True)
    print(
        "Bỏ chữ TV overlay ở đáy từ "
        f"{bottom_overlay_start():.0%} chiều cao và logo/đồng hồ góc màn hình.",
        flush=True,
    )
    if output.suffix == ".gz":
        temporary_output = output.with_name(output.name.removesuffix(".gz") + ".building.gz")
    else:
        temporary_output = output.with_name(output.name + ".building")
    temporary_output.unlink(missing_ok=True)
    limit_reached = False
    with opener(temporary_output, "wt", encoding="utf-8") as stream:
        with tqdm(
            total=total_frames,
            desc="OCR keyframes",
            unit="frame",
            dynamic_ncols=True,
        ) as frame_progress:
            for video_dir in tqdm(video_dirs, desc="Video", unit="video", leave=False, dynamic_ncols=True):
                image_paths = sorted(video_dir.glob("*.jpg"))
                for image_path in track(
                    image_paths,
                    desc=f"OCR {video_dir.name}",
                    total=len(image_paths),
                    unit="frame",
                    nested=True,
                ):
                    processed += 1
                    frame_progress.update(1)
                    try:
                        keyframe_number = int(image_path.stem)
                        text, overlay_text, _scene_count, overlay_count = read_text(
                            reader,
                            image_path,
                            arguments.min_confidence,
                        )
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
                    suppressed_overlay_lines += overlay_count
                    if overlay_text and not text:
                        overlay_only_frames += 1
                    if text:
                        stream.write(
                            json.dumps(
                                {
                                    "ocr_schema": OCR_INDEX_SCHEMA_VERSION,
                                    "video_id": video_dir.name,
                                    "keyframe_number": keyframe_number,
                                    "text": text,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        written += 1
                    if processed % 25 == 0:
                        frame_progress.set_postfix(
                            records=written,
                            overlay=suppressed_overlay_lines,
                            refresh=False,
                        )
                    if processed % 250 == 0:
                        stream.flush()
                    if arguments.limit and processed >= arguments.limit:
                        limit_reached = True
                        break
                if limit_reached:
                    break
    os.replace(temporary_output, output)
    if limit_reached:
        print(
            f"Hoàn thành smoke test: {processed:,} keyframes · {written:,} records · "
            f"bỏ {suppressed_overlay_lines:,} dòng overlay"
        )
        return
    complete_marker.write_text(
        json.dumps(
            {
                "ocr_schema": OCR_INDEX_SCHEMA_VERSION,
                "keyframes": processed,
                "records": written,
                "suppressed_overlay_lines": suppressed_overlay_lines,
                "overlay_only_frames": overlay_only_frames,
                "bottom_overlay_start": bottom_overlay_start(),
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Hoàn thành: {processed:,} keyframes · {written:,} records · "
        f"bỏ {suppressed_overlay_lines:,} dòng overlay ({overlay_only_frames:,} frame chỉ có overlay) "
        f"→ {output}"
    )


if __name__ == "__main__":
    main()
