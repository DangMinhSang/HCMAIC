"""Shardable offline preprocessing for raw AIC videos.

Artifacts are generated per video so an interrupted Kaggle session can resume
without recomputing completed videos. The visual stage decodes each source MP4
once, writes sampled PNG frames, then batches CLIP and YOLO inference. OCR runs
as a separate stage because PaddleOCR is intentionally installed in an
isolated package directory by ``run.py``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import count
from pathlib import Path
from typing import Any, Iterable, Sequence

from progress import track


DIRECT_PREPROCESS_SCHEMA = 2
DEFAULT_OUTPUT_ROOT = Path("/kaggle/working/aic_direct_preprocessed")


_VISUAL_WORKER_STATE: dict[str, Any] = {}
_OCR_WORKER_STATE: dict[str, Any] = {}


@dataclass(frozen=True)
class VideoWindow:
    """A deterministic, human-facing 1-based inclusive video slice."""

    videos: tuple[Path, ...]
    start: int
    end: int
    total: int


@dataclass(frozen=True)
class VideoArtifacts:
    root: Path
    frames_dir: Path
    mapping: Path
    frame_ids: Path
    pts_times: Path
    clip: Path
    objects: Path
    object_scores: Path
    object_classes: Path
    visual_marker: Path
    ocr: Path
    ocr_marker: Path
    complete_marker: Path


def discover_gpu_ids() -> tuple[str, ...]:
    """Return CUDA devices visible to this Kaggle process."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible != "-1":
        devices = tuple(part.strip() for part in visible.split(",") if part.strip())
        if devices:
            return devices
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as error:
        raise RuntimeError(
            "Không tìm thấy nvidia-smi; --pre-direct-video cần Kaggle GPU."
        ) from error
    devices = tuple(line.strip() for line in result.stdout.splitlines() if line.strip())
    if result.returncode != 0 or not devices:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            "Không phát hiện CUDA GPU cho direct preprocessing. "
            + (detail or "Bật Accelerator GPU rồi restart Kaggle session.")
        )
    return devices


def parse_gpu_ids(value: str, *, discovered: Sequence[str] | None = None) -> tuple[str, ...]:
    """Parse ``auto`` or a unique CUDA device list such as ``0,1``."""
    requested = str(value or "").strip()
    if requested.casefold() == "auto":
        available = discover_gpu_ids() if discovered is None else discovered
        devices = tuple(str(device).strip() for device in available)
    else:
        devices = tuple(part.strip() for part in requested.split(",") if part.strip())
    if not devices:
        raise ValueError("--gpus phải là auto hoặc danh sách như 0,1.")
    if any(
        not (device.isdigit() or device.startswith(("GPU-", "MIG-")))
        for device in devices
    ):
        raise ValueError("GPU id không hợp lệ; dùng auto hoặc danh sách như 0,1.")
    if len(set(devices)) != len(devices):
        raise ValueError("Danh sách GPU bị trùng; mỗi worker phải sở hữu một GPU riêng.")
    return devices


def resolve_worker_gpu_ids(
    value: str,
    workers: int,
    *,
    discovered: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Resolve one distinct CUDA device per preprocessing worker."""
    devices = parse_gpu_ids(value, discovered=discovered)
    if str(value or "").strip().casefold() != "auto" and discovered is not None:
        available = {str(device).strip() for device in discovered}
        missing = [device for device in devices if device not in available]
        if missing:
            raise ValueError(
                f"GPU {','.join(missing)} không nằm trong CUDA devices nhìn thấy: "
                f"{','.join(sorted(available)) or 'none'}."
            )
    if workers < 0:
        raise ValueError("--workers phải >= 0; 0 nghĩa là một worker cho mỗi GPU.")
    worker_count = len(devices) if workers == 0 else workers
    if worker_count < 1:
        raise ValueError("Direct preprocessing cần ít nhất một GPU worker.")
    if worker_count > len(devices):
        raise ValueError(
            f"Yêu cầu {worker_count} worker nhưng chỉ cấu hình {len(devices)} GPU; "
            "không chạy hai model worker trên cùng một T4."
        )
    return devices[:worker_count]


class TemporalFrameSampler:
    """Select frames at a stable rate even when source FPS varies."""

    def __init__(self, source_fps: float, sample_fps: float) -> None:
        self.source_fps = source_fps if source_fps > 0 else 30.0
        self.sample_fps = max(0.0, sample_fps)
        self.interval = (
            1.0
            if self.sample_fps <= 0 or self.sample_fps >= self.source_fps
            else self.source_fps / self.sample_fps
        )
        self.next_frame = 0.0

    def accept(self, frame_index: int) -> bool:
        if self.sample_fps <= 0 or self.sample_fps >= self.source_fps:
            return True
        if frame_index + 1e-6 < self.next_frame:
            return False
        while self.next_frame <= frame_index + 1e-6:
            self.next_frame += self.interval
        return True


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def artifact_paths(output_root: str | Path, video_id: str) -> VideoArtifacts:
    root = Path(output_root).expanduser() / "videos" / video_id
    return VideoArtifacts(
        root=root,
        frames_dir=root / "frames",
        mapping=root / "mapping.jsonl",
        frame_ids=root / "frame_ids.npy",
        pts_times=root / "pts_times.npy",
        clip=root / "clip.npy",
        objects=root / "objects.jsonl.gz",
        object_scores=root / "object_scores.npy",
        object_classes=root / "object_classes.json",
        visual_marker=root / "visual.complete.json",
        ocr=root / "ocr.jsonl.gz",
        ocr_marker=root / "ocr.complete.json",
        complete_marker=root / "complete.json",
    )


def select_video_window(
    video_files: Sequence[str | Path],
    start: int,
    end: int,
) -> VideoWindow:
    """Sort by video id and return the requested 1-based inclusive range.

    ``end=0`` means the final video. This convention is explicit because the
    source corpus currently has 873 videos and preprocessing is normally split
    across multiple Kaggle sessions.
    """
    ordered = tuple(sorted((Path(path) for path in video_files), key=lambda path: (path.stem, str(path))))
    total = len(ordered)
    if total == 0:
        raise ValueError("Direct-video dataset không có video nào.")
    video_ids = [path.stem for path in ordered]
    duplicate_ids = sorted(video_id for video_id, count_ in Counter(video_ids).items() if count_ > 1)
    if duplicate_ids:
        raise ValueError(
            f"Trùng video_id {duplicate_ids[0]!r}; không thể ghi artifact theo thư mục an toàn."
        )
    if start < 1:
        raise ValueError("--start-pre-video dùng chỉ số 1-based và phải >= 1.")
    resolved_end = total if end == 0 else end
    if resolved_end < start:
        raise ValueError("--end-pre-video phải >= --start-pre-video; dùng 0 để chạy tới video cuối.")
    if start > total or resolved_end > total:
        raise ValueError(
            f"Khoảng video [{start}, {resolved_end}] vượt corpus 1..{total}. "
            "Danh sách được sort theo video_id."
        )
    return VideoWindow(ordered[start - 1 : resolved_end], start, resolved_end, total)


def write_video_order(video_files: Sequence[Path], output_root: Path) -> None:
    """Persist the exact ordinal-to-video mapping used by shard arguments."""
    records: list[dict[str, Any]] = []
    for index, path in track(
        enumerate(video_files, start=1),
        desc="Ghi thứ tự direct video",
        total=len(video_files),
        unit="video",
        force=True,
    ):
        records.append(
            {
                "index": index,
                "video_id": path.stem,
                "source": str(path),
            }
        )
    write_json_atomic(
        output_root / "video_order.json",
        {
            "schema": DIRECT_PREPROCESS_SCHEMA,
            "indexing": "1-based; --start-pre-video/--end-pre-video are inclusive",
            "total": len(records),
            "videos": records,
            "updated_at": utc_now(),
        },
    )


def estimated_sample_count(frame_count: int, source_fps: float, sample_fps: float) -> int:
    if frame_count <= 0:
        return 0
    if sample_fps <= 0 or sample_fps >= source_fps:
        return frame_count
    return max(1, int(math.ceil(frame_count * sample_fps / max(source_fps, 1e-6))))


def _temporary_path(path: Path, label: str = "building") -> Path:
    suffixes = "".join(path.suffixes)
    base = path.name.removesuffix(suffixes) if suffixes else path.name
    return path.with_name(f"{base}.{label}{suffixes}")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in track(stream, desc=f"Đọc {path.name}", unit="record", nested=True):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def marker_matches(path: Path, expected: dict[str, Any], required: Sequence[Path]) -> bool:
    marker = read_json(path)
    if not marker or any(not candidate.is_file() for candidate in required):
        return False
    return all(marker.get(key) == value for key, value in expected.items())


def _resize_image(image: Any, max_side: int) -> Any:
    if max_side <= 0 or max(image.size) <= max_side:
        return image
    from PIL import Image  # type: ignore

    resized = image.copy()
    resized.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    image.close()
    return resized


def _clip_image(image: Any, mask_overlays: bool) -> Any:
    """Prepare an in-memory CLIP view while preserving the original PNG."""
    clip_image = image.copy()
    if not mask_overlays:
        return clip_image
    from PIL import ImageEnhance, ImageFilter  # type: ignore
    from ocr_regions import bottom_overlay_start

    width, height = clip_image.size
    start = max(0, min(height, int(round(height * bottom_overlay_start()))))
    if start < height:
        region = clip_image.crop((0, start, width, height))
        region = region.filter(ImageFilter.GaussianBlur(radius=max(8.0, height * 0.025)))
        region = ImageEnhance.Contrast(region).enhance(0.25)
        clip_image.paste(region, (0, start))
        region.close()
    return clip_image


def _object_names(detector: Any) -> dict[int, str]:
    names = getattr(detector, "names", {})
    if isinstance(names, dict):
        return {int(index): str(label) for index, label in names.items()}
    if isinstance(names, (list, tuple)):
        return {index: str(label) for index, label in enumerate(names)}
    return {}


def create_object_detector(model_name: str) -> Any:
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "Thiếu Ultralytics cho direct object preprocessing. "
            "Chạy lại run.py để cài requirements-direct-video-preprocess.txt."
        ) from error
    try:
        return YOLO(model_name)
    except Exception as error:
        raise RuntimeError(
            f"Không tải được object model {model_name!r}. Bật Internet lần đầu hoặc mount checkpoint."
        ) from error


def _detections_from_result(
    result: Any,
    names: dict[int, str],
    confidence_threshold: float,
) -> tuple[list[dict[str, Any]], Any]:
    import numpy as np

    class_count = max(names, default=-1) + 1
    scores = np.zeros(class_count, dtype=np.float16)
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return [], scores
    xyxy = boxes.xyxy.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy()
    class_ids = boxes.cls.detach().cpu().numpy().astype(int)
    detections: list[dict[str, Any]] = []
    for index in track(
        range(len(class_ids)),
        desc="Chuẩn hóa object boxes",
        total=len(class_ids),
        unit="box",
        nested=True,
    ):
        confidence = float(confidences[index])
        if confidence < confidence_threshold:
            continue
        class_id = int(class_ids[index])
        if 0 <= class_id < len(scores):
            scores[class_id] = max(float(scores[class_id]), confidence)
        detections.append(
            {
                "label": names.get(class_id, str(class_id)),
                "class_id": class_id,
                "confidence": round(confidence, 6),
                "bbox_xyxy": [round(float(value), 2) for value in xyxy[index]],
            }
        )
    detections.sort(key=lambda item: float(item["confidence"]), reverse=True)
    return detections, scores


def preprocess_visual_video(
    video_path: Path,
    output_root: Path,
    *,
    encoder: Any,
    detector: Any,
    sample_fps: float,
    max_side: int,
    clip_batch: int,
    object_batch: int,
    object_model: str,
    object_device: str,
    object_confidence: float,
    mask_clip_overlays: bool,
    force: bool,
) -> dict[str, Any]:
    """Decode one video once and create PNG/CLIP/object artifacts."""
    visual_started = time.perf_counter()
    try:
        import cv2  # type: ignore
        import numpy as np
        from PIL import Image  # type: ignore
    except ImportError as error:
        raise RuntimeError("Direct preprocessing cần OpenCV, NumPy và Pillow.") from error
    from ocr_regions import bottom_overlay_start

    video_id = video_path.stem
    artifacts = artifact_paths(output_root, video_id)
    source_stat = video_path.stat()
    expected = {
        "schema": DIRECT_PREPROCESS_SCHEMA,
        "video_id": video_id,
        "source_size": int(source_stat.st_size),
        "source_mtime_ns": int(source_stat.st_mtime_ns),
        "sample_fps": sample_fps,
        "max_side": max_side,
        "clip_model": str(encoder.model_name),
        "object_model": object_model,
        "object_confidence": object_confidence,
        "mask_clip_overlays": mask_clip_overlays,
        "clip_overlay_start": bottom_overlay_start() if mask_clip_overlays else None,
    }
    required = (
        artifacts.mapping,
        artifacts.frame_ids,
        artifacts.pts_times,
        artifacts.clip,
        artifacts.objects,
        artifacts.object_scores,
        artifacts.object_classes,
    )
    marker_ready = not force and marker_matches(artifacts.visual_marker, expected, required)
    marker = read_json(artifacts.visual_marker) if marker_ready else {}
    saved_png_count = (
        sum(1 for path in artifacts.frames_dir.glob("*.png") if path.stem.isdigit())
        if marker_ready
        else 0
    )
    if marker_ready and saved_png_count == int(marker.get("sampled_frames") or -1):
        print(f"[skip] Visual {video_id}: {marker.get('sampled_frames', 0):,} frame đã hoàn tất.", flush=True)
        return marker
    if marker_ready:
        print(
            f"[rebuild] Visual {video_id}: marker có {marker.get('sampled_frames', 0)} PNG "
            f"nhưng disk có {saved_png_count}.",
            flush=True,
        )

    artifacts.frames_dir.mkdir(parents=True, exist_ok=True)
    artifacts.visual_marker.unlink(missing_ok=True)
    stale_frames = [
        path
        for path in sorted(artifacts.frames_dir.glob("*.png"))
        if path.stem.isdigit()
    ]
    for stale_frame in track(
        stale_frames,
        desc=f"Dọn PNG cũ {video_id}",
        total=len(stale_frames),
        unit="frame",
        force=bool(stale_frames),
    ):
        stale_frame.unlink(missing_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Không mở được video {video_path}.")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    source_fps = source_fps if source_fps > 0 else 30.0
    sampler = TemporalFrameSampler(source_fps, sample_fps)
    names = _object_names(detector)
    if not names:
        capture.release()
        raise RuntimeError(f"Object model {object_model!r} không trả class names.")

    mapping_records: list[dict[str, Any]] = []
    clip_parts: list[Any] = []
    object_score_rows: list[Any] = []
    object_temporary = _temporary_path(artifacts.objects)
    object_temporary.unlink(missing_ok=True)
    image_batch: list[Any] = []
    clip_image_batch: list[Any] = []
    record_batch: list[dict[str, Any]] = []
    batch_limit = max(1, min(int(clip_batch), int(object_batch)))
    png_seconds = clip_seconds = object_seconds = 0.0

    def flush_batch(object_stream: Any) -> None:
        nonlocal clip_seconds, object_seconds
        if not image_batch:
            return
        clip_started = time.perf_counter()
        vectors = encoder.encode_images(
            clip_image_batch,
            batch_size=max(1, int(clip_batch)),
            progress_desc=f"CLIP direct {video_id}",
            nested=True,
        )
        clip_seconds += time.perf_counter() - clip_started
        object_started = time.perf_counter()
        predictions = detector.predict(
            source=image_batch,
            batch=max(1, int(object_batch)),
            conf=object_confidence,
            device=object_device,
            verbose=False,
        )
        object_seconds += time.perf_counter() - object_started
        if len(vectors) != len(record_batch) or len(predictions) != len(record_batch):
            raise RuntimeError(f"CLIP/object batch không khớp frame ở {video_id}.")
        clip_parts.append(np.asarray(vectors, dtype=np.float32))
        for record, prediction in track(
            zip(record_batch, predictions),
            desc=f"Ghi object {video_id}",
            total=len(record_batch),
            unit="frame",
            nested=True,
        ):
            detections, score_row = _detections_from_result(prediction, names, object_confidence)
            object_score_rows.append(score_row)
            object_stream.write(
                json.dumps({**record, "objects": detections}, ensure_ascii=False) + "\n"
            )
        for image in track(
            (*image_batch, *clip_image_batch),
            desc="Đóng direct image batch",
            total=len(image_batch) + len(clip_image_batch),
            unit="image",
            nested=True,
        ):
            image.close()
        image_batch.clear()
        clip_image_batch.clear()
        record_batch.clear()

    # CAP_PROP_FRAME_COUNT is only an estimate for some codecs. Decode until
    # read() fails so all-frame mode never truncates a video at stale metadata.
    frame_iter: Iterable[int] = count()
    estimated = estimated_sample_count(frame_count, source_fps, sample_fps)
    decoded_frame_count = 0
    try:
        with gzip.open(object_temporary, "wt", encoding="utf-8") as object_stream:
            for frame_index in track(
                frame_iter,
                desc=f"Decode/sample {video_id}",
                total=frame_count or None,
                unit="frame",
                force=True,
                leave=True,
            ):
                ok, bgr = capture.read()
                if not ok:
                    break
                decoded_frame_count = frame_index + 1
                if not sampler.accept(frame_index):
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                image = _resize_image(Image.fromarray(rgb), max_side)
                filename = f"{frame_index:09d}.png"
                image_path = artifacts.frames_dir / filename
                image_temporary = image_path.with_name(f"{image_path.stem}.building.png")
                png_started = time.perf_counter()
                image.save(image_temporary, format="PNG", compress_level=3)
                image_temporary.replace(image_path)
                png_seconds += time.perf_counter() - png_started
                sample_index = len(mapping_records)
                record = {
                    "schema": DIRECT_PREPROCESS_SCHEMA,
                    "video_id": video_id,
                    "keyframe_number": sample_index,
                    "sample_index": sample_index,
                    "frame_id": int(frame_index),
                    "pts_time": float(frame_index) / source_fps,
                    "fps": source_fps,
                    "image": f"frames/{filename}",
                    "width": int(image.width),
                    "height": int(image.height),
                }
                mapping_records.append(record)
                record_batch.append(record)
                image_batch.append(image)
                clip_image_batch.append(_clip_image(image, mask_clip_overlays))
                if len(image_batch) >= batch_limit:
                    flush_batch(object_stream)
            flush_batch(object_stream)
    finally:
        capture.release()

    if not mapping_records or not clip_parts:
        raise RuntimeError(f"Không cắt được frame nào từ {video_path}.")
    clip_matrix = np.concatenate(clip_parts, axis=0).astype(np.float32, copy=False)
    object_matrix = np.stack(object_score_rows).astype(np.float16, copy=False)
    if len(clip_matrix) != len(mapping_records) or len(object_matrix) != len(mapping_records):
        raise RuntimeError(f"Artifact count không khớp mapping ở {video_id}.")

    mapping_temporary = _temporary_path(artifacts.mapping)
    with mapping_temporary.open("w", encoding="utf-8") as stream:
        for record in track(
            mapping_records,
            desc=f"Ghi mapping {video_id}",
            total=len(mapping_records),
            unit="frame",
        ):
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    clip_temporary = _temporary_path(artifacts.clip)
    frame_ids_temporary = _temporary_path(artifacts.frame_ids)
    pts_times_temporary = _temporary_path(artifacts.pts_times)
    object_scores_temporary = _temporary_path(artifacts.object_scores)
    frame_ids = np.empty(len(mapping_records), dtype=np.int64)
    pts_times = np.empty(len(mapping_records), dtype=np.float32)
    for index, record in track(
        enumerate(mapping_records),
        desc=f"Đóng gói frame arrays {video_id}",
        total=len(mapping_records),
        unit="frame",
        force=True,
    ):
        frame_ids[index] = int(record["frame_id"])
        pts_times[index] = float(record["pts_time"])
    with clip_temporary.open("wb") as stream:
        np.save(stream, clip_matrix, allow_pickle=False)
    with frame_ids_temporary.open("wb") as stream:
        np.save(stream, frame_ids, allow_pickle=False)
    with pts_times_temporary.open("wb") as stream:
        np.save(stream, pts_times, allow_pickle=False)
    with object_scores_temporary.open("wb") as stream:
        np.save(stream, object_matrix, allow_pickle=False)
    write_json_atomic(
        artifacts.object_classes,
        {"schema": DIRECT_PREPROCESS_SCHEMA, "model": object_model, "classes": names},
    )
    mapping_temporary.replace(artifacts.mapping)
    frame_ids_temporary.replace(artifacts.frame_ids)
    pts_times_temporary.replace(artifacts.pts_times)
    clip_temporary.replace(artifacts.clip)
    object_scores_temporary.replace(artifacts.object_scores)
    object_temporary.replace(artifacts.objects)
    marker = {
        **expected,
        "source_video": str(video_path),
        "source_fps": source_fps,
        "source_frames": decoded_frame_count,
        "reported_source_frames": frame_count,
        "all_frames": bool(
            (sample_fps <= 0 or sample_fps >= source_fps)
            and (frame_count <= 0 or decoded_frame_count >= frame_count)
        ),
        "estimated_samples": estimated,
        "sampled_frames": len(mapping_records),
        "clip_shape": list(clip_matrix.shape),
        "object_shape": list(object_matrix.shape),
        "worker_gpu": os.environ.get("AIC_PRE_DIRECT_WORKER_GPU", ""),
        "timing_seconds": {
            "total": round(time.perf_counter() - visual_started, 3),
            "png": round(png_seconds, 3),
            "clip": round(clip_seconds, 3),
            "object": round(object_seconds, 3),
        },
        "generated_at": utc_now(),
    }
    write_json_atomic(artifacts.visual_marker, marker)
    if sample_fps <= 0 and not marker["all_frames"]:
        print(
            f"[warning] {video_id}: decoder dừng ở {decoded_frame_count:,}/"
            f"{frame_count:,} frame được container báo; marker không được coi là all-frame.",
            flush=True,
        )
    print(
        f"Visual {video_id}: {len(mapping_records):,} PNG · CLIP {clip_matrix.shape} · "
        f"object {object_matrix.shape}",
        flush=True,
    )
    return marker


def run_visual_stage(
    window: VideoWindow,
    output_root: Path,
    arguments: argparse.Namespace,
) -> None:
    available = discover_gpu_ids()
    gpu_ids = resolve_worker_gpu_ids(
        arguments.gpus,
        arguments.workers,
        discovered=available,
    )
    _run_gpu_stage(
        stage="visual",
        window=window,
        output_root=output_root,
        arguments=arguments,
        gpu_ids=gpu_ids,
        local_initializer=_initialize_visual_worker,
        pool_initializer=_visual_pool_initializer,
        task=_visual_worker_task,
    )


def _bind_cuda_worker(gpu_id: str, stage: str) -> None:
    """Expose one physical GPU as logical device zero before model imports."""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["AIC_PRE_DIRECT_WORKER_GPU"] = str(gpu_id)
    print(
        f"[{stage} worker pid={os.getpid()}] physical GPU {gpu_id} → logical GPU 0",
        flush=True,
    )


def _with_optional_lock(lock: Any, callback: Any) -> Any:
    if lock is None:
        return callback()
    with lock:
        return callback()


def _initialize_visual_worker(
    gpu_id: str,
    initialization_lock: Any,
    output_root: str | Path,
    arguments: argparse.Namespace,
) -> None:
    """Load one CLIP and one YOLO instance on a worker's isolated GPU."""
    _bind_cuda_worker(gpu_id, "visual")

    def load_models() -> tuple[Any, Any]:
        from clip_encoder import ClipTextEncoder
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError(
                f"Visual worker GPU {gpu_id} không được cô lập thành đúng một CUDA device."
            )
        print(
            f"[visual GPU {gpu_id}] CUDA logical 0 = {torch.cuda.get_device_name(0)}",
            flush=True,
        )

        encoder = ClipTextEncoder(model_name=arguments.clip_model, device="cuda:0")
        encoder.warmup()
        detector = create_object_detector(arguments.object_model)
        return encoder, detector

    encoder, detector = _with_optional_lock(initialization_lock, load_models)
    _VISUAL_WORKER_STATE.clear()
    _VISUAL_WORKER_STATE.update(
        gpu_id=str(gpu_id),
        output_root=Path(output_root),
        arguments=arguments,
        encoder=encoder,
        detector=detector,
    )
    print(f"[visual GPU {gpu_id}] CLIP + YOLO sẵn sàng.", flush=True)


def _visual_pool_initializer(
    gpu_queue: Any,
    initialization_lock: Any,
    output_root: str | Path,
    arguments: argparse.Namespace,
) -> None:
    _initialize_visual_worker(gpu_queue.get(), initialization_lock, output_root, arguments)


def _visual_worker_task(ordinal: int, total: int, video_path_value: str) -> dict[str, Any]:
    if not _VISUAL_WORKER_STATE:
        raise RuntimeError("Visual GPU worker chưa được khởi tạo.")
    video_path = Path(video_path_value)
    arguments = _VISUAL_WORKER_STATE["arguments"]
    gpu_id = str(_VISUAL_WORKER_STATE["gpu_id"])
    started = time.perf_counter()
    print(f"[{ordinal}/{total}] [GPU {gpu_id}] visual {video_path.stem}", flush=True)
    marker = preprocess_visual_video(
        video_path,
        _VISUAL_WORKER_STATE["output_root"],
        encoder=_VISUAL_WORKER_STATE["encoder"],
        detector=_VISUAL_WORKER_STATE["detector"],
        sample_fps=arguments.sample_fps,
        max_side=arguments.max_side,
        clip_batch=arguments.clip_batch,
        object_batch=arguments.object_batch,
        object_model=arguments.object_model,
        object_device="0",
        object_confidence=arguments.object_confidence,
        mask_clip_overlays=arguments.mask_clip_overlays,
        force=arguments.force,
    )
    return {
        "stage": "visual",
        "video_id": video_path.stem,
        "gpu_id": gpu_id,
        "frames": int(marker.get("sampled_frames") or 0),
        "seconds": time.perf_counter() - started,
    }


def _print_stage_result(result: dict[str, Any]) -> None:
    print(
        f"[done] {result['stage']} {result['video_id']} · GPU {result['gpu_id']} · "
        f"{int(result.get('frames') or 0):,} frame · {float(result['seconds']):.1f}s",
        flush=True,
    )


def _run_gpu_stage(
    *,
    stage: str,
    window: VideoWindow,
    output_root: Path,
    arguments: argparse.Namespace,
    gpu_ids: tuple[str, ...],
    local_initializer: Any,
    pool_initializer: Any,
    task: Any,
) -> list[dict[str, Any]]:
    """Run dynamically scheduled per-video jobs with one process per GPU."""
    stage_started = time.perf_counter()
    print(
        f"Direct {stage}: {len(gpu_ids)} worker · GPU {','.join(gpu_ids)} · "
        f"{len(window.videos)} video",
        flush=True,
    )
    jobs: list[tuple[int, int, str]] = []
    for ordinal, video_path in track(
        enumerate(window.videos, start=window.start),
        desc=f"Lập hàng đợi {stage}",
        total=len(window.videos),
        unit="video",
        nested=True,
    ):
        jobs.append((ordinal, window.total, str(video_path)))
    results: list[dict[str, Any]] = []
    if len(gpu_ids) == 1:
        local_initializer(gpu_ids[0], None, output_root, arguments)
        for job in track(
            jobs,
            desc=f"{stage.title()} videos {window.start}..{window.end}",
            total=len(jobs),
            unit="video",
            force=True,
            leave=True,
        ):
            result = task(*job)
            results.append(result)
            _print_stage_result(result)
        elapsed = time.perf_counter() - stage_started
        print(
            f"Direct {stage} hoàn tất {len(results)}/{len(jobs)} video trong {elapsed:.1f}s.",
            flush=True,
        )
        return results

    context = mp.get_context("spawn")
    gpu_queue = context.Queue()
    for gpu_id in track(
        gpu_ids,
        desc=f"Gán GPU {stage}",
        total=len(gpu_ids),
        unit="GPU",
        nested=True,
    ):
        gpu_queue.put(gpu_id)
    initialization_lock = context.Lock()
    executor = ProcessPoolExecutor(
        max_workers=len(gpu_ids),
        mp_context=context,
        initializer=pool_initializer,
        initargs=(gpu_queue, initialization_lock, output_root, arguments),
    )
    futures = []
    for job in track(
        jobs,
        desc=f"Submit {stage} videos",
        total=len(jobs),
        unit="video",
        nested=True,
    ):
        futures.append(executor.submit(task, *job))
    try:
        completed = as_completed(futures)
        for future in track(
            completed,
            desc=f"{stage.title()} GPU workers",
            total=len(futures),
            unit="video",
            force=True,
            leave=True,
        ):
            result = future.result()
            results.append(result)
            _print_stage_result(result)
    except BaseException:
        for future in track(
            futures,
            desc=f"Hủy {stage} jobs",
            total=len(futures),
            unit="job",
            nested=True,
        ):
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    finally:
        gpu_queue.close()
        gpu_queue.join_thread()
    elapsed = time.perf_counter() - stage_started
    print(
        f"Direct {stage} hoàn tất {len(results)}/{len(jobs)} video trên "
        f"{len(gpu_ids)} GPU trong {elapsed:.1f}s.",
        flush=True,
    )
    return results


def preprocess_ocr_video(
    video_path: Path,
    output_root: Path,
    *,
    reader: Any,
    language: str,
    device: str,
    minimum_confidence: float,
    force: bool,
) -> dict[str, Any]:
    from build_ocr_index import read_text
    from ocr_regions import OCR_INDEX_SCHEMA_VERSION, bottom_overlay_start

    ocr_started = time.perf_counter()
    video_id = video_path.stem
    artifacts = artifact_paths(output_root, video_id)
    visual_marker = read_json(artifacts.visual_marker)
    if not visual_marker or not artifacts.mapping.is_file():
        raise RuntimeError(f"Chưa có visual artifacts cho {video_id}; chạy stage visual trước.")
    expected = {
        "schema": DIRECT_PREPROCESS_SCHEMA,
        "ocr_schema": OCR_INDEX_SCHEMA_VERSION,
        "video_id": video_id,
        "language": language,
        "minimum_confidence": minimum_confidence,
        "visual_sampled_frames": int(visual_marker.get("sampled_frames") or 0),
        "visual_generated_at": visual_marker.get("generated_at"),
        "bottom_overlay_start": bottom_overlay_start(),
    }
    if not force and marker_matches(artifacts.ocr_marker, expected, (artifacts.ocr,)):
        marker = read_json(artifacts.ocr_marker)
        print(f"[skip] OCR {video_id}: {marker.get('records', 0):,} record đã hoàn tất.", flush=True)
        return marker

    mappings = read_jsonl(artifacts.mapping)
    temporary = _temporary_path(artifacts.ocr)
    temporary.unlink(missing_ok=True)
    records = scene_records = overlay_lines = overlay_only_frames = failures = 0
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        for mapping in track(
            mappings,
            desc=f"PaddleOCR {video_id}",
            total=len(mappings),
            unit="frame",
            force=True,
            leave=True,
        ):
            image_path = artifacts.root / str(mapping.get("image") or "")
            try:
                text, overlay_text, scene_count, overlay_count = read_text(
                    reader,
                    image_path,
                    minimum_confidence,
                )
            except Exception as error:
                failures += 1
                print(f"[skip OCR] {image_path}: {error}", file=sys.stderr, flush=True)
                if failures >= 5 and records == 0:
                    raise RuntimeError(f"OCR {video_id} lỗi liên tiếp từ đầu: {error}") from error
                continue
            records += 1
            scene_records += int(bool(text))
            overlay_lines += overlay_count
            overlay_only_frames += int(bool(overlay_text and not text))
            stream.write(
                json.dumps(
                    {
                        "ocr_schema": OCR_INDEX_SCHEMA_VERSION,
                        "video_id": video_id,
                        "keyframe_number": int(mapping["keyframe_number"]),
                        "frame_id": int(mapping["frame_id"]),
                        "pts_time": float(mapping["pts_time"]),
                        "text": text,
                        "overlay_text": overlay_text,
                        "scene_boxes": scene_count,
                        "overlay_boxes": overlay_count,
                        "text_quality": 1.0,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    temporary.replace(artifacts.ocr)
    marker = {
        **expected,
        "device": device,
        "records": records,
        "scene_text_records": scene_records,
        "suppressed_overlay_lines": overlay_lines,
        "overlay_only_frames": overlay_only_frames,
        "failures": failures,
        "worker_gpu": os.environ.get("AIC_PRE_DIRECT_WORKER_GPU", ""),
        "timing_seconds": {"total": round(time.perf_counter() - ocr_started, 3)},
        "generated_at": utc_now(),
    }
    write_json_atomic(artifacts.ocr_marker, marker)
    print(
        f"OCR {video_id}: {records:,} frame · {scene_records:,} scene-text · "
        f"bỏ {overlay_lines:,} overlay line",
        flush=True,
    )
    return marker


def run_ocr_stage(
    window: VideoWindow,
    output_root: Path,
    arguments: argparse.Namespace,
) -> None:
    available = discover_gpu_ids()
    gpu_ids = resolve_worker_gpu_ids(
        arguments.gpus,
        arguments.workers,
        discovered=available,
    )
    _run_gpu_stage(
        stage="ocr",
        window=window,
        output_root=output_root,
        arguments=arguments,
        gpu_ids=gpu_ids,
        local_initializer=_initialize_ocr_worker,
        pool_initializer=_ocr_pool_initializer,
        task=_ocr_worker_task,
    )


def _initialize_ocr_worker(
    gpu_id: str,
    initialization_lock: Any,
    output_root: str | Path,
    arguments: argparse.Namespace,
) -> None:
    """Load one PaddleOCR reader on a worker's isolated GPU."""
    _bind_cuda_worker(gpu_id, "ocr")

    def load_reader() -> tuple[Any, str]:
        from build_ocr_index import create_reader, resolve_device
        import paddle

        device = resolve_device("gpu:0")
        if paddle.device.cuda.device_count() != 1:
            raise RuntimeError(
                f"OCR worker GPU {gpu_id} không được cô lập thành đúng một CUDA device."
            )
        return create_reader(arguments.ocr_language, device), device

    reader, device = _with_optional_lock(initialization_lock, load_reader)
    _OCR_WORKER_STATE.clear()
    _OCR_WORKER_STATE.update(
        gpu_id=str(gpu_id),
        output_root=Path(output_root),
        arguments=arguments,
        reader=reader,
        device=device,
    )
    print(f"[OCR GPU {gpu_id}] PaddleOCR sẵn sàng trên logical {device}.", flush=True)


def _ocr_pool_initializer(
    gpu_queue: Any,
    initialization_lock: Any,
    output_root: str | Path,
    arguments: argparse.Namespace,
) -> None:
    _initialize_ocr_worker(gpu_queue.get(), initialization_lock, output_root, arguments)


def _ocr_worker_task(ordinal: int, total: int, video_path_value: str) -> dict[str, Any]:
    if not _OCR_WORKER_STATE:
        raise RuntimeError("OCR GPU worker chưa được khởi tạo.")
    video_path = Path(video_path_value)
    arguments = _OCR_WORKER_STATE["arguments"]
    gpu_id = str(_OCR_WORKER_STATE["gpu_id"])
    started = time.perf_counter()
    print(f"[{ordinal}/{total}] [GPU {gpu_id}] OCR {video_path.stem}", flush=True)
    marker = preprocess_ocr_video(
        video_path,
        _OCR_WORKER_STATE["output_root"],
        reader=_OCR_WORKER_STATE["reader"],
        language=arguments.ocr_language,
        device=_OCR_WORKER_STATE["device"],
        minimum_confidence=arguments.ocr_min_confidence,
        force=arguments.force,
    )
    return {
        "stage": "ocr",
        "video_id": video_path.stem,
        "gpu_id": gpu_id,
        "frames": int(marker.get("records") or 0),
        "seconds": time.perf_counter() - started,
    }


def _copy_jsonl_records(source: Path, destination: Any, *, require_text: bool = False) -> int:
    records = 0
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="utf-8") as stream:
        for line in track(stream, desc=f"Merge {source.name}", unit="record", nested=True):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if require_text and not str(payload.get("text") or "").strip():
                continue
            destination.write(json.dumps(payload, ensure_ascii=False) + "\n")
            records += 1
    return records


def finalize_artifacts(
    window: VideoWindow,
    output_root: Path,
) -> dict[str, Any]:
    """Create global indexes from every completed per-video artifact."""
    from ocr_regions import OCR_INDEX_SCHEMA_VERSION

    videos_root = output_root / "videos"
    video_dirs = sorted((path for path in videos_root.glob("*") if path.is_dir()), key=lambda path: path.name)
    output_root.mkdir(parents=True, exist_ok=True)
    global_ocr = output_root / "ocr_index.jsonl.gz"
    global_objects = output_root / "object_index.jsonl.gz"
    ocr_temporary = _temporary_path(global_ocr)
    object_temporary = _temporary_path(global_objects)
    ocr_records = object_records = 0
    manifest_videos: list[dict[str, Any]] = []
    with gzip.open(ocr_temporary, "wt", encoding="utf-8") as ocr_stream, gzip.open(
        object_temporary, "wt", encoding="utf-8"
    ) as object_stream:
        for video_dir in track(
            video_dirs,
            desc="Finalize direct videos",
            total=len(video_dirs),
            unit="video",
            force=True,
            leave=True,
        ):
            artifacts = artifact_paths(output_root, video_dir.name)
            visual = read_json(artifacts.visual_marker)
            ocr = read_json(artifacts.ocr_marker)
            visual_ready = bool(
                visual
                and artifacts.clip.is_file()
                and artifacts.mapping.is_file()
                and artifacts.frame_ids.is_file()
                and artifacts.pts_times.is_file()
            )
            ocr_ready = bool(ocr and artifacts.ocr.is_file())
            if visual_ready and artifacts.objects.is_file():
                object_records += _copy_jsonl_records(artifacts.objects, object_stream)
            if ocr_ready:
                ocr_records += _copy_jsonl_records(artifacts.ocr, ocr_stream, require_text=True)
            complete = {
                "schema": DIRECT_PREPROCESS_SCHEMA,
                "video_id": video_dir.name,
                "visual_ready": visual_ready,
                "ocr_ready": ocr_ready,
                "all_frames": bool(visual.get("all_frames")),
                "sampled_frames": int(visual.get("sampled_frames") or 0),
                "ocr_scene_records": int(ocr.get("scene_text_records") or 0),
                "updated_at": utc_now(),
            }
            if visual_ready and ocr_ready:
                write_json_atomic(artifacts.complete_marker, complete)
            manifest_videos.append(complete)
    ocr_temporary.replace(global_ocr)
    object_temporary.replace(global_objects)
    complete_count = sum(item["visual_ready"] and item["ocr_ready"] for item in manifest_videos)
    visual_count = sum(item["visual_ready"] for item in manifest_videos)
    all_frame_count = sum(item["visual_ready"] and item["all_frames"] for item in manifest_videos)
    manifest = {
        "schema": DIRECT_PREPROCESS_SCHEMA,
        "coordinate_system": (
            "video frame_id zero-based; keyframe_number artifact row zero-based "
            "(equals frame_id for complete all-frame decode)"
        ),
        "video_order": "sorted by video_id; CLI start/end are 1-based inclusive",
        "corpus_videos": window.total,
        "visual_videos": visual_count,
        "all_frame_videos": all_frame_count,
        "complete_videos": complete_count,
        "ocr_records": ocr_records,
        "object_records": object_records,
        "ocr_index": global_ocr.name,
        "object_index": global_objects.name,
        "videos": manifest_videos,
        "updated_at": utc_now(),
    }
    write_json_atomic(output_root / "manifest.json", manifest)
    write_json_atomic(
        Path(f"{global_ocr}.complete"),
        {
            "schema": DIRECT_PREPROCESS_SCHEMA,
            "ocr_schema": OCR_INDEX_SCHEMA_VERSION,
            "records": ocr_records,
            "updated_at": utc_now(),
        },
    )
    shard_dir = output_root / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    selected_status = []
    for video_path in track(
        window.videos,
        desc="Ghi direct shard status",
        total=len(window.videos),
        unit="video",
        nested=True,
    ):
        artifacts = artifact_paths(output_root, video_path.stem)
        selected_status.append(
            {
                "video_id": video_path.stem,
                "visual_ready": artifacts.visual_marker.is_file(),
                "ocr_ready": artifacts.ocr_marker.is_file(),
            }
        )
    shard = {
        "schema": DIRECT_PREPROCESS_SCHEMA,
        "start": window.start,
        "end": window.end,
        "total": window.total,
        "videos": selected_status,
        "updated_at": utc_now(),
    }
    write_json_atomic(shard_dir / f"pre_{window.start:04d}_{window.end:04d}.json", shard)
    print(
        f"Finalize: {complete_count:,}/{window.total:,} video đủ visual+OCR · "
        f"OCR {ocr_records:,} · object {object_records:,} → {output_root}",
        flush=True,
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess a deterministic shard of raw AIC videos")
    parser.add_argument("--stage", choices=("visual", "ocr", "finalize"), required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start-video", type=int, default=1)
    parser.add_argument("--end-video", type=int, default=0)
    parser.add_argument(
        "--gpus",
        default=os.environ.get("AIC_PRE_DIRECT_GPUS", "auto"),
        help="CUDA GPU vật lý, mặc định auto dùng mọi GPU nhìn thấy; ví dụ 0,1",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("AIC_PRE_DIRECT_WORKERS", "0")),
        help="Số process GPU; 0 mặc định tạo một worker cho mỗi GPU trong --gpus",
    )
    parser.add_argument("--sample-fps", type=float, default=0.0, help="0 means every decoded frame (default)")
    parser.add_argument("--max-side", type=int, default=0, help="0 preserves original PNG resolution")
    parser.add_argument("--clip-model", default=os.environ.get("AIC_DIRECT_CLIP_MODEL", "ViT-B/32"))
    parser.add_argument("--clip-batch", type=int, default=int(os.environ.get("AIC_DIRECT_VIDEO_BATCH", "64")))
    parser.add_argument("--object-model", default=os.environ.get("AIC_DIRECT_OBJECT_MODEL", "yolo11m.pt"))
    parser.add_argument("--object-batch", type=int, default=int(os.environ.get("AIC_DIRECT_OBJECT_BATCH", "16")))
    parser.add_argument("--object-device", default=os.environ.get("AIC_DIRECT_OBJECT_DEVICE", "0"))
    parser.add_argument("--object-confidence", type=float, default=float(os.environ.get("AIC_DIRECT_OBJECT_CONF", "0.20")))
    parser.add_argument("--ocr-language", default=os.environ.get("AIC_DIRECT_OCR_LANGUAGE", "vi"))
    parser.add_argument("--ocr-device", default=os.environ.get("AIC_OCR_DEVICE", "gpu:0"))
    parser.add_argument("--ocr-min-confidence", type=float, default=float(os.environ.get("AIC_DIRECT_OCR_CONF", "0.45")))
    parser.add_argument(
        "--mask-clip-overlays",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("AIC_DIRECT_CLIP_MASK_OVERLAYS", "1").lower() not in {"0", "false", "no"},
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if not math.isfinite(arguments.sample_fps) or arguments.sample_fps < 0:
        raise ValueError("--sample-fps phải >= 0; dùng 0 để lấy tất cả frame.")
    if arguments.max_side < 0:
        raise ValueError("--max-side phải >= 0.")
    if not math.isfinite(arguments.object_confidence) or not 0.0 <= arguments.object_confidence <= 1.0:
        raise ValueError("--object-confidence phải trong [0, 1].")
    if not math.isfinite(arguments.ocr_min_confidence) or not 0.0 <= arguments.ocr_min_confidence <= 1.0:
        raise ValueError("--ocr-min-confidence phải trong [0, 1].")
    if arguments.workers < 0:
        raise ValueError("--workers phải >= 0; 0 nghĩa là một worker cho mỗi GPU.")
    if arguments.stage == "ocr" and not str(arguments.ocr_device).startswith("gpu"):
        raise ValueError("Direct multi-worker OCR yêu cầu --ocr-device gpu:0.")

    from direct_video_retrieval import resolve_video_dataset

    dataset_root, video_files, source_kind = resolve_video_dataset()
    window = select_video_window(video_files, arguments.start_video, arguments.end_video)
    output_root = arguments.output.expanduser()
    write_video_order(tuple(sorted(video_files, key=lambda path: (path.stem, str(path)))), output_root)
    print(
        f"Direct preprocess stage={arguments.stage} · source={source_kind}:{dataset_root} · "
        f"video [{window.start}, {window.end}]/{window.total} · output={output_root}",
        flush=True,
    )
    if arguments.stage == "visual":
        run_visual_stage(window, output_root, arguments)
    elif arguments.stage == "ocr":
        run_ocr_stage(window, output_root, arguments)
    else:
        finalize_artifacts(window, output_root)


if __name__ == "__main__":
    main()
