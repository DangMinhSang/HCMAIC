"""Optional retrieval directly from the mounted AIC video dataset.

This module is deliberately separate from :mod:`retrieval`: the normal path
uses the BTC-supplied CLIP arrays and official keyframe mapping, while this
path reads raw MP4 files, samples them, and builds its own local CLIP index.
It is an experimental, opt-in path because the generated frame index is not
an official BTC feature/mapping artifact.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from clip_encoder import ClipTextEncoder
from progress import track
from retrieval import AICRetrievalEngine, SearchResult, TrakeVideoResult, tokenize


VIDEO_EXTENSIONS = frozenset({".mp4", ".mkv", ".avi", ".mov", ".webm"})
DEFAULT_DATASET_ID = "doanminhtuan/video-aic"
DEFAULT_MOUNT_ROOT = Path("/kaggle/input/datasets/doanminhtuan/video-aic")
DIRECT_ARTIFACT_SCHEMA = 3


class DirectVideoUnavailableError(RuntimeError):
    """Raised when the opt-in raw-video path cannot be initialized."""


@dataclass(frozen=True)
class DirectFrame:
    """One indexed frame in the locally generated direct-video artifacts."""

    keyframe_number: int
    frame_id: int
    pts_time: float
    fps: float
    source_path: Path
    image_path: Path | None = None


@dataclass(frozen=True)
class _DirectCandidate:
    video_id: str
    feature_index: int
    visual_score: float


@dataclass(frozen=True)
class _PreprocessedShard:
    """Memory-mapped full-frame arrays for one preprocessed video."""

    video_id: str
    root: Path
    source_path: Path
    embeddings: np.ndarray
    frame_ids: np.ndarray
    pts_times: np.ndarray
    fps: float
    all_frames: bool


def parse_frame_steps(value: str | Sequence[int]) -> tuple[int, ...]:
    """Validate a coarse-to-fine modulo schedule such as ``4,2,1``."""
    try:
        if isinstance(value, str):
            steps = tuple(int(part.strip()) for part in value.split(",") if part.strip())
        else:
            steps = tuple(int(part) for part in value)
    except (TypeError, ValueError) as error:
        raise ValueError("AIC_DIRECT_FRAME_STEPS phải có dạng 4,2,1.") from error
    if not steps or steps[-1] != 1:
        raise ValueError("AIC_DIRECT_FRAME_STEPS phải kết thúc bằng 1 để refinement dùng mọi frame.")
    if any(step < 1 or step > 300 for step in steps):
        raise ValueError("Mỗi direct frame step phải nằm trong 1..300.")
    if any(previous <= current or previous % current for previous, current in zip(steps, steps[1:])):
        raise ValueError("Direct frame steps phải giảm dần và chia hết nhau, ví dụ 4,2,1.")
    return steps


def temporal_modulo_indices(
    frame_ids: np.ndarray,
    centers: Sequence[int],
    previous_step: int,
    next_step: int,
) -> np.ndarray:
    """Return finer modulo rows near coarser candidate rows.

    ``centers`` are local row indices, while modulo and radius use decoded
    ``frame_id`` coordinates. With 4→2→1, every decoded frame surrounding a
    retained ``%4`` frame becomes reachable without scanning the full matrix.
    """
    values = np.asarray(frame_ids)
    if values.ndim != 1 or not len(values):
        return np.empty(0, dtype=np.int32)
    selected: set[int] = set()
    radius = max(1, int(previous_step))
    for center in track(
        centers,
        desc=f"Refine frame %{previous_step}→%{next_step}",
        total=len(centers),
        unit="center",
        nested=True,
    ):
        local_index = int(center)
        if local_index < 0 or local_index >= len(values):
            continue
        center_frame = int(values[local_index])
        left = int(np.searchsorted(values, center_frame - radius, side="left"))
        right = int(np.searchsorted(values, center_frame + radius, side="right"))
        candidates = np.arange(left, right, dtype=np.int32)
        if next_step > 1:
            candidates = candidates[np.asarray(values[candidates]) % next_step == 0]
        selected.update(int(index) for index in candidates)
    return np.asarray(sorted(selected), dtype=np.int32)


def discover_video_files(root: str | Path) -> list[Path]:
    """Find mounted video files recursively in the Kaggle dataset layout."""
    root = Path(root).expanduser()
    if not root.is_dir():
        return []
    paths = sorted(root.rglob("*"))
    return [
        path
        for path in track(
            paths,
            desc=f"Quét video direct: {root.name}",
            total=len(paths),
            unit="path",
            nested=True,
        )
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]


def _candidate_roots(input_root: str | Path | None = None) -> list[Path]:
    configured = os.environ.get("AIC_DIRECT_VIDEO_ROOT", "").strip()
    data_root = Path(input_root or os.environ.get("AIC_DATA_ROOT", "/kaggle/input")).expanduser()
    roots = [
        Path(configured).expanduser() if configured else None,
        DEFAULT_MOUNT_ROOT,
        data_root / "datasets" / "doanminhtuan" / "video-aic",
        data_root / "video-aic",
    ]
    output: list[Path] = []
    seen: set[str] = set()
    for root in track(
        roots,
        desc="Tìm mount direct video",
        total=len(roots),
        unit="root",
        nested=True,
    ):
        if root is None:
            continue
        key = str(root)
        if key not in seen:
            seen.add(key)
            output.append(root)
    return output


def resolve_video_dataset(
    input_root: str | Path | None = None,
    *,
    dataset_id: str | None = None,
) -> tuple[Path, list[Path], str]:
    """Resolve the mounted dataset first, then use the requested KaggleHub fallback.

    Kaggle already exposes ``/kaggle/input/datasets/doanminhtuan/video-aic``.
    Avoiding a second download is important: the direct option must be cheap to
    turn on when the input is attached, while still working in a clean kernel.
    """
    for root in _candidate_roots(input_root):
        files = discover_video_files(root)
        if files:
            print(f"Direct video source: mounted {root} ({len(files):,} video files).", flush=True)
            return root.resolve(), files, "mounted"

    dataset_id = (dataset_id or os.environ.get("AIC_DIRECT_VIDEO_DATASET", DEFAULT_DATASET_ID)).strip()
    if not dataset_id:
        raise DirectVideoUnavailableError("AIC_DIRECT_VIDEO_DATASET không được để trống.")
    try:
        import kagglehub  # type: ignore
    except ImportError as error:
        raise DirectVideoUnavailableError(
            "Không thấy video mount và thiếu kagglehub. Chạy lại run.py với --direct-video "
            "hoặc cài Code/requirements-direct-video.txt."
        ) from error
    try:
        downloaded = Path(kagglehub.dataset_download(dataset_id)).expanduser()
    except Exception as error:
        raise DirectVideoUnavailableError(
            f"Không resolve được Kaggle dataset {dataset_id!r} bằng kagglehub: {error}"
        ) from error
    files = discover_video_files(downloaded)
    if not files:
        raise DirectVideoUnavailableError(
            f"kagglehub trả về {downloaded}, nhưng không tìm thấy file video trong đó."
        )
    print(f"Direct video source: kagglehub {downloaded} ({len(files):,} video files).", flush=True)
    return downloaded.resolve(), files, "kagglehub"


class DirectVideoRetrievalEngine:
    """CLIP retrieval built from raw videos, without BTC feature arrays.

    Frame ids in this mode are decoded OpenCV frame indices (zero-based), and
    ``keyframe_number`` is the artifact-row ordinal. This preserves temporal
    ordering for the dashboard/TRAKE path, but the generated index must be
    benchmarked against the official submission convention before submission.
    """

    source_mode = "direct-video"

    def __init__(
        self,
        dataset_root: str | Path,
        video_files: Sequence[str | Path] | None = None,
        encoder: ClipTextEncoder | None = None,
        source_kind: str = "mounted",
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.source_kind = source_kind
        files = [Path(path).expanduser().resolve() for path in (video_files or discover_video_files(self.dataset_root))]
        self._features: dict[str, Path] = {}
        for path in track(
            files,
            desc="Lập danh sách direct video",
            total=len(files),
            unit="video",
            leave=True,
        ):
            video_id = path.stem
            if video_id in self._features:
                raise DirectVideoUnavailableError(
                    f"Trùng video_id {video_id!r} trong direct dataset; không thể map frame an toàn."
                )
            self._features[video_id] = path
        if not self._features:
            raise DirectVideoUnavailableError(f"Không tìm thấy video trong {self.dataset_root}.")

        self.encoder = encoder or ClipTextEncoder(
            model_name=os.environ.get("AIC_DIRECT_CLIP_MODEL", "ViT-B/32")
        )
        self.sample_stride = self._bounded_int("AIC_DIRECT_VIDEO_STRIDE", 15, 1, 300)
        self.batch_size = self._bounded_int("AIC_DIRECT_VIDEO_BATCH", 64, 1, 256)
        self.max_samples_per_video = self._bounded_int("AIC_DIRECT_VIDEO_MAX_SAMPLES", 0, 0, 1000000)
        self.frame_steps = parse_frame_steps(os.environ.get("AIC_DIRECT_FRAME_STEPS", "4,2,1"))
        self.refine_pool = self._bounded_int("AIC_DIRECT_REFINE_POOL", 1200, 100, 10000)
        self.trake_refine_videos = self._bounded_int("AIC_DIRECT_TRAKE_REFINE_VIDEOS", 50, 10, 200)
        runtime_dir = Path(os.environ.get("AIC_RUNTIME_DIR", "/kaggle/working")) / "aic_direct_video_cache"
        self.runtime_dir = runtime_dir
        self.preprocessed_root = Path(
            os.environ.get("AIC_DIRECT_PREPROCESSED_ROOT", "/kaggle/working/aic_direct_preprocessed")
        ).expanduser()
        self._cache_path = runtime_dir / f"index_{self._cache_key()}.npz"
        self._frame_cache_dir = runtime_dir / f"frames_{self._cache_key()}"
        self.frame_cache_max = self._bounded_int("AIC_DIRECT_FRAME_CACHE_MAX", 512, 1, 5000)
        self._frame_cache_lock = threading.Lock()
        self._frame_cache_lru: OrderedDict[Path, None] | None = None
        self._cv2 = None
        self._embeddings: np.ndarray | None = None
        self._records: list[DirectFrame] = []
        self._records_by_video: dict[str, list[DirectFrame]] = {}
        self._mapping_lookup: dict[str, dict[int, DirectFrame]] = {}
        self._mapping_positions: dict[str, dict[int, int]] = {}
        self._offsets: np.ndarray | None = None
        self._video_order = tuple(self._features)
        self._preprocessed_checked = False
        self._preprocessed_video_count = 0
        self._preprocessed_frame_count = 0
        self._preprocessed_all_frame_videos = 0
        self._preprocessed_shards: dict[str, _PreprocessedShard] = {}
        self._coarse_embeddings: np.ndarray | None = None
        self._coarse_video_positions: np.ndarray | None = None
        self._coarse_local_indices: np.ndarray | None = None
        self._coarse_offsets: np.ndarray | None = None
        self._direct_object_cache: dict[str, dict[int, tuple[str, ...]]] = {}
        self._direct_object_matrices: dict[
            str,
            tuple[np.ndarray, tuple[str, ...]] | None,
        ] = {}

    @classmethod
    def from_environment(cls, input_root: str | Path | None = None) -> "DirectVideoRetrievalEngine":
        root, files, source = resolve_video_dataset(input_root)
        return cls(root, files, source_kind=source)

    @staticmethod
    def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
        try:
            return max(minimum, min(int(os.environ.get(name, str(default))), maximum))
        except ValueError:
            return default

    def _cache_key(self) -> str:
        digest = hashlib.sha256()
        digest.update(str(self.dataset_root).encode("utf-8"))
        digest.update(self.encoder.model_name.encode("utf-8"))
        digest.update(f"stride={self.sample_stride};max={self.max_samples_per_video}".encode("ascii"))
        for video_id, path in track(
            self._features.items(),
            desc="Fingerprint direct videos",
            total=len(self._features),
            unit="video",
            nested=True,
        ):
            try:
                stat = path.stat()
                digest.update(video_id.encode("utf-8"))
                digest.update(str(stat.st_size).encode("ascii"))
                digest.update(str(stat.st_mtime_ns).encode("ascii"))
            except OSError:
                digest.update(str(path).encode("utf-8"))
        return digest.hexdigest()[:20]

    @property
    def video_count(self) -> int:
        return len(self._features)

    @property
    def vector_count(self) -> int:
        if self._preprocessed_shards:
            return self._preprocessed_frame_count
        return int(len(self._embeddings)) if self._embeddings is not None else len(self._records)

    @property
    def feature_cache_loaded(self) -> bool:
        return self._embeddings is not None or self._coarse_embeddings is not None

    @property
    def source_description(self) -> str:
        preprocessed = (
            f" · preprocessed={self._preprocessed_video_count}/{len(self._video_order)} video"
            f" · all-frame={self._preprocessed_all_frame_videos}/{self._preprocessed_video_count}"
            if self._preprocessed_video_count
            else ""
        )
        sampling = (
            f"query modulo={'→'.join(map(str, self.frame_steps))}"
            if self._preprocessed_video_count
            else f"fallback stride={self.sample_stride} frame"
        )
        return (
            f"{self.source_kind}:{self.dataset_root} · {sampling} · "
            f"local CLIP index, không dùng feature/mapping BTC{preprocessed}"
        )

    def _load_cv2(self):
        if self._cv2 is not None:
            return self._cv2
        try:
            import cv2  # type: ignore
        except ImportError as error:
            raise DirectVideoUnavailableError(
                "Direct video cần OpenCV (cv2). Kaggle thường đã có sẵn; nếu kernel thiếu, "
                "cài opencv-python-headless rồi restart kernel."
            ) from error
        self._cv2 = cv2
        return cv2

    def prepare_runtime(self) -> None:
        """Load the generated index or build it once before serving queries."""
        if self._embeddings is not None or self._coarse_embeddings is not None:
            return
        if self._load_preprocessed_index():
            return
        if self._cache_path.is_file():
            try:
                self._load_cache()
                print(
                    f"Đã nạp direct video index: {len(self._records):,} sampled frames → {self._cache_path}",
                    flush=True,
                )
                return
            except (OSError, ValueError, KeyError) as error:
                print(f"Direct video cache không hợp lệ ({error}); đang dựng lại…", flush=True)
        self._build_index()

    def _load_preprocessed_index(self) -> bool:
        """Load full-frame shards and materialize only the coarsest RAM index."""
        if self._preprocessed_checked:
            return False
        self._preprocessed_checked = True
        videos_root = self.preprocessed_root / "videos"
        if not videos_root.is_dir():
            return False
        coarse_entries: list[tuple[int, str, np.ndarray]] = []
        coarse_counts = np.zeros(len(self._video_order), dtype=np.int64)
        embedding_dimension = 0
        loaded_videos = 0
        artifact_candidates = 0
        all_frame_videos = 0
        frame_count = 0
        coarse_step = self.frame_steps[0]
        for video_position, video_id in track(
            enumerate(self._video_order),
            desc="Nạp pre-direct CLIP shards",
            total=len(self._video_order),
            unit="video",
            force=True,
            leave=True,
        ):
            video_dir = videos_root / video_id
            clip_path = video_dir / "clip.npy"
            mapping_path = video_dir / "mapping.jsonl"
            frame_ids_path = video_dir / "frame_ids.npy"
            pts_times_path = video_dir / "pts_times.npy"
            marker_path = video_dir / "visual.complete.json"
            if not (clip_path.is_file() and mapping_path.is_file() and marker_path.is_file()):
                continue
            artifact_candidates += 1
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                artifact_schema = int(marker.get("schema") or 0)
                if artifact_schema != DIRECT_ARTIFACT_SCHEMA:
                    raise ValueError(
                        f"artifact schema={artifact_schema}, cần schema={DIRECT_ARTIFACT_SCHEMA}; "
                        "chạy lại --pre-direct-video 1 để migrate không cần rerun model"
                    )
                if marker.get("stores_frame_images") is not False:
                    raise ValueError("artifact chưa khai báo chế độ không lưu frame image")
                artifact_model = str(marker.get("clip_model") or "")
                if artifact_model != self.encoder.model_name:
                    raise ValueError(
                        f"CLIP model artifact={artifact_model!r}, runtime={self.encoder.model_name!r}"
                    )
                vectors = np.load(clip_path, mmap_mode="r")
                if frame_ids_path.is_file() and pts_times_path.is_file():
                    frame_ids = np.load(frame_ids_path, mmap_mode="r")
                    pts_times = np.load(pts_times_path, mmap_mode="r")
                else:
                    legacy_frame_ids: list[int] = []
                    legacy_pts_times: list[float] = []
                    with mapping_path.open(encoding="utf-8") as stream:
                        for line in track(
                            stream,
                            desc=f"Mapping pre-direct {video_id}",
                            unit="frame",
                            nested=True,
                        ):
                            payload = json.loads(line)
                            legacy_frame_ids.append(int(payload["frame_id"]))
                            legacy_pts_times.append(float(payload["pts_time"]))
                    frame_ids = np.asarray(legacy_frame_ids, dtype=np.int64)
                    pts_times = np.asarray(legacy_pts_times, dtype=np.float32)
                if vectors.ndim != 2 or vectors.shape[1] < 1:
                    raise ValueError(f"clip shape không hợp lệ: {vectors.shape}")
                if embedding_dimension and vectors.shape[1] != embedding_dimension:
                    raise ValueError(
                        f"CLIP dimension {vectors.shape[1]} khác shard trước {embedding_dimension}"
                    )
                if frame_ids.ndim != 1 or pts_times.ndim != 1:
                    raise ValueError("frame_ids/pts_times phải là vector 1-D")
                if len(vectors) != len(frame_ids) or len(vectors) != len(pts_times):
                    raise ValueError(
                        f"clip={len(vectors)}, frame_ids={len(frame_ids)}, pts={len(pts_times)}"
                    )
                frame_differences = np.diff(frame_ids)
                if len(frame_ids) and np.any(frame_differences <= 0):
                    raise ValueError("frame_ids phải tăng nghiêm ngặt")
                fps = float(marker.get("source_fps") or 30.0)
                marker_all_frames = bool(
                    marker.get(
                        "all_frames",
                        float(marker.get("sample_fps") or -1) == 0.0,
                    )
                )
                continuous_frames = bool(
                    len(frame_ids)
                    and int(frame_ids[0]) == 0
                    and np.all(frame_differences == 1)
                )
                all_frames = marker_all_frames and continuous_frames
                shard = _PreprocessedShard(
                    video_id=video_id,
                    root=video_dir,
                    source_path=self._features[video_id],
                    embeddings=vectors,
                    frame_ids=frame_ids,
                    pts_times=pts_times,
                    fps=fps if fps > 0 else 30.0,
                    all_frames=all_frames,
                )
                local_indices = np.flatnonzero(np.asarray(frame_ids) % coarse_step == 0).astype(
                    np.int32,
                    copy=False,
                )
                if not len(local_indices):
                    raise ValueError(f"không có frame thỏa frame_id % {coarse_step} == 0")
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                print(f"[warning] Bỏ pre-direct artifact lỗi {video_id}: {error}", flush=True)
                continue
            self._preprocessed_shards[video_id] = shard
            embedding_dimension = int(vectors.shape[1])
            coarse_entries.append((video_position, video_id, local_indices))
            coarse_counts[video_position] = len(local_indices)
            loaded_videos += 1
            all_frame_videos += int(all_frames)
            frame_count += len(vectors)
        if not coarse_entries:
            if artifact_candidates:
                raise DirectVideoUnavailableError(
                    f"Có {artifact_candidates} pre-direct shard nhưng không shard nào hợp lệ; "
                    "không fallback sang index khác để tránh query sai corpus. Xem warning phía trên."
                )
            return False
        total_coarse = int(coarse_counts.sum())
        estimated_gib = total_coarse * embedding_dimension * 4 / (1024**3)
        print(
            f"Direct coarse RAM dự kiến: {total_coarse:,} × {embedding_dimension} float32 "
            f"≈ {estimated_gib:.2f} GiB.",
            flush=True,
        )
        try:
            self._coarse_embeddings = np.empty(
                (total_coarse, embedding_dimension),
                dtype=np.float32,
            )
            self._coarse_video_positions = np.empty(total_coarse, dtype=np.int32)
            self._coarse_local_indices = np.empty(total_coarse, dtype=np.int32)
        except MemoryError as error:
            raise DirectVideoUnavailableError(
                f"Không đủ RAM cho coarse frame %{coarse_step} (~{estimated_gib:.2f} GiB). "
                "Khởi động lại với --direct-frame-steps 8,4,2,1 hoặc step đầu lớn hơn."
            ) from error
        cursor = 0
        for video_position, video_id, local_indices in track(
            coarse_entries,
            desc=f"Materialize direct frame %{coarse_step}",
            total=len(coarse_entries),
            unit="video",
            force=True,
            leave=True,
        ):
            end = cursor + len(local_indices)
            shard = self._preprocessed_shards[video_id]
            self._coarse_embeddings[cursor:end] = self._normalize_rows(
                np.asarray(shard.embeddings[local_indices], dtype=np.float32)
            )
            self._coarse_video_positions[cursor:end] = video_position
            self._coarse_local_indices[cursor:end] = local_indices
            cursor = end
        self._coarse_offsets = np.zeros(len(self._video_order) + 1, dtype=np.int64)
        self._coarse_offsets[1:] = np.cumsum(coarse_counts)
        self._preprocessed_video_count = loaded_videos
        self._preprocessed_all_frame_videos = all_frame_videos
        self._preprocessed_frame_count = frame_count
        print(
            f"Đã nạp pre-direct hierarchy: {frame_count:,} indexed frame · "
            f"{len(self._coarse_embeddings):,} frame % {coarse_step} trong RAM · "
            f"{loaded_videos:,}/{len(self._video_order):,} video → {self.preprocessed_root}",
            flush=True,
        )
        if loaded_videos < len(self._video_order):
            print(
                "[warning] Pre-direct index mới là một phần corpus; query chỉ phủ các shard đã hoàn tất.",
                flush=True,
            )
        if all_frame_videos < loaded_videos:
            print(
                f"[warning] Chỉ {all_frame_videos:,}/{loaded_videos:,} shard được đánh dấu all-frame; "
                "refinement %2/%1 không thể phục hồi frame chưa từng precompute.",
                flush=True,
            )
        return True

    def _load_cache(self) -> None:
        with np.load(self._cache_path, allow_pickle=False) as payload:
            embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
            video_ids = np.asarray(payload["video_ids"]).astype(str)
            frame_ids = np.asarray(payload["frame_ids"], dtype=np.int64)
            sample_numbers = np.asarray(payload["sample_numbers"], dtype=np.int64)
            pts_times = np.asarray(payload["pts_times"], dtype=np.float32)
            fps_values = np.asarray(payload["fps_values"], dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] != 512:
            raise ValueError(f"direct embedding shape không hợp lệ: {embeddings.shape}")
        lengths = {len(embeddings), len(video_ids), len(frame_ids), len(sample_numbers), len(pts_times), len(fps_values)}
        if len(lengths) != 1:
            raise ValueError("direct index có các mảng metadata khác độ dài")
        records = [
            DirectFrame(
                int(sample_number),
                int(frame_id),
                float(pts_time),
                float(fps) if float(fps) > 0 else 30.0,
                self._features[str(video_id)],
            )
            for video_id, frame_id, sample_number, pts_time, fps in track(
                zip(video_ids, frame_ids, sample_numbers, pts_times, fps_values),
                desc="Khôi phục direct frame mapping",
                total=len(video_ids),
                unit="frame",
                force=True,
            )
        ]
        self._embeddings = self._normalize_rows(embeddings)
        self._set_records(records)

    def _build_index(self) -> None:
        cv2 = self._load_cv2()
        try:
            from PIL import Image
        except ImportError as error:
            raise DirectVideoUnavailableError("Direct video cần Pillow để tiền xử lý ảnh CLIP.") from error

        self.encoder.warmup()
        vector_parts: list[np.ndarray] = []
        records: list[DirectFrame] = []
        for video_id, video_path in track(
            self._features.items(),
            desc="Đọc video và tạo direct CLIP index",
            total=len(self._features),
            unit="video",
            force=True,
            leave=True,
        ):
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                print(f"[warning] Bỏ qua video không mở được: {video_path}", flush=True)
                capture.release()
                continue
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            fps = fps if fps > 0.0 else 30.0
            frame_iter: Iterable[int] = range(frame_count) if frame_count > 0 else count()
            image_batch = []
            frame_batch: list[DirectFrame] = []
            sample_number = 0

            def flush_batch() -> None:
                if not image_batch:
                    return
                vectors = self.encoder.encode_images(
                    image_batch,
                    batch_size=self.batch_size,
                    progress_desc=f"CLIP ảnh {video_id}",
                    nested=True,
                )
                vector_parts.append(vectors)
                records.extend(frame_batch)
                for image in track(
                    image_batch,
                    desc="Đóng ảnh direct batch",
                    total=len(image_batch),
                    unit="image",
                    nested=True,
                ):
                    close = getattr(image, "close", None)
                    if close:
                        close()
                image_batch.clear()
                frame_batch.clear()

            for frame_index in track(
                frame_iter,
                desc=f"Decode {video_id}",
                total=frame_count or None,
                unit="frame",
                nested=True,
            ):
                ok, bgr = capture.read()
                if not ok:
                    break
                if frame_index % self.sample_stride:
                    continue
                if self.max_samples_per_video and sample_number >= self.max_samples_per_video:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                image_batch.append(Image.fromarray(rgb))
                frame_batch.append(
                    DirectFrame(
                        keyframe_number=sample_number,
                        frame_id=int(frame_index),
                        pts_time=float(frame_index) / fps,
                        fps=fps,
                        source_path=video_path,
                    )
                )
                sample_number += 1
                if len(image_batch) >= self.batch_size:
                    flush_batch()
            flush_batch()
            capture.release()

        if not vector_parts or not records:
            raise DirectVideoUnavailableError(
                "Không đọc được frame nào từ video direct; kiểm tra codec hoặc mount dataset."
            )
        embeddings = self._normalize_rows(np.concatenate(vector_parts, axis=0).astype(np.float32, copy=False))
        if len(embeddings) != len(records):
            raise DirectVideoUnavailableError("Số direct embedding không khớp số frame đã đọc.")
        self._embeddings = embeddings
        self._set_records(records)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        temporary = self._cache_path.with_suffix(self._cache_path.suffix + ".tmp.npz")
        np.savez(
            temporary,
            embeddings=embeddings,
            video_ids=np.asarray([record.source_path.stem for record in records]),
            frame_ids=np.asarray([record.frame_id for record in records], dtype=np.int64),
            sample_numbers=np.asarray([record.keyframe_number for record in records], dtype=np.int64),
            pts_times=np.asarray([record.pts_time for record in records], dtype=np.float32),
            fps_values=np.asarray([record.fps for record in records], dtype=np.float32),
        )
        temporary.replace(self._cache_path)
        print(
            f"Đã tạo direct video index: {len(records):,} sampled frames → {self._cache_path}",
            flush=True,
        )

    @staticmethod
    def _normalize_rows(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        return values / np.maximum(norms, 1e-12)

    def _set_records(self, records: Sequence[DirectFrame]) -> None:
        self._records = list(records)
        self._records_by_video = {video_id: [] for video_id in self._video_order}
        for record in track(
            self._records,
            desc="Nhóm direct frames theo video",
            total=len(self._records),
            unit="frame",
            nested=True,
        ):
            self._records_by_video.setdefault(record.source_path.stem, []).append(record)
        self._mapping_lookup = {
            video_id: {record.keyframe_number: record for record in records}
            for video_id, records in self._records_by_video.items()
        }
        self._mapping_positions = {
            video_id: {record.keyframe_number: index for index, record in enumerate(records)}
            for video_id, records in self._records_by_video.items()
        }
        offsets = np.zeros(len(self._video_order) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(self._records_by_video.get(video_id, ())) for video_id in self._video_order])
        self._offsets = offsets

    @staticmethod
    def _top_indices(scores: np.ndarray, limit: int) -> np.ndarray:
        if len(scores) == 0:
            return np.array([], dtype=np.int64)
        limit = min(max(1, limit), len(scores))
        if limit == len(scores):
            indices = np.arange(len(scores))
        else:
            indices = np.argpartition(scores, -limit)[-limit:]
        return indices[np.argsort(scores[indices])[::-1]]

    def _raw_candidates(
        self,
        query_vector: np.ndarray,
        top_k: int,
        allowed_video_ids: set[str] | None = None,
    ) -> list[_DirectCandidate]:
        self.prepare_runtime()
        if self._preprocessed_shards:
            return self._hierarchical_candidates(query_vector, top_k, allowed_video_ids)
        assert self._embeddings is not None
        assert self._offsets is not None
        scores = np.asarray(self._embeddings @ query_vector, dtype=np.float32)
        if allowed_video_ids is not None:
            unknown = allowed_video_ids.difference(self._features)
            if unknown:
                raise ValueError(f"Không có video direct: {sorted(unknown)[0]}")
            allowed_positions = {
                position
                for position, video_id in enumerate(self._video_order)
                if video_id in allowed_video_ids
            }
            mask = np.zeros(len(scores), dtype=bool)
            for position in track(
                sorted(allowed_positions),
                desc="Lọc direct video",
                total=len(allowed_positions),
                unit="video",
                nested=True,
            ):
                start, end = self._offsets[position : position + 2]
                mask[int(start) : int(end)] = True
            scores[~mask] = -np.inf
        candidate_pool = min(len(scores), max(800, top_k * 30))
        indices = self._top_indices(scores, candidate_pool)
        candidates: list[_DirectCandidate] = []
        for global_index in track(
            indices,
            desc="Ánh xạ direct candidates",
            total=len(indices),
            unit="frame",
            nested=True,
        ):
            if not np.isfinite(scores[global_index]):
                continue
            position = int(np.searchsorted(self._offsets, int(global_index), side="right") - 1)
            video_id = self._video_order[position]
            candidates.append(
                _DirectCandidate(
                    video_id,
                    int(global_index - self._offsets[position]),
                    float(scores[global_index]),
                )
            )
        return candidates

    @staticmethod
    def _rank_candidates(
        candidates: Sequence[_DirectCandidate],
        limit: int,
        video_positions: dict[str, int],
    ) -> list[_DirectCandidate]:
        return sorted(
            candidates,
            key=lambda item: (
                -item.visual_score,
                video_positions.get(item.video_id, 1 << 30),
                item.feature_index,
            ),
        )[:limit]

    def _score_preprocessed_rows(
        self,
        video_id: str,
        local_indices: np.ndarray,
        query_vector: np.ndarray,
    ) -> list[_DirectCandidate]:
        shard = self._preprocessed_shards[video_id]
        if not len(local_indices):
            return []
        vectors = self._normalize_rows(
            np.asarray(shard.embeddings[local_indices], dtype=np.float32)
        )
        scores = np.asarray(vectors @ query_vector, dtype=np.float32)
        candidates: list[_DirectCandidate] = []
        for local_index, score in track(
            zip(local_indices, scores),
            desc=f"Ánh xạ direct refine {video_id}",
            total=len(local_indices),
            unit="frame",
            nested=True,
        ):
            candidates.append(
                _DirectCandidate(video_id, int(local_index), float(score))
            )
        return candidates

    def _hierarchical_candidates(
        self,
        query_vector: np.ndarray,
        top_k: int,
        allowed_video_ids: set[str] | None,
    ) -> list[_DirectCandidate]:
        """Global %N scan followed by local modulo refinement down to %1."""
        assert self._coarse_embeddings is not None
        assert self._coarse_video_positions is not None
        assert self._coarse_local_indices is not None
        assert self._coarse_offsets is not None
        video_positions = {video_id: index for index, video_id in enumerate(self._video_order)}
        if allowed_video_ids is not None:
            unknown = allowed_video_ids.difference(self._features)
            if unknown:
                raise ValueError(f"Không có video direct: {sorted(unknown)[0]}")
            coarse_rows: list[np.ndarray] = []
            for video_id in track(
                sorted(allowed_video_ids),
                desc="Lọc coarse direct video",
                total=len(allowed_video_ids),
                unit="video",
                nested=True,
            ):
                position = video_positions[video_id]
                start, end = self._coarse_offsets[position : position + 2]
                if end > start:
                    coarse_rows.append(np.arange(int(start), int(end), dtype=np.int64))
            selected_rows = (
                np.concatenate(coarse_rows)
                if coarse_rows
                else np.empty(0, dtype=np.int64)
            )
        else:
            selected_rows = np.empty(0, dtype=np.int64)
        coarse_matrix = (
            self._coarse_embeddings[selected_rows]
            if allowed_video_ids is not None
            else self._coarse_embeddings
        )
        if not len(coarse_matrix):
            return []

        coarse_scores = np.asarray(
            coarse_matrix @ query_vector,
            dtype=np.float32,
        )
        candidate_pool = min(
            len(coarse_scores),
            max(self.refine_pool, top_k * 30),
        )
        top_positions = self._top_indices(coarse_scores, candidate_pool)
        current: list[_DirectCandidate] = []
        for position in track(
            top_positions,
            desc=f"Direct coarse frame %{self.frame_steps[0]}",
            total=len(top_positions),
            unit="frame",
            force=True,
        ):
            coarse_row = (
                int(selected_rows[int(position)])
                if allowed_video_ids is not None
                else int(position)
            )
            video_position = int(self._coarse_video_positions[coarse_row])
            current.append(
                _DirectCandidate(
                    self._video_order[video_position],
                    int(self._coarse_local_indices[coarse_row]),
                    float(coarse_scores[int(position)]),
                )
            )

        for previous_step, next_step in zip(self.frame_steps, self.frame_steps[1:]):
            centers_by_video: dict[str, list[int]] = {}
            for candidate in track(
                current,
                desc=f"Nhóm direct %{previous_step}→%{next_step}",
                total=len(current),
                unit="frame",
                nested=True,
            ):
                centers_by_video.setdefault(candidate.video_id, []).append(candidate.feature_index)
            refined: list[_DirectCandidate] = []
            for video_id, centers in track(
                sorted(centers_by_video.items()),
                desc=f"Direct refine frame %{next_step}",
                total=len(centers_by_video),
                unit="video",
                force=True,
            ):
                shard = self._preprocessed_shards[video_id]
                local_indices = temporal_modulo_indices(
                    shard.frame_ids,
                    centers,
                    previous_step,
                    next_step,
                )
                refined.extend(
                    self._score_preprocessed_rows(video_id, local_indices, query_vector)
                )
            current = self._rank_candidates(refined, candidate_pool, video_positions)
            if not current:
                break
        return current

    def _mapping(self, video_id: str) -> list[DirectFrame]:
        self.prepare_runtime()
        shard = self._preprocessed_shards.get(video_id)
        if shard is not None and video_id not in self._records_by_video:
            records: list[DirectFrame] = []
            for local_index in track(
                range(len(shard.frame_ids)),
                desc=f"Materialize direct mapping {video_id}",
                total=len(shard.frame_ids),
                unit="frame",
                force=True,
            ):
                frame = self._frame_from_local_index(video_id, local_index)
                if frame is not None:
                    records.append(frame)
            self._records_by_video[video_id] = records
        return self._records_by_video.get(video_id, [])

    def _frame_from_local_index(self, video_id: str, local_index: int) -> DirectFrame | None:
        shard = self._preprocessed_shards.get(video_id)
        if shard is None:
            records = self._records_by_video.get(video_id, [])
            return records[local_index] if 0 <= local_index < len(records) else None
        if local_index < 0 or local_index >= len(shard.frame_ids):
            return None
        frame_id = int(shard.frame_ids[local_index])
        return DirectFrame(
            keyframe_number=local_index,
            frame_id=frame_id,
            pts_time=float(shard.pts_times[local_index]),
            fps=shard.fps,
            source_path=shard.source_path,
            image_path=None,
        )

    def _touch_frame_cache(self, preserve: Path) -> None:
        """Maintain an in-memory LRU and a bounded on-demand JPEG directory."""
        if self._frame_cache_lru is None:
            def modified_time(path: Path) -> int:
                try:
                    return path.stat().st_mtime_ns
                except OSError:
                    return 0

            cached = sorted(self._frame_cache_dir.glob("*/*.jpg"), key=modified_time)
            self._frame_cache_lru = OrderedDict((path, None) for path in cached)
        self._frame_cache_lru.pop(preserve, None)
        self._frame_cache_lru[preserve] = None
        overflow = max(0, len(self._frame_cache_lru) - self.frame_cache_max)
        for _index in track(
            range(overflow),
            desc="Giới hạn direct frame cache",
            total=overflow,
            unit="frame",
            nested=True,
        ):
            oldest, _value = self._frame_cache_lru.popitem(last=False)
            oldest.unlink(missing_ok=True)

    def _materialize_frame(self, frame: DirectFrame) -> Path | None:
        if frame.image_path is not None and frame.image_path.is_file():
            return frame.image_path
        target = self._frame_cache_dir / frame.source_path.stem / f"{frame.frame_id:08d}.jpg"
        with self._frame_cache_lock:
            if target.is_file():
                target.touch()
                self._touch_frame_cache(target)
                return target
            cv2 = self._load_cv2()
            capture = cv2.VideoCapture(str(frame.source_path))
            try:
                if not capture.isOpened():
                    return None
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame.frame_id)
                ok, bgr = capture.read()
                if not ok:
                    return None
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(".tmp.jpg")
                if not cv2.imwrite(str(temporary), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
                    return None
                temporary.replace(target)
                self._touch_frame_cache(target)
                return target
            finally:
                capture.release()

    def _frame_for_result(self, result: SearchResult) -> DirectFrame | None:
        if result.video_id in self._preprocessed_shards:
            return self._frame_from_local_index(result.video_id, result.keyframe_number)
        self._mapping(result.video_id)
        return self._mapping_lookup.get(result.video_id, {}).get(result.keyframe_number)

    def ensure_result_images(self, results: Sequence[SearchResult]) -> None:
        """Seek many requested frames with at most one open decoder per video."""
        grouped: dict[Path, dict[int, tuple[DirectFrame, list[SearchResult]]]] = {}
        for result in track(
            results,
            desc="Nhóm direct frame cần giải mã",
            total=len(results),
            unit="frame",
            nested=True,
        ):
            if result.image_path and Path(result.image_path).is_file():
                continue
            frame = self._frame_for_result(result)
            if frame is None:
                continue
            by_frame = grouped.setdefault(frame.source_path, {})
            existing = by_frame.get(frame.frame_id)
            if existing is None:
                by_frame[frame.frame_id] = (frame, [result])
            else:
                existing[1].append(result)

        with self._frame_cache_lock:
            for source_path, frames in track(
                grouped.items(),
                desc="Giải mã direct video theo batch",
                total=len(grouped),
                unit="video",
                nested=True,
            ):
                capture = None
                next_frame_id: int | None = None
                try:
                    for frame_id, (frame, frame_results) in track(
                        sorted(frames.items()),
                        desc=f"Seek {source_path.stem}",
                        total=len(frames),
                        unit="frame",
                        nested=True,
                    ):
                        target = (
                            self._frame_cache_dir
                            / frame.source_path.stem
                            / f"{frame.frame_id:08d}.jpg"
                        )
                        if target.is_file():
                            target.touch()
                        else:
                            cv2 = self._load_cv2()
                            if capture is None:
                                capture = cv2.VideoCapture(str(source_path))
                                if not capture.isOpened():
                                    capture.release()
                                    capture = None
                                    break
                            if next_frame_id != frame_id:
                                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                            ok, bgr = capture.read()
                            next_frame_id = frame_id + 1 if ok else None
                            if not ok:
                                continue
                            target.parent.mkdir(parents=True, exist_ok=True)
                            temporary = target.with_suffix(".tmp.jpg")
                            if not cv2.imwrite(
                                str(temporary),
                                bgr,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 95],
                            ):
                                temporary.unlink(missing_ok=True)
                                continue
                            temporary.replace(target)
                        self._touch_frame_cache(target)
                        for result in track(
                            frame_results,
                            desc="Gắn direct image path",
                            total=len(frame_results),
                            unit="result",
                            nested=True,
                        ):
                            result.image_path = str(target)
                finally:
                    if capture is not None:
                        capture.release()

    def ensure_result_image(self, result: SearchResult) -> str | None:
        """Materialize one returned frame only when a UI/model needs it."""
        if result.image_path and Path(result.image_path).is_file():
            return result.image_path
        self.ensure_result_images([result])
        return result.image_path

    def _candidate_to_result(
        self,
        candidate: _DirectCandidate,
        *,
        materialize_image: bool = False,
    ) -> SearchResult | None:
        self.prepare_runtime()
        frame = self._frame_from_local_index(candidate.video_id, candidate.feature_index)
        if frame is None:
            return None
        result = SearchResult(
            rank=0,
            video_id=candidate.video_id,
            frame_id=frame.frame_id,
            keyframe_number=frame.keyframe_number,
            pts_time=frame.pts_time,
            visual_score=candidate.visual_score,
            metadata_score=0.0,
            score=candidate.visual_score,
            retrieval_score=candidate.visual_score,
            title=f"Raw video {candidate.video_id}",
            image_path=(str(frame.image_path) if frame.image_path is not None and frame.image_path.is_file() else None),
            video_path=str(frame.source_path),
        )
        if materialize_image:
            self.ensure_result_image(result)
        return result

    def result_for_keyframe(
        self,
        video_id: str,
        keyframe_number: int,
        *,
        score: float = 0.0,
        ocr_score: float = 0.0,
        ocr_quality: float = 1.0,
        ocr_text: str = "",
    ) -> SearchResult | None:
        self.prepare_runtime()
        if video_id in self._preprocessed_shards:
            frame = self._frame_from_local_index(video_id, keyframe_number)
        else:
            self._mapping(video_id)
            frame = self._mapping_lookup.get(video_id, {}).get(keyframe_number)
        if frame is None:
            return None
        result = SearchResult(
            rank=0,
            video_id=video_id,
            frame_id=frame.frame_id,
            keyframe_number=keyframe_number,
            pts_time=frame.pts_time,
            visual_score=0.0,
            metadata_score=0.0,
            score=score,
            retrieval_score=score,
            title=f"Raw video {video_id}",
            image_path=(str(frame.image_path) if frame.image_path is not None and frame.image_path.is_file() else None),
            video_path=str(frame.source_path),
            ocr_score=ocr_score,
            ocr_quality=ocr_quality,
            ocr_text=ocr_text,
        )
        return result

    def neighboring_keyframes(
        self,
        video_id: str,
        keyframe_number: int,
        *,
        radius: int = 1,
        query_vector: np.ndarray | None = None,
    ) -> list[tuple[int, SearchResult]]:
        self.prepare_runtime()
        shard = self._preprocessed_shards.get(video_id)
        if shard is not None:
            record_count = len(shard.frame_ids)
            center = keyframe_number if 0 <= keyframe_number < record_count else None
        else:
            records = self._mapping(video_id)
            record_count = len(records)
            center = self._mapping_positions.get(video_id, {}).get(keyframe_number)
        if center is None:
            return []
        radius = max(0, min(int(radius), 5))
        start = max(0, center - radius)
        end = min(record_count, center + radius + 1)
        output: list[tuple[int, SearchResult]] = []
        for local_index in track(
            range(start, end),
            desc="Direct frame lân cận",
            total=max(0, end - start),
            unit="frame",
            nested=True,
        ):
            score = 0.0
            if query_vector is not None:
                if shard is not None:
                    vector = self._normalize_rows(
                        np.asarray(shard.embeddings[local_index : local_index + 1], dtype=np.float32)
                    )[0]
                    score = float(vector @ query_vector)
                elif self._embeddings is not None and self._offsets is not None:
                    position = self._video_order.index(video_id)
                    global_index = int(self._offsets[position]) + local_index
                    score = float(self._embeddings[global_index] @ query_vector)
            result = self._candidate_to_result(
                _DirectCandidate(video_id, local_index, score),
            )
            if result is not None:
                output.append((local_index, result))
        self.ensure_result_images([result for _index, result in output])
        return output

    def _preprocessed_object_labels(self, video_id: str, keyframe_number: int) -> tuple[str, ...]:
        cached = self._direct_object_cache.setdefault(video_id, {})
        if keyframe_number in cached:
            return cached[keyframe_number]
        matrix_bundle = self._direct_object_matrices.get(video_id)
        if video_id not in self._direct_object_matrices:
            video_root = self.preprocessed_root / "videos" / video_id
            scores_path = video_root / "object_scores.npy"
            classes_path = video_root / "object_classes.json"
            try:
                class_payload = json.loads(classes_path.read_text(encoding="utf-8"))
                class_map = {
                    int(index): str(label)
                    for index, label in dict(class_payload.get("classes") or {}).items()
                }
                scores = np.load(scores_path, mmap_mode="r")
                if scores.ndim != 2 or scores.shape[1] < 1:
                    raise ValueError(f"object score shape không hợp lệ: {scores.shape}")
                labels = tuple(class_map.get(index, str(index)) for index in range(scores.shape[1]))
                matrix_bundle = (scores, labels)
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                matrix_bundle = None
            self._direct_object_matrices[video_id] = matrix_bundle
        if matrix_bundle is not None:
            scores, labels = matrix_bundle
            if 0 <= keyframe_number < len(scores):
                row = np.asarray(scores[keyframe_number], dtype=np.float32)
                indices = np.flatnonzero(row > 0)
                indices = indices[np.argsort(row[indices])[::-1]]
                matched = tuple(labels[int(index)] for index in indices)
                cached[keyframe_number] = matched
                return matched

        # Compatibility fallback for early artifacts that contain detailed
        # JSON boxes but predate the dense score matrix.
        path = self.preprocessed_root / "videos" / video_id / "objects.jsonl.gz"
        if path.is_file():
            try:
                with gzip.open(path, "rt", encoding="utf-8") as stream:
                    for line in track(
                        stream,
                        desc=f"Đọc direct objects {video_id}",
                        unit="frame",
                        nested=True,
                    ):
                        payload = json.loads(line)
                        labels = tuple(
                            dict.fromkeys(
                                str(item.get("label") or "").strip()
                                for item in payload.get("objects", [])
                                if str(item.get("label") or "").strip()
                            )
                        )
                        cached[int(payload["keyframe_number"])] = labels
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                return ()
        return cached.get(keyframe_number, ())

    def search(
        self,
        query: str,
        english_expansion: str = "",
        *,
        top_k: int = 100,
        min_frame_gap: int = 90,
        max_per_video: int | None = None,
        video_id: str | None = None,
        metadata_weight: float = 0.10,
    ) -> list[SearchResult]:
        if not 1 <= top_k <= 100:
            raise ValueError("top_k phải nằm trong khoảng 1..100 theo quy định AIC.")
        if min_frame_gap < 0:
            raise ValueError("min_frame_gap không thể âm.")
        if max_per_video is not None and max_per_video < 1:
            raise ValueError("max_per_video phải lớn hơn 0.")
        query_vector = self.encoder.encode(query, english_expansion)
        allowed = {video_id} if video_id else None
        candidates = self._raw_candidates(query_vector, top_k, allowed)
        results = [
            result
            for candidate in track(
                candidates,
                desc="Tạo direct kết quả",
                total=len(candidates),
                unit="frame",
                nested=True,
            )
            if (result := self._candidate_to_result(candidate)) is not None
        ]
        chosen: list[SearchResult] = []
        frames_by_video: dict[str, list[int]] = {}
        for result in track(
            results,
            desc="Chọn direct kết quả đa dạng",
            total=len(results),
            unit="frame",
        ):
            nearby = frames_by_video.setdefault(result.video_id, [])
            if max_per_video is not None and len(nearby) >= max_per_video:
                continue
            if any(abs(result.frame_id - previous) <= min_frame_gap for previous in nearby):
                continue
            nearby.append(result.frame_id)
            result.rank = len(chosen) + 1
            chosen.append(result)
            if len(chosen) == top_k:
                break
        translated = getattr(getattr(self.encoder, "last_query", None), "text_for_model", "")
        query_tokens = set(tokenize(f"{query} {english_expansion} {translated}"))
        for result in track(
            chosen,
            desc="Gắn direct object labels",
            total=len(chosen),
            unit="frame",
            nested=True,
        ):
            result.object_labels = self._preprocessed_object_labels(
                result.video_id,
                result.keyframe_number,
            )
            result.object_score = AICRetrievalEngine._object_match_score(
                result.object_labels,
                query_tokens,
            )
            result.score += 0.03 * result.object_score
            result.retrieval_score = result.score
        chosen.sort(key=lambda item: item.score, reverse=True)
        for rank, result in track(
            enumerate(chosen, start=1),
            desc="Xếp hạng direct object",
            total=len(chosen),
            unit="frame",
            nested=True,
        ):
            result.rank = rank
        return chosen

    def search_trake(
        self,
        events: Sequence[str],
        english_events: Sequence[str] | None = None,
        *,
        top_videos: int = 10,
    ) -> list[TrakeVideoResult]:
        clean_events = [event.strip() for event in events if event and event.strip()]
        if len(clean_events) < 2:
            raise ValueError("TRAKE cần ít nhất 2 mốc sự kiện, mỗi mốc một dòng.")
        if len(clean_events) > 12:
            raise ValueError("Giới hạn 12 mốc để demo phản hồi nhanh.")
        self.prepare_runtime()
        english_events = list(english_events or [])
        event_vectors = [
            self.encoder.encode(event, english_events[index] if index < len(english_events) else "")
            for index, event in track(
                enumerate(clean_events),
                desc="Mã hóa direct TRAKE events",
                total=len(clean_events),
                unit="event",
                force=True,
            )
        ]
        vectors = np.stack(event_vectors)
        if self._preprocessed_shards:
            return self._search_trake_hierarchical(clean_events, vectors, top_videos)
        assert self._embeddings is not None
        assert self._offsets is not None
        candidates: list[tuple[float, str, np.ndarray, np.ndarray]] = []
        for position, video_id in track(
            enumerate(self._video_order),
            desc="TRAKE scan direct videos",
            total=len(self._video_order),
            unit="video",
            force=True,
        ):
            start, end = self._offsets[position : position + 2]
            features = self._embeddings[int(start) : int(end)]
            if len(features) < len(clean_events):
                continue
            scores = np.asarray(features @ vectors.T, dtype=np.float32)
            alignment = AICRetrievalEngine._ordered_alignment(scores)
            if alignment is None:
                continue
            indices, best_scores = alignment
            sequence_score = 0.72 * float(best_scores.mean()) + 0.28 * float(best_scores.min())
            candidates.append((sequence_score, video_id, indices, best_scores))

        output: list[TrakeVideoResult] = []
        best_candidates = sorted(candidates, key=lambda item: item[0], reverse=True)[:top_videos]
        for rank, (score, video_id, indices, values) in track(
            enumerate(best_candidates, start=1),
            desc="Dựng direct TRAKE sequences",
            total=len(best_candidates),
            unit="video",
        ):
            frames: list[SearchResult] = []
            for local_index, event_score in track(
                zip(indices, values),
                desc=f"Direct TRAKE {video_id}",
                total=len(indices),
                unit="event",
                nested=True,
            ):
                result = self._candidate_to_result(
                    _DirectCandidate(video_id, int(local_index), float(event_score)),
                )
                if result is None:
                    break
                result.rank = rank
                result.score = float(event_score)
                frames.append(result)
            if len(frames) == len(clean_events):
                output.append(TrakeVideoResult(rank, video_id, score, frames))
        return output

    def _search_trake_hierarchical(
        self,
        events: Sequence[str],
        event_vectors: np.ndarray,
        top_videos: int,
    ) -> list[TrakeVideoResult]:
        """Retrieve TRAKE videos on %N frames, then align %2/%1 locally."""
        assert self._coarse_embeddings is not None
        assert self._coarse_local_indices is not None
        assert self._coarse_offsets is not None
        candidates: list[tuple[float, str, np.ndarray, np.ndarray]] = []
        for video_position, video_id in track(
            enumerate(self._video_order),
            desc=f"TRAKE global frame %{self.frame_steps[0]}",
            total=len(self._video_order),
            unit="video",
            force=True,
        ):
            if video_id not in self._preprocessed_shards:
                continue
            start, end = self._coarse_offsets[video_position : video_position + 2]
            if int(end - start) < len(events):
                continue
            features = self._coarse_embeddings[int(start) : int(end)]
            scores = np.asarray(features @ event_vectors.T, dtype=np.float32)
            alignment = AICRetrievalEngine._ordered_alignment(scores)
            if alignment is None:
                continue
            coarse_positions, best_scores = alignment
            local_indices = np.asarray(
                self._coarse_local_indices[int(start) : int(end)][coarse_positions],
                dtype=np.int32,
            )
            sequence_score = 0.72 * float(best_scores.mean()) + 0.28 * float(best_scores.min())
            candidates.append((sequence_score, video_id, local_indices, best_scores))

        candidates.sort(key=lambda item: (-item[0], item[1]))
        refine_count = min(
            len(candidates),
            max(top_videos, self.trake_refine_videos),
        )
        refined_candidates: list[tuple[float, str, np.ndarray, np.ndarray]] = []
        for _coarse_score, video_id, local_indices, values in track(
            candidates[:refine_count],
            desc="TRAKE modulo refinement",
            total=refine_count,
            unit="video",
            force=True,
        ):
            shard = self._preprocessed_shards[video_id]
            selected_indices = np.asarray(local_indices, dtype=np.int32)
            selected_scores = np.asarray(values, dtype=np.float32)
            for previous_step, next_step in zip(self.frame_steps, self.frame_steps[1:]):
                event_frame_indices: list[list[int]] = []
                event_candidate_scores: list[list[float]] = []
                for event_index, center in track(
                    enumerate(selected_indices),
                    desc=f"TRAKE {video_id} %{next_step}",
                    total=len(events),
                    unit="event",
                    nested=True,
                ):
                    candidate_indices = temporal_modulo_indices(
                        shard.frame_ids,
                        [int(center)],
                        previous_step,
                        next_step,
                    )
                    if not len(candidate_indices):
                        event_frame_indices.append([])
                        event_candidate_scores.append([])
                        continue
                    features = self._normalize_rows(
                        np.asarray(shard.embeddings[candidate_indices], dtype=np.float32)
                    )
                    scores = np.asarray(features @ event_vectors[event_index], dtype=np.float32)
                    event_frame_indices.append([int(index) for index in candidate_indices])
                    event_candidate_scores.append([float(score) for score in scores])
                choices = self._ordered_candidate_alignment(
                    event_frame_indices,
                    event_candidate_scores,
                )
                if choices is None:
                    break
                selected_indices = np.asarray(
                    [
                        event_frame_indices[event_index][choice]
                        for event_index, choice in enumerate(choices)
                    ],
                    dtype=np.int32,
                )
                selected_scores = np.asarray(
                    [
                        event_candidate_scores[event_index][choice]
                        for event_index, choice in enumerate(choices)
                    ],
                    dtype=np.float32,
                )
            score = 0.72 * float(selected_scores.mean()) + 0.28 * float(selected_scores.min())
            refined_candidates.append((score, video_id, selected_indices, selected_scores))

        best_candidates = sorted(
            refined_candidates,
            key=lambda item: (-item[0], item[1]),
        )[:top_videos]
        output: list[TrakeVideoResult] = []
        for rank, (score, video_id, indices, values) in track(
            enumerate(best_candidates, start=1),
            desc="Dựng hierarchical TRAKE sequences",
            total=len(best_candidates),
            unit="video",
            force=True,
        ):
            frames: list[SearchResult] = []
            for local_index, event_score in track(
                zip(indices, values),
                desc=f"Direct TRAKE {video_id}",
                total=len(indices),
                unit="event",
                nested=True,
            ):
                result = self._candidate_to_result(
                    _DirectCandidate(video_id, int(local_index), float(event_score)),
                )
                if result is None:
                    break
                result.rank = rank
                result.score = float(event_score)
                frames.append(result)
            if len(frames) == len(events):
                output.append(TrakeVideoResult(rank, video_id, score, frames))
        return output

    @staticmethod
    def _ordered_candidate_alignment(
        frame_indices: Sequence[Sequence[int]],
        candidate_scores: Sequence[Sequence[float]],
    ) -> list[int] | None:
        return AICRetrievalEngine._ordered_candidate_alignment(frame_indices, candidate_scores)
