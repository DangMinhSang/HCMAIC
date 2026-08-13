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


class DirectVideoUnavailableError(RuntimeError):
    """Raised when the opt-in raw-video path cannot be initialized."""


@dataclass(frozen=True)
class DirectFrame:
    """One sampled frame in the locally generated direct-video index."""

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
    ``keyframe_number`` is the sampled-frame ordinal. This preserves temporal
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
        runtime_dir = Path(os.environ.get("AIC_RUNTIME_DIR", "/kaggle/working")) / "aic_direct_video_cache"
        self.runtime_dir = runtime_dir
        self.preprocessed_root = Path(
            os.environ.get("AIC_DIRECT_PREPROCESSED_ROOT", "/kaggle/working/aic_direct_preprocessed")
        ).expanduser()
        self._cache_path = runtime_dir / f"index_{self._cache_key()}.npz"
        self._frame_cache_dir = runtime_dir / f"frames_{self._cache_key()}"
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
        return int(len(self._embeddings)) if self._embeddings is not None else len(self._records)

    @property
    def feature_cache_loaded(self) -> bool:
        return self._embeddings is not None

    @property
    def source_description(self) -> str:
        preprocessed = (
            f" · preprocessed={self._preprocessed_video_count}/{len(self._video_order)} video"
            if self._preprocessed_video_count
            else ""
        )
        return (
            f"{self.source_kind}:{self.dataset_root} · stride={self.sample_stride} frame · "
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
        if self._embeddings is not None:
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
        """Load shard artifacts produced by ``--pre-direct-video 1``."""
        if self._preprocessed_checked:
            return False
        self._preprocessed_checked = True
        videos_root = self.preprocessed_root / "videos"
        if not videos_root.is_dir():
            return False
        vector_parts: list[np.ndarray] = []
        records: list[DirectFrame] = []
        loaded_videos = 0
        for video_id in track(
            self._video_order,
            desc="Nạp pre-direct CLIP shards",
            total=len(self._video_order),
            unit="video",
            force=True,
            leave=True,
        ):
            video_dir = videos_root / video_id
            clip_path = video_dir / "clip.npy"
            mapping_path = video_dir / "mapping.jsonl"
            marker_path = video_dir / "visual.complete.json"
            if not (clip_path.is_file() and mapping_path.is_file() and marker_path.is_file()):
                continue
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                artifact_model = str(marker.get("clip_model") or "")
                if artifact_model != self.encoder.model_name:
                    raise ValueError(
                        f"CLIP model artifact={artifact_model!r}, runtime={self.encoder.model_name!r}"
                    )
                vectors = np.asarray(np.load(clip_path, mmap_mode="r"), dtype=np.float32)
                mappings = []
                with mapping_path.open(encoding="utf-8") as stream:
                    for line in track(
                        stream,
                        desc=f"Mapping pre-direct {video_id}",
                        unit="frame",
                        nested=True,
                    ):
                        payload = json.loads(line)
                        image = video_dir / str(payload.get("image") or "")
                        mappings.append(
                            DirectFrame(
                                keyframe_number=int(payload["keyframe_number"]),
                                frame_id=int(payload["frame_id"]),
                                pts_time=float(payload["pts_time"]),
                                fps=float(payload.get("fps") or 30.0),
                                source_path=self._features[video_id],
                                image_path=image if image.is_file() else None,
                            )
                        )
                if vectors.ndim != 2 or vectors.shape[1] < 1 or len(vectors) != len(mappings):
                    raise ValueError(f"clip={vectors.shape}, mapping={len(mappings)}")
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                print(f"[warning] Bỏ pre-direct artifact lỗi {video_id}: {error}", flush=True)
                continue
            vector_parts.append(vectors)
            records.extend(mappings)
            loaded_videos += 1
        if not vector_parts:
            return False
        self._embeddings = self._normalize_rows(np.concatenate(vector_parts, axis=0))
        self._set_records(records)
        self._preprocessed_video_count = loaded_videos
        print(
            f"Đã nạp pre-direct index: {len(records):,} frame từ "
            f"{loaded_videos:,}/{len(self._video_order):,} video → {self.preprocessed_root}",
            flush=True,
        )
        if loaded_videos < len(self._video_order):
            print(
                "[warning] Pre-direct index mới là một phần corpus; query chỉ phủ các shard đã hoàn tất.",
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

    def _mapping(self, video_id: str) -> list[DirectFrame]:
        self.prepare_runtime()
        return self._records_by_video.get(video_id, [])

    def _materialize_frame(self, frame: DirectFrame) -> Path | None:
        if frame.image_path is not None and frame.image_path.is_file():
            return frame.image_path
        target = self._frame_cache_dir / frame.source_path.stem / f"{frame.frame_id:08d}.jpg"
        if target.is_file():
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
            return target
        finally:
            capture.release()

    def ensure_result_image(self, result: SearchResult) -> str | None:
        """Materialize one returned frame only when a UI/model needs it."""
        if result.image_path and Path(result.image_path).is_file():
            return result.image_path
        self._mapping(result.video_id)
        frame = self._mapping_lookup.get(result.video_id, {}).get(result.keyframe_number)
        if frame is None:
            return None
        image = self._materialize_frame(frame)
        result.image_path = str(image) if image is not None else None
        return result.image_path

    def _candidate_to_result(
        self,
        candidate: _DirectCandidate,
        *,
        materialize_image: bool = False,
    ) -> SearchResult | None:
        mapping = self._mapping(candidate.video_id)
        if candidate.feature_index < 0 or candidate.feature_index >= len(mapping):
            return None
        frame = mapping[candidate.feature_index]
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
        self.ensure_result_image(result)
        return result

    def neighboring_keyframes(
        self,
        video_id: str,
        keyframe_number: int,
        *,
        radius: int = 1,
        query_vector: np.ndarray | None = None,
    ) -> list[tuple[int, SearchResult]]:
        records = self._mapping(video_id)
        center = self._mapping_positions.get(video_id, {}).get(keyframe_number)
        if center is None:
            return []
        radius = max(0, min(int(radius), 5))
        start = max(0, center - radius)
        end = min(len(records), center + radius + 1)
        output: list[tuple[int, SearchResult]] = []
        for local_index in track(
            range(start, end),
            desc="Direct frame lân cận",
            total=max(0, end - start),
            unit="frame",
            nested=True,
        ):
            score = 0.0
            if query_vector is not None and self._embeddings is not None and self._offsets is not None:
                position = self._video_order.index(video_id)
                global_index = int(self._offsets[position]) + local_index
                score = float(self._embeddings[global_index] @ query_vector)
            result = self._candidate_to_result(
                _DirectCandidate(video_id, local_index, score),
                materialize_image=True,
            )
            if result is not None:
                output.append((local_index, result))
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
        assert self._embeddings is not None
        assert self._offsets is not None
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
                    materialize_image=True,
                )
                if result is None:
                    break
                result.rank = rank
                result.score = float(event_score)
                frames.append(result)
            if len(frames) == len(clean_events):
                output.append(TrakeVideoResult(rank, video_id, score, frames))
        return output

    @staticmethod
    def _ordered_candidate_alignment(
        frame_indices: Sequence[Sequence[int]],
        candidate_scores: Sequence[Sequence[float]],
    ) -> list[int] | None:
        return AICRetrievalEngine._ordered_candidate_alignment(frame_indices, candidate_scores)
