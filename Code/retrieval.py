"""Feature-first retrieval for the AIC 2026 preliminary tasks.

The search reads pre-computed CLIP vectors with NumPy memory maps.  It does
not build a duplicate vector index, copy image/video files, or download data.
"""

from __future__ import annotations

import csv
import heapq
import json
import math
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from clip_encoder import ClipTextEncoder
from data_paths import AICPaths


TOKEN_RE = re.compile(r"[\w]+", flags=re.UNICODE)
LABEL_KEYS = {
    "label",
    "labels",
    "name",
    "class",
    "class_name",
    "display_name",
    "category",
    "category_name",
}


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFD", value.lower())
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    return value.replace("đ", "d")


def tokenize(value: str) -> list[str]:
    return TOKEN_RE.findall(normalize_text(value))


@dataclass(frozen=True)
class FrameMapping:
    """The official mapping from a keyframe sequence number to video frame."""

    keyframe_number: int
    frame_id: int
    pts_time: float
    fps: float


@dataclass
class VideoMetadata:
    title: str = ""
    description: str = ""
    keywords: tuple[str, ...] = ()

    @property
    def searchable_text(self) -> str:
        return " ".join((self.title, self.description, " ".join(self.keywords)))


@dataclass
class SearchResult:
    rank: int
    video_id: str
    frame_id: int
    keyframe_number: int
    pts_time: float
    visual_score: float
    metadata_score: float
    score: float
    title: str = ""
    object_labels: tuple[str, ...] = ()
    image_path: str | None = None
    video_path: str | None = None
    answer: str = ""
    qa_confidence: float = 0.0
    ocr_score: float = 0.0
    ocr_text: str = ""

    def caption(self) -> str:
        labels = ", ".join(self.object_labels[:6]) or "không có object metadata"
        answer = f"\nQ&A: {self.answer} ({self.qa_confidence:.0%})" if self.answer else ""
        return (
            f"#{self.rank} | {self.video_id}.mp4 | frame {self.frame_id} | "
            f"{self.pts_time:.2f}s | score {self.score:.4f}\n{self.title}\nObjects: {labels}{answer}"
        )

    def table_row(self) -> list[Any]:
        return [
            self.rank,
            self.video_id,
            self.frame_id,
            self.keyframe_number,
            round(self.pts_time, 3),
            round(self.visual_score, 5),
            round(self.metadata_score, 5),
            round(self.score, 5),
            self.title,
            ", ".join(self.object_labels[:8]),
        ]


@dataclass
class TrakeVideoResult:
    rank: int
    video_id: str
    score: float
    frames: list[SearchResult]

    def table_rows(self) -> list[list[Any]]:
        rows: list[list[Any]] = []
        for event_index, result in enumerate(self.frames, start=1):
            rows.append(
                [
                    self.rank,
                    self.video_id,
                    event_index,
                    result.frame_id,
                    result.keyframe_number,
                    round(result.pts_time, 3),
                    round(result.visual_score, 5),
                    round(self.score, 5),
                    result.title,
                ]
            )
        return rows


@dataclass(frozen=True)
class _Candidate:
    video_id: str
    feature_index: int
    visual_score: float


class AICRetrievalEngine:
    """CLIP retrieval with metadata/object reranking and temporal diversity."""

    def __init__(self, paths: AICPaths, encoder: ClipTextEncoder | None = None) -> None:
        self.paths = paths
        self.encoder = encoder or ClipTextEncoder()
        self._features = {
            feature.stem: feature
            for feature in sorted(paths.features_dir.glob("*.npy"))
            if (paths.mapping_dir / f"{feature.stem}.csv").is_file()
        }
        if not self._features:
            raise FileNotFoundError(
                f"Không tìm thấy cặp .npy/.csv trong {paths.features_dir} và {paths.mapping_dir}"
            )
        self._mapping_cache: dict[str, list[FrameMapping]] = {}
        self._metadata_cache: dict[str, VideoMetadata] = {}
        self._metadata_tokens: dict[str, Counter[str]] = {}
        self._metadata_ready = False
        self._vector_count: int | None = None
        self._ram_features: np.ndarray | None = None
        self._ram_offsets: np.ndarray | None = None
        self._ram_video_ids: tuple[str, ...] = ()
        if os.environ.get("AIC_PRELOAD_FEATURES", "0").lower() in {"1", "true", "yes"}:
            self.preload_features()

    @classmethod
    def from_environment(cls, input_root: str | Path | None = None) -> "AICRetrievalEngine":
        return cls(AICPaths.from_environment(input_root))

    @property
    def video_count(self) -> int:
        return len(self._features)

    @property
    def vector_count(self) -> int:
        """Count vectors from .npy headers only; feature values stay memory-mapped."""
        if self._vector_count is None:
            self._vector_count = sum(
                int(np.load(path, mmap_mode="r").shape[0]) for path in self._features.values()
            )
        return self._vector_count

    @property
    def feature_cache_loaded(self) -> bool:
        return self._ram_features is not None

    def preload_features(self) -> None:
        """Keep normalized CLIP vectors in RAM for low-latency repeated queries.

        This reads the mounted ``.npy`` files once but never creates a copy on
        disk. Batch 1 needs roughly 0.4 GB; larger releases can need around
        1–2 GB, hence this is opt-in through ``AIC_PRELOAD_FEATURES=1``.
        """
        if self._ram_features is not None:
            return
        video_ids = tuple(self._features)
        shapes = [np.load(self._features[video_id], mmap_mode="r").shape for video_id in video_ids]
        if not shapes or any(len(shape) != 2 or shape[1] != 512 for shape in shapes):
            raise ValueError("Không thể preload feature: cần các mảng CLIP 512 chiều.")
        offsets = np.zeros(len(video_ids) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([shape[0] for shape in shapes])
        matrix = np.empty((int(offsets[-1]), 512), dtype=np.float32)
        for index, video_id in enumerate(video_ids):
            values = np.asarray(np.load(self._features[video_id]), dtype=np.float32)
            values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
            matrix[offsets[index] : offsets[index + 1]] = values
        self._ram_features = matrix
        self._ram_offsets = offsets
        self._ram_video_ids = video_ids

    def _ram_candidates(
        self,
        query_vector: np.ndarray,
        candidate_pool: int,
        allowed_video_ids: set[str] | None,
    ) -> list[_Candidate]:
        """Top candidates from one matrix multiplication over RAM-resident CLIP."""
        if self._ram_features is None or self._ram_offsets is None:
            return []
        if allowed_video_ids and len(allowed_video_ids) == 1:
            video_id = next(iter(allowed_video_ids))
            try:
                video_position = self._ram_video_ids.index(video_id)
            except ValueError:
                return []
            start, end = self._ram_offsets[video_position : video_position + 2]
            scores = self._ram_features[start:end] @ query_vector
            indices = self._top_indices(scores, min(len(scores), candidate_pool * 8))
            return [_Candidate(video_id, int(index), float(scores[index])) for index in indices]

        scores = self._ram_features @ query_vector
        # NMS and max-per-video happen later, so retrieve a wider raw pool.
        indices = self._top_indices(scores, min(len(scores), candidate_pool * 10))
        output: list[_Candidate] = []
        for index in indices:
            video_position = int(np.searchsorted(self._ram_offsets, index, side="right") - 1)
            video_id = self._ram_video_ids[video_position]
            if allowed_video_ids is not None and video_id not in allowed_video_ids:
                continue
            output.append(
                _Candidate(video_id, int(index - self._ram_offsets[video_position]), float(scores[index]))
            )
        return output

    def _mapping(self, video_id: str) -> list[FrameMapping]:
        cached = self._mapping_cache.get(video_id)
        if cached is not None:
            return cached
        mappings: list[FrameMapping] = []
        with (self.paths.mapping_dir / f"{video_id}.csv").open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                try:
                    mappings.append(
                        FrameMapping(
                            keyframe_number=int(float(row["n"])),
                            frame_id=int(float(row["frame_idx"])),
                            pts_time=float(row["pts_time"]),
                            fps=float(row["fps"]),
                        )
                    )
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"Mapping không hợp lệ: {video_id}.csv") from error
        self._mapping_cache[video_id] = mappings
        return mappings

    def _metadata(self, video_id: str) -> VideoMetadata:
        cached = self._metadata_cache.get(video_id)
        if cached is not None:
            return cached
        metadata = VideoMetadata()
        if self.paths.metadata_dir is not None:
            path = self.paths.metadata_dir / f"{video_id}.json"
            if path.is_file():
                try:
                    with path.open(encoding="utf-8") as stream:
                        source = json.load(stream)
                    keywords = source.get("keywords") or []
                    if not isinstance(keywords, list):
                        keywords = [str(keywords)]
                    metadata = VideoMetadata(
                        title=str(source.get("title") or ""),
                        description=str(source.get("description") or ""),
                        keywords=tuple(str(word) for word in keywords),
                    )
                except (OSError, json.JSONDecodeError, TypeError):
                    # Metadata is optional according to the AIC specification.
                    metadata = VideoMetadata()
        self._metadata_cache[video_id] = metadata
        return metadata

    def _prepare_metadata(self) -> None:
        if self._metadata_ready:
            return
        for video_id in self._features:
            self._metadata_tokens[video_id] = Counter(tokenize(self._metadata(video_id).searchable_text))
        self._metadata_ready = True

    def _metadata_scores(self, query: str, video_ids: Iterable[str]) -> dict[str, float]:
        """A bounded lexical score for names/dates/channels missing from CLIP."""
        tokens = [token for token in tokenize(query) if len(token) > 1]
        unique_video_ids = set(video_ids)
        if not tokens or self.paths.metadata_dir is None:
            return {video_id: 0.0 for video_id in unique_video_ids}
        self._prepare_metadata()
        query_counts = Counter(tokens)
        scores: dict[str, float] = {}
        for video_id in unique_video_ids:
            document = self._metadata_tokens.get(video_id, Counter())
            if not document:
                scores[video_id] = 0.0
                continue
            matched = sum(min(count, document.get(token, 0)) for token, count in query_counts.items())
            scores[video_id] = matched / sum(query_counts.values())
        return scores

    def _object_labels(self, video_id: str, keyframe_number: int) -> tuple[str, ...]:
        path = self.paths.object_path(video_id, keyframe_number)
        if path is None:
            return ()
        try:
            with path.open(encoding="utf-8") as stream:
                payload = json.load(stream)
        except (OSError, json.JSONDecodeError):
            return ()

        labels: list[str] = []

        def visit(node: Any, key: str = "") -> None:
            if isinstance(node, dict):
                for child_key, child in node.items():
                    visit(child, child_key.lower())
            elif isinstance(node, list):
                for child in node:
                    visit(child, key)
            elif key in LABEL_KEYS and isinstance(node, str) and node.strip():
                labels.append(node.strip())

        visit(payload)
        return tuple(dict.fromkeys(labels))

    @staticmethod
    def _top_indices(scores: np.ndarray, limit: int) -> np.ndarray:
        if len(scores) == 0:
            return np.array([], dtype=np.int64)
        limit = min(limit, len(scores))
        if limit == len(scores):
            indices = np.arange(len(scores))
        else:
            indices = np.argpartition(scores, -limit)[-limit:]
        return indices[np.argsort(scores[indices])[::-1]]

    @staticmethod
    def _cosine_scores(features: np.ndarray, query_vector: np.ndarray) -> np.ndarray:
        # Some releases have normalized vectors and some do not. Explicit
        # normalization preserves cosine ranking in both cases.
        scores = np.asarray(features @ query_vector, dtype=np.float32)
        norms = np.linalg.norm(features, axis=1)
        return scores / np.maximum(norms, 1e-12)

    def _candidate_to_result(self, candidate: _Candidate) -> SearchResult | None:
        mapping = self._mapping(candidate.video_id)
        # A few supplied arrays contain one trailing feature without a mapping.
        # Ignore it rather than inventing a frame id, which is essential for the
        # exact-frame scoring rule in the PDF.
        if candidate.feature_index >= len(mapping):
            return None
        frame = mapping[candidate.feature_index]
        metadata = self._metadata(candidate.video_id)
        image = self.paths.image_path(candidate.video_id, frame.keyframe_number)
        video = self.paths.video_path(candidate.video_id)
        return SearchResult(
            rank=0,
            video_id=candidate.video_id,
            frame_id=frame.frame_id,
            keyframe_number=frame.keyframe_number,
            pts_time=frame.pts_time,
            visual_score=candidate.visual_score,
            metadata_score=0.0,
            score=candidate.visual_score,
            title=metadata.title,
            image_path=str(image) if image is not None else None,
            video_path=str(video) if video is not None else None,
        )

    def result_for_keyframe(
        self,
        video_id: str,
        keyframe_number: int,
        *,
        score: float = 0.0,
        ocr_score: float = 0.0,
        ocr_text: str = "",
    ) -> SearchResult | None:
        """Materialize an OCR hit with the official keyframe-to-frame mapping."""
        frame = next(
            (item for item in self._mapping(video_id) if item.keyframe_number == keyframe_number),
            None,
        )
        if frame is None:
            return None
        metadata = self._metadata(video_id)
        image = self.paths.image_path(video_id, keyframe_number)
        video = self.paths.video_path(video_id)
        return SearchResult(
            rank=0,
            video_id=video_id,
            frame_id=frame.frame_id,
            keyframe_number=keyframe_number,
            pts_time=frame.pts_time,
            visual_score=0.0,
            metadata_score=0.0,
            score=score,
            title=metadata.title,
            image_path=str(image) if image is not None else None,
            video_path=str(video) if video is not None else None,
            ocr_score=ocr_score,
            ocr_text=ocr_text,
        )

    def _raw_candidates(
        self,
        query_vector: np.ndarray,
        top_k: int,
        allowed_video_ids: set[str] | None = None,
    ) -> list[_Candidate]:
        candidate_pool = max(400, top_k * 25)
        if self._ram_features is not None:
            return self._ram_candidates(query_vector, candidate_pool, allowed_video_ids)
        per_video_limit = max(30, min(100, candidate_pool // 8))
        all_candidates: list[_Candidate] = []
        for video_id, feature_path in self._features.items():
            if allowed_video_ids is not None and video_id not in allowed_video_ids:
                continue
            features = np.load(feature_path, mmap_mode="r")
            if features.ndim != 2 or features.shape[1] != query_vector.shape[0]:
                raise ValueError(
                    f"Feature {feature_path.name} có shape {features.shape}, không khớp "
                    f"CLIP ViT-B/32 ({query_vector.shape[0]} chiều)."
                )
            scores = self._cosine_scores(features, query_vector)
            for index in self._top_indices(scores, per_video_limit):
                all_candidates.append(_Candidate(video_id, int(index), float(scores[index])))
        return heapq.nlargest(candidate_pool, all_candidates, key=lambda item: item.visual_score)

    @staticmethod
    def _object_match_score(labels: Sequence[str], query_tokens: set[str]) -> float:
        if not labels or not query_tokens:
            return 0.0
        label_tokens = set(token for label in labels for token in tokenize(label))
        if not label_tokens:
            return 0.0
        return len(label_tokens & query_tokens) / len(query_tokens)

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
        """Search Textual KIS/Q&A event descriptions.

        ``min_frame_gap`` removes near-identical neighbouring I-frames from
        each video after ranking. Keeping it around 90 frames (3 seconds at
        30fps) increases coverage across the 100 permitted submissions.
        """
        if not 1 <= top_k <= 100:
            raise ValueError("top_k phải nằm trong khoảng 1..100 theo quy định AIC.")
        if min_frame_gap < 0:
            raise ValueError("min_frame_gap không thể âm.")
        if max_per_video is not None and max_per_video < 1:
            raise ValueError("max_per_video phải lớn hơn 0.")
        metadata_weight = min(max(float(metadata_weight), 0.0), 0.35)
        query_vector = self.encoder.encode(query, english_expansion)
        allowed_video_ids = {video_id} if video_id else None
        if allowed_video_ids is not None and video_id not in self._features:
            raise ValueError(f"Không có video `{video_id}` trong CLIP features.")
        results = [
            self._candidate_to_result(item)
            for item in self._raw_candidates(query_vector, top_k, allowed_video_ids)
        ]
        results = [item for item in results if item is not None]
        translated = getattr(getattr(self.encoder, "last_query", None), "text_for_model", "")
        metadata_query = f"{query} {english_expansion} {translated}"
        metadata_scores = self._metadata_scores(metadata_query, (r.video_id for r in results))
        query_tokens = set(tokenize(metadata_query))

        for result in results:
            result.metadata_score = metadata_scores.get(result.video_id, 0.0)
            result.object_labels = self._object_labels(result.video_id, result.keyframe_number)
            object_score = self._object_match_score(result.object_labels, query_tokens)
            # Object labels are only a small tie-breaker: Faster R-CNN has a
            # narrower vocabulary than CLIP and should never dominate vision.
            result.score = (
                (1.0 - metadata_weight) * result.visual_score
                + metadata_weight * result.metadata_score
                + 0.03 * object_score
            )
        results.sort(key=lambda item: item.score, reverse=True)

        chosen: list[SearchResult] = []
        frames_by_video: dict[str, list[int]] = {}
        for result in results:
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
        return chosen

    def search_trake(
        self,
        events: Sequence[str],
        english_events: Sequence[str] | None = None,
        *,
        top_videos: int = 10,
    ) -> list[TrakeVideoResult]:
        """Retrieve one video then align one semantic frame for every event.

        Each line in ``events`` must be a temporal stage. The video score is
        the mean of the best cosine similarity for all stages, preventing a
        video that contains just one stage from winning retrieval.
        """
        clean_events = [event.strip() for event in events if event and event.strip()]
        if len(clean_events) < 2:
            raise ValueError("TRAKE cần ít nhất 2 mốc sự kiện, mỗi mốc một dòng.")
        if len(clean_events) > 12:
            raise ValueError("Giới hạn 12 mốc để demo phản hồi nhanh.")
        english_events = list(english_events or [])
        vectors = np.stack(
            [
                self.encoder.encode(event, english_events[index] if index < len(english_events) else "")
                for index, event in enumerate(clean_events)
            ]
        )

        video_candidates: list[tuple[float, str, np.ndarray, np.ndarray]] = []
        for video_id, feature_path in self._features.items():
            features = np.load(feature_path, mmap_mode="r")
            if features.ndim != 2 or features.shape[1] != vectors.shape[1]:
                continue
            scores = np.asarray(features @ vectors.T, dtype=np.float32)
            scores /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
            best_indices = scores.argmax(axis=0)
            best_scores = scores[best_indices, np.arange(len(clean_events))]
            video_candidates.append((float(best_scores.mean()), video_id, best_indices, best_scores))

        best_videos = heapq.nlargest(top_videos, video_candidates, key=lambda item: item[0])
        output: list[TrakeVideoResult] = []
        for rank, (score, video_id, indices, scores) in enumerate(best_videos, start=1):
            aligned: list[SearchResult] = []
            for feature_index, event_score in zip(indices, scores):
                result = self._candidate_to_result(
                    _Candidate(video_id, int(feature_index), float(event_score))
                )
                if result is None:
                    break
                result.rank = rank
                result.score = float(event_score)
                aligned.append(result)
            if len(aligned) == len(clean_events):
                output.append(TrakeVideoResult(rank, video_id, score, aligned))
        return output


def write_kis_submission(
    results: Sequence[SearchResult],
    destination: str | Path,
    answer: str = "",
    *,
    include_answer: bool | None = None,
) -> Path:
    """Write ordered answers in the PDF's Textual KIS or Q&A field format."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    has_answer = bool(answer.strip()) if include_answer is None else include_answer
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["video_id", "frame_id", "answer"] if has_answer else ["video_id", "frame_id"])
        for result in results[:100]:
            row = [result.video_id, result.frame_id]
            if has_answer:
                row.append(result.answer or answer.strip())
            writer.writerow(row)
    return destination


def write_trake_submission(results: Sequence[TrakeVideoResult], destination: str | Path) -> Path:
    """Write ranked TRAKE candidates; every row contains one aligned sequence."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    max_events = max((len(item.frames) for item in results), default=0)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["video_id", *[f"frame_id_{index}" for index in range(1, max_events + 1)]])
        for item in results[:100]:
            writer.writerow([item.video_id, *[frame.frame_id for frame in item.frames]])
    return destination
