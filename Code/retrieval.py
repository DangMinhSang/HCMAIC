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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from clip_encoder import ClipTextEncoder
from data_paths import AICPaths
from progress import track


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
    object_score: float = 0.0
    retrieval_score: float = 0.0
    title: str = ""
    object_labels: tuple[str, ...] = ()
    image_path: str | None = None
    video_path: str | None = None
    answer: str = ""
    qa_confidence: float = 0.0
    ocr_score: float = 0.0
    ocr_quality: float = 1.0
    ocr_text: str = ""
    rerank_score: float = 0.0
    rerank_joint_score: float = 0.0
    rerank_visual_score: float = 0.0
    rerank_ocr_score: float = 0.0

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
        for event_index, result in track(
            enumerate(self.frames, start=1),
            desc="Dựng bảng TRAKE",
            total=len(self.frames),
            unit="event",
            nested=True,
        ):
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
        feature_paths = sorted(paths.features_dir.glob("*.npy"))
        self._features: dict[str, Path] = {}
        for feature in track(
            feature_paths,
            desc="Kiểm tra CLIP feature",
            unit="video",
            leave=True,
        ):
            if (paths.mapping_dir / f"{feature.stem}.csv").is_file():
                self._features[feature.stem] = feature
        if not self._features:
            raise FileNotFoundError(
                f"Không tìm thấy cặp .npy/.csv trong {paths.features_dir} và {paths.mapping_dir}"
            )
        self._mapping_cache: dict[str, list[FrameMapping]] = {}
        self._mapping_lookup: dict[str, dict[int, FrameMapping]] = {}
        self._mapping_positions: dict[str, dict[int, int]] = {}
        self._metadata_cache: dict[str, VideoMetadata] = {}
        self._metadata_tokens: dict[str, Counter[str]] = {}
        self._metadata_document_lengths: dict[str, int] = {}
        self._metadata_average_length = 1.0
        self._metadata_idf: dict[str, float] = {}
        self._metadata_ready = False
        self._vector_count: int | None = None
        self._ram_features: np.ndarray | None = None
        self._ram_offsets: np.ndarray | None = None
        self._ram_video_ids: tuple[str, ...] = ()
        self._ram_video_positions: dict[str, int] = {}
        self._keyframe_dirs: dict[str, Path | None] = {}
        self._video_path_cache: dict[str, Path | None] = {}
        self._object_label_cache: dict[tuple[str, int], tuple[str, ...]] = {}
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
            self._vector_count = 0
            for path in track(
                self._features.values(),
                desc="Đếm CLIP vectors",
                total=len(self._features),
                unit="video",
                leave=True,
            ):
                self._vector_count += int(np.load(path, mmap_mode="r").shape[0])
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
        shapes = []
        for video_id in track(
            video_ids,
            desc="Đọc CLIP headers",
            unit="video",
            leave=True,
        ):
            shapes.append(np.load(self._features[video_id], mmap_mode="r").shape)
        if not shapes or any(len(shape) != 2 or shape[1] != 512 for shape in shapes):
            raise ValueError("Không thể preload feature: cần các mảng CLIP 512 chiều.")
        offsets = np.zeros(len(video_ids) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([shape[0] for shape in shapes])
        matrix = np.empty((int(offsets[-1]), 512), dtype=np.float32)
        for index, video_id in track(
            enumerate(video_ids),
            desc="Nạp CLIP vào RAM",
            total=len(video_ids),
            unit="video",
            force=True,
            leave=True,
        ):
            values = np.asarray(np.load(self._features[video_id]), dtype=np.float32)
            values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
            matrix[offsets[index] : offsets[index + 1]] = values
        self._ram_features = matrix
        self._ram_offsets = offsets
        self._ram_video_ids = video_ids
        self._ram_video_positions = {video_id: index for index, video_id in enumerate(video_ids)}

    def prepare_runtime(self) -> None:
        """Preload query-invariant metadata and warm models before serving."""
        if self._ram_features is None and os.environ.get("AIC_PRELOAD_FEATURES", "0").lower() in {
            "1",
            "true",
            "yes",
        }:
            self.preload_features()
        for video_id in track(
            self._features,
            desc="Preload mapping/metadata",
            total=len(self._features),
            unit="video",
            force=True,
            leave=True,
        ):
            self._mapping(video_id)
            self._metadata(video_id)
            self._keyframe_dir(video_id)
        self._prepare_metadata()
        self.encoder.warmup()

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
                video_position = self._ram_video_positions[video_id]
            except KeyError:
                return []
            start, end = self._ram_offsets[video_position : video_position + 2]
            scores = self._ram_features[start:end] @ query_vector
            indices = self._top_indices(scores, min(len(scores), candidate_pool * 8))
            return [_Candidate(video_id, int(index), float(scores[index])) for index in indices]

        scores = self._ram_features @ query_vector
        # Candidate diversification happens later. Avoid materializing tens of
        # thousands of frames only to discard them before the reranker.
        raw_limit = candidate_pool if allowed_video_ids is None else candidate_pool * 3
        indices = self._top_indices(scores, min(len(scores), raw_limit))
        output: list[_Candidate] = []
        for index in track(
            indices,
            desc="Ánh xạ CLIP candidates",
            total=len(indices),
            unit="frame",
        ):
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
            reader = csv.DictReader(stream)
            for row in track(
                reader,
                desc=f"Đọc mapping {video_id}",
                unit="frame",
                nested=True,
            ):
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
        self._mapping_lookup[video_id] = {item.keyframe_number: item for item in mappings}
        self._mapping_positions[video_id] = {
            item.keyframe_number: index for index, item in enumerate(mappings)
        }
        return mappings

    def _keyframe_dir(self, video_id: str) -> Path | None:
        if video_id not in self._keyframe_dirs:
            self._keyframe_dirs[video_id] = next(
                (
                    root / "keyframes" / video_id
                    for root in self.paths.keyframe_roots
                    if (root / "keyframes" / video_id).is_dir()
                ),
                None,
            )
        return self._keyframe_dirs[video_id]

    def _image_path(self, video_id: str, keyframe_number: int) -> Path | None:
        directory = self._keyframe_dir(video_id)
        if directory is None:
            return None
        candidate = directory / f"{keyframe_number:03d}.jpg"
        return candidate if candidate.is_file() else None

    def _video_path(self, video_id: str) -> Path | None:
        if video_id not in self._video_path_cache:
            self._video_path_cache[video_id] = self.paths.video_path(video_id)
        return self._video_path_cache[video_id]

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
        document_frequency: Counter[str] = Counter()
        for video_id in track(
            self._features,
            desc="Lập metadata BM25",
            total=len(self._features),
            unit="video",
            force=True,
            leave=True,
        ):
            terms = Counter(tokenize(self._metadata(video_id).searchable_text))
            self._metadata_tokens[video_id] = terms
            self._metadata_document_lengths[video_id] = sum(terms.values())
            document_frequency.update(terms.keys())
        lengths = list(self._metadata_document_lengths.values())
        self._metadata_average_length = sum(lengths) / len(lengths) if lengths else 1.0
        total = max(1, len(self._metadata_tokens))
        self._metadata_idf = {}
        for term, frequency in track(
            document_frequency.items(),
            desc="Tính metadata IDF",
            total=len(document_frequency),
            unit="term",
            leave=True,
        ):
            self._metadata_idf[term] = math.log(
                1.0 + (total - frequency + 0.5) / (frequency + 0.5)
            )
        self._metadata_ready = True

    def _metadata_scores(self, query: str, video_ids: Iterable[str]) -> dict[str, float]:
        """Normalized BM25 for names, dates and channels missing from CLIP."""
        tokens = [token for token in tokenize(query) if len(token) > 1]
        unique_video_ids = set(video_ids)
        if not tokens or self.paths.metadata_dir is None:
            return {video_id: 0.0 for video_id in unique_video_ids}
        self._prepare_metadata()
        query_counts = Counter(tokens)
        raw_scores: dict[str, float] = {}
        k1, b = 1.2, 0.75
        for video_id in track(
            unique_video_ids,
            desc="Chấm metadata BM25",
            total=len(unique_video_ids),
            unit="video",
        ):
            document = self._metadata_tokens.get(video_id, Counter())
            if not document:
                raw_scores[video_id] = 0.0
                continue
            document_length = self._metadata_document_lengths.get(video_id, 0)
            normalizer = 1.0 - b + b * document_length / max(self._metadata_average_length, 1.0)
            score = 0.0
            matched = 0
            for token, query_frequency in track(
                query_counts.items(),
                desc="Metadata query terms",
                total=len(query_counts),
                unit="term",
                nested=True,
            ):
                frequency = document.get(token, 0)
                if not frequency:
                    continue
                matched += 1
                term_frequency = frequency * (k1 + 1.0) / (frequency + k1 * normalizer)
                score += self._metadata_idf.get(token, 0.0) * term_frequency * min(query_frequency, 2)
            if matched:
                coverage = matched / len(query_counts)
                score *= 1.0 + 0.45 * coverage
            raw_scores[video_id] = score
        maximum = max(raw_scores.values(), default=0.0)
        if maximum <= 0.0:
            return {video_id: 0.0 for video_id in unique_video_ids}
        return {video_id: score / maximum for video_id, score in raw_scores.items()}

    def _object_labels(self, video_id: str, keyframe_number: int) -> tuple[str, ...]:
        cache_key = (video_id, keyframe_number)
        cached = self._object_label_cache.get(cache_key)
        if cached is not None:
            return cached
        path = self.paths.object_path(video_id, keyframe_number)
        if path is None:
            self._object_label_cache[cache_key] = ()
            return ()
        try:
            with path.open(encoding="utf-8") as stream:
                payload = json.load(stream)
        except (OSError, json.JSONDecodeError):
            self._object_label_cache[cache_key] = ()
            return ()

        labels: list[str] = []

        def visit(node: Any, key: str = "") -> None:
            if isinstance(node, dict):
                for child_key, child in track(
                    node.items(),
                    desc="Đọc object fields",
                    total=len(node),
                    unit="field",
                    nested=True,
                ):
                    visit(child, child_key.lower())
            elif isinstance(node, list):
                for child in track(
                    node,
                    desc="Đọc object list",
                    total=len(node),
                    unit="item",
                    nested=True,
                ):
                    visit(child, key)
            elif key in LABEL_KEYS and isinstance(node, str) and node.strip():
                labels.append(node.strip())

        visit(payload)
        output = tuple(dict.fromkeys(labels))
        self._object_label_cache[cache_key] = output
        return output

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
        image = self._image_path(candidate.video_id, frame.keyframe_number)
        video = self._video_path(candidate.video_id)
        return SearchResult(
            rank=0,
            video_id=candidate.video_id,
            frame_id=frame.frame_id,
            keyframe_number=frame.keyframe_number,
            pts_time=frame.pts_time,
            visual_score=candidate.visual_score,
            metadata_score=0.0,
            score=candidate.visual_score,
            retrieval_score=candidate.visual_score,
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
        ocr_quality: float = 1.0,
        ocr_text: str = "",
    ) -> SearchResult | None:
        """Materialize an OCR hit with the official keyframe-to-frame mapping."""
        self._mapping(video_id)
        frame = self._mapping_lookup[video_id].get(keyframe_number)
        if frame is None:
            return None
        metadata = self._metadata(video_id)
        image = self._image_path(video_id, keyframe_number)
        video = self._video_path(video_id)
        return SearchResult(
            rank=0,
            video_id=video_id,
            frame_id=frame.frame_id,
            keyframe_number=keyframe_number,
            pts_time=frame.pts_time,
            visual_score=0.0,
            metadata_score=0.0,
            score=score,
            retrieval_score=score,
            title=metadata.title,
            image_path=str(image) if image is not None else None,
            video_path=str(video) if video is not None else None,
            ocr_score=ocr_score,
            ocr_quality=ocr_quality,
            ocr_text=ocr_text,
        )

    def neighboring_keyframes(
        self,
        video_id: str,
        keyframe_number: int,
        *,
        radius: int = 1,
        query_vector: np.ndarray | None = None,
    ) -> list[tuple[int, SearchResult]]:
        """Materialize a small temporal neighborhood for exact-frame refinement."""
        mapping = self._mapping(video_id)
        center = self._mapping_positions[video_id].get(keyframe_number)
        if center is None:
            return []
        radius = max(0, min(int(radius), 5))
        output: list[tuple[int, SearchResult]] = []
        neighbor_indices = range(max(0, center - radius), min(len(mapping), center + radius + 1))
        for feature_index in track(
            neighbor_indices,
            desc="TRAKE lân cận",
            total=len(neighbor_indices),
            unit="frame",
        ):
            visual_score = 0.0
            if query_vector is not None:
                if self._ram_features is not None and self._ram_offsets is not None:
                    video_position = self._ram_video_positions[video_id]
                    vector = self._ram_features[int(self._ram_offsets[video_position]) + feature_index]
                    visual_score = float(vector @ query_vector)
                else:
                    features = np.load(self._features[video_id], mmap_mode="r")
                    vector = np.asarray(features[feature_index], dtype=np.float32)
                    visual_score = float(vector @ query_vector) / max(float(np.linalg.norm(vector)), 1e-12)
            result = self._candidate_to_result(_Candidate(video_id, feature_index, visual_score))
            if result is not None:
                output.append((feature_index, result))
        return output

    def _raw_candidates(
        self,
        query_vector: np.ndarray,
        top_k: int,
        allowed_video_ids: set[str] | None = None,
    ) -> list[_Candidate]:
        candidate_pool = max(800, top_k * 30)
        if self._ram_features is not None:
            return self._ram_candidates(query_vector, candidate_pool, allowed_video_ids)
        per_video_limit = max(30, min(100, candidate_pool // 8))
        all_candidates: list[_Candidate] = []
        for video_id, feature_path in track(
            self._features.items(),
            desc="CLIP corpus recall",
            total=len(self._features),
            unit="video",
            force=True,
        ):
            if allowed_video_ids is not None and video_id not in allowed_video_ids:
                continue
            features = np.load(feature_path, mmap_mode="r")
            if features.ndim != 2 or features.shape[1] != query_vector.shape[0]:
                raise ValueError(
                    f"Feature {feature_path.name} có shape {features.shape}, không khớp "
                    f"CLIP ViT-B/32 ({query_vector.shape[0]} chiều)."
                )
            scores = self._cosine_scores(features, query_vector)
            top_indices = self._top_indices(scores, per_video_limit)
            for index in track(
                top_indices,
                desc="CLIP top/video",
                total=len(top_indices),
                unit="frame",
                nested=True,
            ):
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

        for result in track(
            results,
            desc="Fuse CLIP/metadata",
            total=len(results),
            unit="frame",
        ):
            result.metadata_score = metadata_scores.get(result.video_id, 0.0)
            result.score = (
                (1.0 - metadata_weight) * result.visual_score
                + metadata_weight * result.metadata_score
            )
            result.retrieval_score = result.score
        results.sort(key=lambda item: item.score, reverse=True)

        try:
            object_budget = max(0, min(int(os.environ.get("AIC_OBJECT_RERANK_CANDIDATES", "300")), 600))
        except ValueError:
            object_budget = 300
        object_candidates = results[:object_budget]
        for result in track(
            object_candidates,
            desc="Đọc object labels",
            total=len(object_candidates),
            unit="frame",
        ):
            result.object_labels = self._object_labels(result.video_id, result.keyframe_number)
            result.object_score = self._object_match_score(result.object_labels, query_tokens)
            # Faster R-CNN labels are a tie-breaker, never the primary signal.
            result.score += 0.03 * result.object_score
            result.retrieval_score = result.score
        results.sort(key=lambda item: item.score, reverse=True)

        chosen: list[SearchResult] = []
        frames_by_video: dict[str, list[int]] = {}
        for result in track(
            results,
            desc="Chọn kết quả đa dạng",
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
        event_vectors = []
        for index, event in track(
            enumerate(clean_events),
            desc="Mã hóa TRAKE events",
            total=len(clean_events),
            unit="event",
            force=True,
        ):
            event_vectors.append(
                self.encoder.encode(event, english_events[index] if index < len(english_events) else "")
            )
        vectors = np.stack(event_vectors)

        global_scores = (
            np.asarray(self._ram_features @ vectors.T, dtype=np.float32)
            if self._ram_features is not None and self._ram_offsets is not None
            else None
        )
        video_candidates: list[tuple[float, str, np.ndarray, np.ndarray]] = []
        for video_id, feature_path in track(
            self._features.items(),
            desc="TRAKE scan videos",
            total=len(self._features),
            unit="video",
            force=True,
        ):
            if global_scores is not None and self._ram_offsets is not None:
                position = self._ram_video_positions[video_id]
                start, end = self._ram_offsets[position : position + 2]
                scores = global_scores[start:end]
            else:
                features = np.load(feature_path, mmap_mode="r")
                if features.ndim != 2 or features.shape[1] != vectors.shape[1]:
                    continue
                scores = np.asarray(features @ vectors.T, dtype=np.float32)
                scores /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
            alignment = self._ordered_alignment(scores)
            if alignment is None:
                continue
            best_indices, best_scores = alignment
            # The mean rewards overall fit; the minimum prevents a video with
            # one missing stage from winning merely because other stages match.
            sequence_score = 0.72 * float(best_scores.mean()) + 0.28 * float(best_scores.min())
            video_candidates.append((sequence_score, video_id, best_indices, best_scores))

        best_videos = heapq.nlargest(top_videos, video_candidates, key=lambda item: item[0])
        output: list[TrakeVideoResult] = []
        for rank, (score, video_id, indices, scores) in track(
            enumerate(best_videos, start=1),
            desc="Dựng TRAKE sequences",
            total=len(best_videos),
            unit="video",
        ):
            aligned: list[SearchResult] = []
            for feature_index, event_score in track(
                zip(indices, scores),
                desc=f"TRAKE {video_id}",
                total=len(indices),
                unit="event",
                nested=True,
            ):
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

    @staticmethod
    def _ordered_alignment(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """Viterbi alignment with one strictly increasing frame per event."""
        if scores.ndim != 2:
            return None
        frame_count, event_count = scores.shape
        if frame_count < event_count or event_count == 0:
            return None
        dp = scores[:, 0].astype(np.float32, copy=True)
        back = np.full((event_count, frame_count), -1, dtype=np.int32)
        negative_infinity = np.float32(-1e30)
        event_indices = range(1, event_count)
        for event_index in track(
            event_indices,
            desc="TRAKE Viterbi events",
            total=len(event_indices),
            unit="event",
            nested=True,
        ):
            previous_best = np.full(frame_count, negative_infinity, dtype=np.float32)
            previous_index = np.full(frame_count, -1, dtype=np.int32)
            running_score = negative_infinity
            running_index = -1
            frame_indices = range(frame_count)
            for frame_index in track(
                frame_indices,
                desc="TRAKE Viterbi frames",
                total=len(frame_indices),
                unit="frame",
                nested=True,
            ):
                candidate_index = frame_index - 1
                if candidate_index >= 0 and dp[candidate_index] > running_score:
                    running_score = dp[candidate_index]
                    running_index = candidate_index
                previous_best[frame_index] = running_score
                previous_index[frame_index] = running_index
            dp = scores[:, event_index] + previous_best
            back[event_index] = previous_index
        final_index = int(np.argmax(dp))
        if dp[final_index] <= negative_infinity / 2:
            return None
        indices = np.empty(event_count, dtype=np.int32)
        indices[-1] = final_index
        backtrack_indices = range(event_count - 1, 0, -1)
        for event_index in track(
            backtrack_indices,
            desc="TRAKE Viterbi backtrack",
            total=len(backtrack_indices),
            unit="event",
            nested=True,
        ):
            indices[event_index - 1] = back[event_index, indices[event_index]]
        if np.any(indices < 0):
            return None
        event_scores = scores[indices, np.arange(event_count)]
        return indices, event_scores

    @staticmethod
    def _ordered_candidate_alignment(
        frame_indices: Sequence[Sequence[int]],
        candidate_scores: Sequence[Sequence[float]],
    ) -> list[int] | None:
        """Choose one candidate per event while preserving strict chronology."""
        if not frame_indices or len(frame_indices) != len(candidate_scores):
            return None
        if any(not indices for indices in frame_indices):
            return None
        previous_scores = np.asarray(candidate_scores[0], dtype=np.float32)
        if len(previous_scores) != len(frame_indices[0]):
            return None
        back_pointers: list[np.ndarray] = []
        event_indices = range(1, len(frame_indices))
        for event_index in track(
            event_indices,
            desc="TRAKE refine events",
            total=len(event_indices),
            unit="event",
            nested=True,
        ):
            current_indices = frame_indices[event_index]
            current_values = np.asarray(candidate_scores[event_index], dtype=np.float32)
            if len(current_values) != len(current_indices):
                return None
            previous_indices = frame_indices[event_index - 1]
            current_dp = np.full(len(current_indices), -np.inf, dtype=np.float32)
            pointers = np.full(len(current_indices), -1, dtype=np.int32)
            for current_position, frame_index in track(
                enumerate(current_indices),
                desc="TRAKE refine candidates",
                total=len(current_indices),
                unit="frame",
                nested=True,
            ):
                valid = [
                    position
                    for position, previous_frame in enumerate(previous_indices)
                    if previous_frame < frame_index and np.isfinite(previous_scores[position])
                ]
                if not valid:
                    continue
                best_previous = max(valid, key=lambda position: float(previous_scores[position]))
                current_dp[current_position] = (
                    previous_scores[best_previous] + current_values[current_position]
                )
                pointers[current_position] = best_previous
            previous_scores = current_dp
            back_pointers.append(pointers)
        if not np.isfinite(previous_scores).any():
            return None
        choices = [0] * len(frame_indices)
        choices[-1] = int(np.nanargmax(previous_scores))
        backtrack_indices = range(len(frame_indices) - 1, 0, -1)
        for event_index in track(
            backtrack_indices,
            desc="TRAKE refine backtrack",
            total=len(backtrack_indices),
            unit="event",
            nested=True,
        ):
            choices[event_index - 1] = int(back_pointers[event_index - 1][choices[event_index]])
            if choices[event_index - 1] < 0:
                return None
        return choices


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
        submission_results = results[:100]
        for result in track(
            submission_results,
            desc="Ghi KIS/Q&A CSV",
            total=len(submission_results),
            unit="row",
        ):
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
        submission_results = results[:100]
        for item in track(
            submission_results,
            desc="Ghi TRAKE CSV",
            total=len(submission_results),
            unit="row",
        ):
            writer.writerow([item.video_id, *[frame.frame_id for frame in item.frames]])
    return destination
