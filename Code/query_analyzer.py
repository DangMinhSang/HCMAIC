"""Lightweight multilingual query decomposition for AIC evidence routing.

The expensive Qwen reranker belongs on a small image candidate pool. It is a
poor fit for deciding, on every request, whether a phrase is visual, OCR,
metadata, or object evidence. This module uses one cached multilingual
MiniLM sentence encoder for that small text-only decision and keeps a
deterministic structural/lexical fallback when the checkpoint is unavailable.
"""

from __future__ import annotations

import math
import os
import re
import threading
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


MODALITIES = ("visual", "ocr", "metadata", "object")
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

MODALITY_DOCUMENTS: dict[str, str] = {
    "visual": (
        "Hình ảnh và ngữ cảnh cảnh quay: người, địa điểm, hoạt động, hành động, sự kiện, "
        "tư thế, chuyển động và khoảnh khắc nhìn thấy trong video. "
        "Visual scene, action, people, place, event and camera context."
    ),
    "ocr": (
        "Chữ nhìn thấy trong khung hình: nội dung trên biển báo, bảng hiệu, màn hình, tài liệu, "
        "phụ đề, banner, nhãn hoặc dòng chữ; tìm đúng từ được viết. "
        "Visible words, text, sign, label, document or caption content."
    ),
    "metadata": (
        "Thông tin mô tả video: tên video, tiêu đề chương trình, kênh, ngày, địa điểm, chủ đề, "
        "từ khóa và mô tả không nhất thiết xuất hiện trong hình. "
        "Video title, program, channel, date, location, topic and description metadata."
    ),
    "object": (
        "Đặc điểm vật thể nhìn thấy: số lượng người hoặc vật, màu sắc, quần áo, vật đang cầm, "
        "xe cộ, động vật và các đối tượng cụ thể. "
        "Visible object, count, color, clothing, held item, vehicle or animal."
    ),
}

OCR_CONNECTOR_RE = re.compile(
    r"\b(?:có\s+nội\s+dung(?:\s+là)?|nội\s+dung(?:\s+là)?|"
    r"có\s+chữ|chứa\s+(?:dòng\s+)?chữ|ghi(?:\s+là)?|viết(?:\s+là)?|"
    r"says?|with\s+(?:the\s+)?(?:text|words?)|containing\s+(?:the\s+)?(?:text|words?))\b",
    flags=re.IGNORECASE,
)
CLAUSE_SPLIT_RE = re.compile(r"[.!?;\n]+")
WORD_RE = re.compile(r"[\w]+", flags=re.UNICODE)

OCR_CUES = (
    "noi dung", "dong chu", "chu viet", "ghi gi", "viet gi", "doc chu", "van ban",
    "bien bao", "bang hieu", "phu de", "tieu de tren", "tai lieu", "logo", "nhan hieu",
    "bien so", "man hinh hien", "text", "written", "word", "words", "says", "sign",
    "banner", "subtitle", "caption", "document", "license plate",
)
METADATA_CUES = (
    "ten video", "tieu de video", "ten chuong trinh", "chuong trinh nao", "kenh nao",
    "kenh truyen hinh", "ngay phat", "metadata", "video title", "program title", "channel",
    "broadcast date", "description", "keyword", "episode", "dia diem", "location",
)
OBJECT_CUES = (
    "bao nhieu", "dem so", "mau gi", "mau nao", "mac ao", "cam vat", "vat gi",
    "do vat", "phuong tien", "con vat", "how many", "what color", "which color",
    "wearing", "holding", "object", "vehicle", "animal", "count", "so luong",
)
VISUAL_CUES = (
    "dang lam", "hanh dong", "khoanh khac", "canh", "di bo", "chay", "nhay", "ngoi",
    "dung", "noi chuyen", "phat bieu", "trao", "mo", "dong", "roi", "what happens",
    "action", "scene", "moment", "walking", "running", "jumping", "speaking", "opening",
)


def normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", str(value).lower())
    plain = "".join(character for character in decomposed if unicodedata.category(character) != "Mn")
    return " ".join(plain.replace("đ", "d").split())


@dataclass(frozen=True)
class QueryClause:
    text: str
    scores: Mapping[str, float]
    dominant: str
    confidence: float
    role_hint: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "scores": {name: round(float(self.scores[name]), 4) for name in MODALITIES},
            "dominant": self.dominant,
            "confidence": round(self.confidence, 4),
            "role_hint": self.role_hint,
        }


@dataclass(frozen=True)
class QueryAnalysis:
    original: str
    clauses: tuple[QueryClause, ...]
    weights: Mapping[str, float]
    source: str

    def values(self) -> dict[str, float]:
        return {name: float(self.weights.get(name, 0.0)) for name in MODALITIES}

    def query_for(self, modality: str) -> str:
        if modality not in MODALITIES:
            raise ValueError(f"Unknown query modality: {modality}")
        selected: list[str] = []
        for clause in self.clauses:
            scores = {name: float(clause.scores.get(name, 0.0)) for name in MODALITIES}
            if clause.role_hint:
                include = modality == clause.role_hint
            else:
                normalized = normalize_text(clause.text)
                visual_sign = "bien" in normalized and (
                    "mau" in normalized or "yellow" in normalized
                )
                if visual_sign:
                    # The object is a visual sign; OCR is reserved for a
                    # separate clause explicitly introduced as its content.
                    include = modality == "visual"
                    if include and clause.text not in selected:
                        selected.append(clause.text)
                    continue
                ambiguous_warning = any(
                    cue in normalized
                    for cue in ("canh bao", "nguy hiem", "sat lo", "lu quet", "warning", "landslide")
                )
                if ambiguous_warning and modality == "ocr":
                    # A short warning query may refer to a physical warning
                    # sign without saying "the text says". Keep it in OCR
                    # recall as a secondary source; visual scoring still
                    # decides whether the frame is a real scene or a slide.
                    include = True
                    if clause.text not in selected:
                        selected.append(clause.text)
                    continue
                maximum = max(scores.values(), default=0.0)
                # Keep a genuinely ambiguous phrase in both relevant pools;
                # a clear phrase belongs only to its strongest evidence source.
                include = scores[modality] >= 0.20 and scores[modality] >= maximum - 0.12
            if include and clause.text not in selected:
                selected.append(clause.text)
        return " ".join(selected)

    def source_queries(self) -> dict[str, str]:
        return {modality: self.query_for(modality) for modality in MODALITIES}

    def summary(self) -> str:
        pieces = [
            f"{clause.dominant.upper()}: {clause.text}"
            for clause in self.clauses
        ]
        return " | ".join(pieces)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "weights": {name: round(float(self.weights[name]), 4) for name in MODALITIES},
            "clauses": [clause.as_dict() for clause in self.clauses],
        }


@dataclass(frozen=True)
class _ClauseSeed:
    text: str
    role_hint: str = ""


def split_query_clauses(value: str) -> list[_ClauseSeed]:
    """Split text clauses while preserving the OCR connector's right side."""
    raw = " ".join(str(value or "").split()).strip()
    if not raw:
        raise ValueError("Nhập truy vấn.")
    seeds: list[_ClauseSeed] = []
    for sentence in CLAUSE_SPLIT_RE.split(raw):
        sentence = sentence.strip(" ,:-—–")
        if not sentence:
            continue
        cursor = 0
        found_connector = False
        for match in OCR_CONNECTOR_RE.finditer(sentence):
            left = sentence[cursor : match.start()].strip(" ,:-—–")
            if left:
                seeds.append(_ClauseSeed(left))
            cursor = match.end()
            found_connector = True
        right = sentence[cursor:].strip(" ,:-—–")
        if right:
            seeds.append(_ClauseSeed(right, "ocr" if found_connector else ""))
    return seeds or [_ClauseSeed(raw)]


def _cue_count(text: str, cues: Sequence[str]) -> int:
    return sum(1 for cue in cues if cue in text)


def lexical_clause_scores(text: str, role_hint: str = "") -> dict[str, float]:
    normalized = normalize_text(text)
    scores = {name: 0.08 for name in MODALITIES}
    scores["visual"] += 0.45
    scores["object"] += 0.18
    scores["ocr"] += min(1.6, 0.80 * _cue_count(normalized, OCR_CUES))
    scores["metadata"] += min(1.8, 0.90 * _cue_count(normalized, METADATA_CUES))
    scores["object"] += min(1.8, 0.90 * _cue_count(normalized, OBJECT_CUES))
    scores["visual"] += min(1.4, 0.45 * _cue_count(normalized, VISUAL_CUES))
    # A physical sign plus a color describes what the frame looks like; it is
    # not automatically an OCR request merely because the sign says warning.
    if "bien" in normalized and ("mau" in normalized or "yellow" in normalized):
        scores["visual"] += 1.0
        scores["ocr"] *= 0.55
    if role_hint in MODALITIES:
        scores[role_hint] += 2.4
    return scores


def _softmax(values: Sequence[float], temperature: float = 0.18) -> list[float]:
    if not values:
        return []
    scale = max(0.02, float(temperature))
    maximum = max(values)
    exponentials = [math.exp((value - maximum) / scale) for value in values]
    total = sum(exponentials) or 1.0
    return [value / total for value in exponentials]


def _normalize_scores(values: Mapping[str, float], floor: float = 0.02) -> dict[str, float]:
    bounded = {name: max(floor, float(values.get(name, 0.0))) for name in MODALITIES}
    total = sum(bounded.values()) or 1.0
    return {name: bounded[name] / total for name in MODALITIES}


class LightweightQueryAnalyzer:
    """Cached MiniLM query analyzer with a deterministic offline fallback."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.environ.get("AIC_QUERY_ANALYZER_MODEL", DEFAULT_MODEL)
        self.device = os.environ.get("AIC_QUERY_ANALYZER_DEVICE", "cpu")
        self._model: Any | None = None
        self._prototype_embeddings: Any | None = None
        self._lock = threading.RLock()
        self._cache: OrderedDict[str, QueryAnalysis] = OrderedDict()
        try:
            self._cache_size = max(0, min(int(os.environ.get("AIC_QUERY_ANALYZER_CACHE", "512")), 4096))
        except ValueError:
            self._cache_size = 512
        self.load_error = ""

    @property
    def ready(self) -> bool:
        return self._model is not None and self._prototype_embeddings is not None

    def _load(self) -> None:
        if self.ready:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as error:
            raise RuntimeError(
                "Thiếu sentence-transformers cho query analyzer nhẹ; dùng lexical fallback."
            ) from error
        device = self.device
        if device == "auto":
            try:
                import torch  # type: ignore

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self._model = SentenceTransformer(self.model_name, device=device)
        self.device = device
        self._prototype_embeddings = self._model.encode(
            [MODALITY_DOCUMENTS[name] for name in MODALITIES],
            batch_size=4,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def _model_scores(self, clauses: Sequence[_ClauseSeed]) -> list[dict[str, float]]:
        self._load()
        embeddings = self._model.encode(
            [clause.text for clause in clauses],
            batch_size=max(1, min(16, len(clauses))),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        scores: list[dict[str, float]] = []
        for vector in embeddings:
            similarities = [float(value) for value in vector @ self._prototype_embeddings.T]
            probabilities = _softmax(similarities)
            scores.append(dict(zip(MODALITIES, probabilities)))
        return scores

    @staticmethod
    def _combine_clause_scores(
        model_scores: Mapping[str, float],
        lexical_scores: Mapping[str, float],
        role_hint: str,
    ) -> dict[str, float]:
        lexical = _normalize_scores(lexical_scores)
        combined = {
            name: 0.68 * float(model_scores.get(name, 0.0)) + 0.32 * lexical[name]
            for name in MODALITIES
        }
        if role_hint in MODALITIES:
            # Structural connectors are high-precision supervision: the text
            # after "có nội dung là" is OCR, not a visual scene description.
            for name in MODALITIES:
                combined[name] *= 0.08 if name != role_hint else 1.0
        return _normalize_scores(combined)

    @staticmethod
    def _dominant(text: str, scores: Mapping[str, float], role_hint: str) -> str:
        if role_hint in MODALITIES:
            return role_hint
        normalized = normalize_text(text)
        if "bien" in normalized and ("mau" in normalized or "yellow" in normalized):
            return "visual"
        return max(MODALITIES, key=lambda name: float(scores.get(name, 0.0)))

    def _analyze_uncached(self, query: str) -> QueryAnalysis:
        seeds = split_query_clauses(query)
        try:
            model_scores = self._model_scores(seeds)
            source = f"MiniLM:{self.model_name.rsplit('/', 1)[-1]}+structure"
        except Exception as error:
            self.load_error = f"{type(error).__name__}: {error}"
            model_scores = [
                _normalize_scores(lexical_clause_scores(seed.text, seed.role_hint))
                for seed in seeds
            ]
            source = "lexical+structure fallback"

        clauses: list[QueryClause] = []
        for seed, semantic in zip(seeds, model_scores):
            scores = self._combine_clause_scores(
                semantic,
                lexical_clause_scores(seed.text, seed.role_hint),
                seed.role_hint,
            )
            dominant = self._dominant(seed.text, scores, seed.role_hint)
            ordered = sorted((float(scores[name]) for name in MODALITIES), reverse=True)
            confidence = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
            clauses.append(QueryClause(seed.text, scores, dominant, confidence, seed.role_hint))

        total_tokens = sum(max(1, len(WORD_RE.findall(clause.text))) for clause in clauses) or 1
        aggregate = {name: 0.0 for name in MODALITIES}
        for clause in clauses:
            mass = max(1, len(WORD_RE.findall(clause.text))) / total_tokens
            for name in MODALITIES:
                aggregate[name] += mass * float(clause.scores[name])
        return QueryAnalysis(query.strip(), tuple(clauses), _normalize_scores(aggregate), source)

    def analyze(self, query: str) -> QueryAnalysis:
        query = (query or "").strip()
        if not query:
            raise ValueError("Nhập truy vấn.")
        with self._lock:
            cached = self._cache.get(query)
            if cached is not None:
                self._cache.move_to_end(query)
                return cached
            analysis = self._analyze_uncached(query)
            if self._cache_size:
                self._cache[query] = analysis
                self._cache.move_to_end(query)
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
            return analysis

    def warmup(self) -> QueryAnalysis:
        return self.analyze("Biển cảnh báo màu vàng có nội dung là cảnh báo sạt lở nguy hiểm")
