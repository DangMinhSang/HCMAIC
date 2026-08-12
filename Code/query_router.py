"""Query-adaptive evidence routing for multimodal AIC retrieval.

Qwen scores the query against descriptions of each evidence source before
frame reranking. The resulting percentages control recall quotas and score
fusion. A deterministic bilingual lexical prior keeps routing useful when the
optional Qwen model is disabled or temporarily unavailable.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from query_analyzer import LightweightQueryAnalyzer, QueryAnalysis

MODALITIES = ("visual", "ocr", "metadata", "object")
MODALITY_DOCUMENTS: dict[str, str] = {
    "visual": (
        "Visual scene and action evidence: people, places, activities, events, camera view, "
        "and the overall appearance of a video frame."
    ),
    "ocr": (
        "Visible text and OCR evidence: exact words on signs, banners, subtitles, documents, "
        "screens, logos, labels, or license plates."
    ),
    "metadata": (
        "Video metadata evidence: program or video title, description, channel, named entity, "
        "date, topic, location, and keywords not necessarily visible in the frame."
    ),
    "object": (
        "Object-level evidence: specific objects, clothing, colors, counts, animals, vehicles, "
        "and other concrete visible attributes."
    ),
}
ROUTING_PROMPT = (
    "Estimate how useful this evidence source is for retrieving the user's requested video moment. "
    "Score each source independently; the application will normalize the scores into percentages."
)

EXPLICIT_OCR_CUES = (
    "noi dung", "dong chu", "chu viet", "ghi gi", "viet gi", "doc chu", "van ban",
    "bien bao", "bang hieu", "phu de", "tieu de tren", "tai lieu", "logo", "nhan hieu",
    "bien so", "man hinh hien", "text", "written", "word", "words", "says", "sign",
    "banner", "subtitle", "caption", "document", "license plate",
)
AMBIGUOUS_WARNING_CUES = (
    "canh bao", "nguy hiem", "sat lo", "lu quet", "warning", "dangerous", "landslide",
)
METADATA_CUES = (
    "ten video", "tieu de video", "ten chuong trinh", "chuong trinh nao", "kenh nao",
    "kenh truyen hinh", "ngay phat", "metadata", "video title", "program title", "channel",
    "broadcast date", "description", "keyword", "episode",
)
OBJECT_CUES = (
    "bao nhieu", "dem so", "mau gi", "mau nao", "mac ao", "cam vat", "vat gi",
    "do vat", "phuong tien", "con vat", "how many", "what color", "which color",
    "wearing", "holding", "object", "vehicle", "animal",
)
VISUAL_CUES = (
    "dang lam", "hanh dong", "khoanh khac", "canh", "di bo", "chay", "nhay", "ngoi",
    "dung", "noi chuyen", "phat bieu", "trao", "mo", "dong", "roi", "what happens",
    "action", "scene", "moment", "walking", "running", "jumping", "speaking", "opening",
)


def normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value.lower())
    plain = "".join(character for character in decomposed if unicodedata.category(character) != "Mn")
    return " ".join(plain.replace("đ", "d").split())


def has_explicit_ocr_intent(query: str) -> bool:
    """Whether the user explicitly asks for visible text evidence."""
    text = normalize_text(query)
    if any(cue in text for cue in EXPLICIT_OCR_CUES):
        return True
    if re.search(r"[\"“”'][^\"“”']{3,}[\"“”']", query):
        return True
    return bool(re.search(r"\b[A-ZÀ-ỸĐ][A-ZÀ-ỸĐ\s]{5,}\b", query))


def is_ambiguous_warning_query(query: str) -> bool:
    text = normalize_text(query)
    return (
        not has_explicit_ocr_intent(query)
        and sum(1 for cue in AMBIGUOUS_WARNING_CUES if cue in text) >= 1
    )


def normalize_distribution(values: Mapping[str, float], *, floor: float = 0.02) -> dict[str, float]:
    bounded = {name: max(floor, float(values.get(name, 0.0))) for name in MODALITIES}
    total = sum(bounded.values()) or 1.0
    return {name: bounded[name] / total for name in MODALITIES}


def model_distribution(scores: Mapping[str, float]) -> dict[str, float]:
    """Turn independent sigmoid scores into a moderately sharp distribution."""
    bounded = [max(1e-4, min(1.0, float(scores.get(name, 0.0)))) for name in MODALITIES]
    # Squaring preserves ordering while making four uniformly high sigmoid
    # outputs more useful as routing percentages.
    powered = [value * value for value in bounded]
    return normalize_distribution(dict(zip(MODALITIES, powered)))


def lexical_distribution(query: str) -> dict[str, float]:
    text = normalize_text(query)
    strengths = {"visual": 0.55, "ocr": 0.08, "metadata": 0.12, "object": 0.25}

    def cue_count(cues: Sequence[str]) -> int:
        return sum(1 for cue in cues if cue in text)

    ocr_matches = cue_count(EXPLICIT_OCR_CUES)
    warning_matches = cue_count(AMBIGUOUS_WARNING_CUES)
    metadata_matches = cue_count(METADATA_CUES)
    object_matches = cue_count(OBJECT_CUES)
    visual_matches = cue_count(VISUAL_CUES)
    strengths["ocr"] += min(3.2, 1.35 * ocr_matches)
    # Warning language helps OCR recall, but does not mean the user wants a
    # presentation slide containing those words instead of the real scene.
    strengths["ocr"] += min(0.55, 0.18 * warning_matches)
    strengths["visual"] += min(1.10, 0.32 * warning_matches)
    strengths["metadata"] += min(2.4, 1.20 * metadata_matches)
    strengths["object"] += min(2.6, 1.15 * object_matches)
    strengths["visual"] += min(2.0, 0.55 * visual_matches)
    # Quoted phrases and long all-caps fragments are usually intended as exact
    # visible text, even if the query does not explicitly say "OCR".
    if re.search(r"[\"“”'][^\"“”']{3,}[\"“”']", query):
        strengths["ocr"] += 1.4
    if re.search(r"\b[A-ZÀ-ỸĐ][A-ZÀ-ỸĐ\s]{5,}\b", query):
        strengths["ocr"] += 0.8
    return normalize_distribution(strengths)


@dataclass(frozen=True)
class QueryProfile:
    visual: float
    ocr: float
    metadata: float
    object: float
    source: str
    analysis: QueryAnalysis | None = None

    def as_dict(self) -> dict[str, float | str]:
        return {
            "visual": round(self.visual, 4),
            "ocr": round(self.ocr, 4),
            "metadata": round(self.metadata, 4),
            "object": round(self.object, 4),
            "source": self.source,
            "analysis_source": self.analysis.source if self.analysis else self.source,
            "analysis": self.analysis.summary() if self.analysis else "",
        }

    def values(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in MODALITIES}

    @property
    def metadata_weight(self) -> float:
        return min(0.30, 0.025 + 0.32 * self.metadata)

    def summary(self) -> str:
        return (
            f"Router {self.source}: hình {self.visual:.0%} · OCR {self.ocr:.0%} · "
            f"metadata {self.metadata:.0%} · object {self.object:.0%}"
        )

    def analysis_summary(self) -> str:
        return self.analysis.summary() if self.analysis else ""

    def support_score(
        self,
        *,
        visual: float,
        ocr: float,
        metadata: float,
        object_score: float,
    ) -> float:
        return (
            self.visual * max(0.0, min(1.0, visual))
            + self.ocr * max(0.0, min(1.0, ocr))
            + self.metadata * max(0.0, min(1.0, metadata))
            + self.object * max(0.0, min(1.0, object_score))
        )


def build_query_analysis(
    query: str,
    analyzer: LightweightQueryAnalyzer | None = None,
) -> QueryAnalysis:
    """Analyze a query once; the dashboard owns the long-lived analyzer."""
    return (analyzer or LightweightQueryAnalyzer()).analyze(query)


def build_query_profile(
    query: str,
    reranker: Any | None = None,
    *,
    analyzer: LightweightQueryAnalyzer | None = None,
    analysis: QueryAnalysis | None = None,
) -> QueryProfile:
    """Build four-source weights from the lightweight clause analyzer.

    ``reranker`` remains an accepted compatibility argument for callers from
    older dashboard processes, but Qwen is deliberately not used for routing:
    its image-capable cross-encoder is reserved for the small frame pool.
    """
    del reranker
    analysis = analysis or build_query_analysis(query, analyzer)
    return QueryProfile(source=analysis.source, analysis=analysis, **analysis.values())
