"""Persistent OCR index with in-memory lexical retrieval for AIC keyframes.

The OCR builder reads mounted keyframes once and writes only recognized text
to an index file. The dashboard loads that small index into RAM; no AIC image,
feature, or video file is copied or downloaded.
"""

from __future__ import annotations

import gzip
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from retrieval import normalize_text, tokenize


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "cua", "cho", "co", "da", "de", "do", "duoc",
    "for", "from", "in", "is", "la", "mot", "nhung", "o", "of", "on", "or", "the", "to",
    "trong", "va", "voi", "ve", "with",
}

# OCR on the AIC broadcasts is predominantly Vietnamese, while KIS queries
# may be Vietnamese or English. These short, high-value sign aliases make an
# English query such as "dangerous landslide warning" match Vietnamese OCR
# text without a network translation on the hot query path. They are kept
# deliberately small so generic words do not create noisy OCR matches.
SIGN_TERM_ALIASES: dict[str, tuple[str, ...]] = {
    "warning": ("canh", "bao"),
    "warnings": ("canh", "bao"),
    "danger": ("nguy", "hiem"),
    "dangerous": ("nguy", "hiem"),
    "landslide": ("sat", "lo"),
    "landslides": ("sat", "lo"),
    "flood": ("lu",),
    "flooding": ("lu",),
    "yellow": ("vang",),
    "stop": ("dung",),
    "prohibited": ("cam",),
    "exit": ("loi", "ra"),
    "entrance": ("loi", "vao"),
    "canh": ("warning",),
    "bao": ("warning",),
    "nguy": ("danger", "dangerous"),
    "hiem": ("danger", "dangerous"),
    "sat": ("landslide",),
    "lo": ("landslide",),
    "lu": ("flood", "flooding"),
    "vang": ("yellow",),
    "dung": ("stop",),
    "cam": ("prohibited",),
}


@dataclass(frozen=True)
class OCRRecord:
    video_id: str
    keyframe_number: int
    text: str


@dataclass(frozen=True)
class OCRHit:
    video_id: str
    keyframe_number: int
    text: str
    score: float
    matched_terms: int


def _open_index(path: Path, mode: str):
    return gzip.open(path, mode, encoding="utf-8") if path.suffix == ".gz" else path.open(mode, encoding="utf-8")


class OCRMemoryIndex:
    """An inverted OCR index whose query path only touches RAM."""

    def __init__(self, records: Iterable[OCRRecord]) -> None:
        self.records = list(records)
        self._terms: list[Counter[str]] = []
        self._postings: dict[str, list[int]] = defaultdict(list)
        self._normalized_text: list[str] = []
        for index, record in enumerate(self.records):
            terms = Counter(token for token in tokenize(record.text) if len(token) > 1 and token not in STOP_WORDS)
            self._terms.append(terms)
            self._normalized_text.append(normalize_text(record.text))
            for term in terms:
                self._postings[term].append(index)

    @property
    def record_count(self) -> int:
        return len(self.records)

    @classmethod
    def load(cls, path: str | Path) -> "OCRMemoryIndex":
        index_path = Path(path)
        if not index_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy OCR index: {index_path}")
        records: list[OCRRecord] = []
        with _open_index(index_path, "rt") as stream:
            for line in stream:
                try:
                    payload = json.loads(line)
                    text = str(payload.get("text") or "").strip()
                    if text:
                        records.append(
                            OCRRecord(
                                video_id=str(payload["video_id"]),
                                keyframe_number=int(payload["keyframe_number"]),
                                text=text,
                            )
                        )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
        return cls(records)

    def search(
        self,
        query: str,
        *,
        limit: int = 200,
        video_id: str | None = None,
    ) -> list[OCRHit]:
        """BM25-like OCR search; runtime is proportional to matching postings.

        The aliases make common Vietnamese/English warning-sign language
        interoperable entirely in RAM; no OCR or online translation runs here.
        """
        source_terms = [term for term in tokenize(query) if len(term) > 1 and term not in STOP_WORDS]
        terms = [
            expanded
            for term in source_terms
            for expanded in (term, *SIGN_TERM_ALIASES.get(term, ()))
        ]
        if not terms or not self.records:
            return []
        query_terms = Counter(terms)
        candidate_indices: set[int] = set()
        for term in query_terms:
            candidate_indices.update(self._postings.get(term, ()))
        if not candidate_indices:
            return []

        total = len(self.records)
        scored: list[tuple[float, int, int]] = []
        for index in candidate_indices:
            record = self.records[index]
            if video_id and record.video_id != video_id:
                continue
            doc_terms = self._terms[index]
            shared = sum(1 for term in query_terms if term in doc_terms)
            # Require two distinctive terms for long descriptions. This avoids
            # a random subtitle containing only "cảnh" outranking a real sign.
            required = 1 if len(query_terms) <= 2 else 2
            if shared < required:
                continue
            score = 0.0
            for term, query_count in query_terms.items():
                frequency = doc_terms.get(term, 0)
                if not frequency:
                    continue
                idf = math.log((total + 1) / (len(self._postings[term]) + 1)) + 1.0
                score += idf * (1.0 + math.log(frequency)) * min(query_count, frequency)
            # A contiguous phrase is very strong evidence for sign text.
            normalized_query = " ".join(query_terms)
            if len(normalized_query) >= 6 and normalized_query in self._normalized_text[index]:
                score *= 1.75
            scored.append((score, shared, index))
        if not scored:
            return []
        scored.sort(reverse=True)
        maximum = scored[0][0] or 1.0
        return [
            OCRHit(
                video_id=self.records[index].video_id,
                keyframe_number=self.records[index].keyframe_number,
                text=self.records[index].text,
                score=score / maximum,
                matched_terms=shared,
            )
            for score, shared, index in scored[:limit]
        ]
