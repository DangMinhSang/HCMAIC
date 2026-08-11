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

from progress import track
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
        self._document_lengths: list[int] = []
        for index, record in track(
            enumerate(self.records),
            desc="Lập OCR postings",
            total=len(self.records),
            unit="record",
            force=True,
            leave=True,
        ):
            terms = Counter(token for token in tokenize(record.text) if len(token) > 1 and token not in STOP_WORDS)
            self._terms.append(terms)
            self._normalized_text.append(normalize_text(record.text))
            self._document_lengths.append(sum(terms.values()))
            for term in track(
                terms,
                desc="OCR terms",
                unit="term",
                nested=True,
            ):
                self._postings[term].append(index)
        self._average_length = (
            sum(self._document_lengths) / len(self._document_lengths)
            if self._document_lengths
            else 1.0
        )
        total = len(self.records)
        self._idf: dict[str, float] = {}
        for term, postings in track(
            self._postings.items(),
            desc="Tính OCR IDF",
            total=len(self._postings),
            unit="term",
            force=True,
            leave=True,
        ):
            self._idf[term] = math.log(
                1.0 + (total - len(postings) + 0.5) / (len(postings) + 0.5)
            )

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
            for line in track(
                stream,
                desc="Đọc OCR index",
                unit="record",
                force=True,
                leave=True,
            ):
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
        term_groups = [
            tuple(dict.fromkeys((term, *SIGN_TERM_ALIASES.get(term, ()))))
            for term in source_terms
        ]
        terms = [
            expanded
            for group in term_groups
            for expanded in group
        ]
        if not terms or not self.records:
            return []
        query_terms = Counter(terms)
        candidate_indices: set[int] = set()
        for term in track(query_terms, desc="OCR query terms", unit="term"):
            candidate_indices.update(self._postings.get(term, ()))
        if not candidate_indices:
            return []

        scored: list[tuple[float, int, int]] = []
        normalized_query = normalize_text(query).strip()
        compact_query = " ".join(source_terms)
        for index in track(
            candidate_indices,
            desc="Chấm OCR BM25",
            total=len(candidate_indices),
            unit="frame",
        ):
            record = self.records[index]
            if video_id and record.video_id != video_id:
                continue
            doc_terms = self._terms[index]
            matched_groups = sum(
                1 for group in term_groups if any(term in doc_terms for term in group)
            )
            # Count semantic groups rather than expanded aliases. A random
            # subtitle matching only one common word should not enter recall.
            required = 1 if len(term_groups) <= 2 else max(2, math.ceil(len(term_groups) * 0.35))
            if matched_groups < required:
                continue
            score = 0.0
            document_length = self._document_lengths[index]
            length_normalizer = 1.0 - 0.75 + 0.75 * document_length / max(self._average_length, 1.0)
            for term, query_count in track(
                query_terms.items(),
                desc="OCR BM25 terms",
                total=len(query_terms),
                unit="term",
                nested=True,
            ):
                frequency = doc_terms.get(term, 0)
                if not frequency:
                    continue
                tf = frequency * 2.2 / (frequency + 1.2 * length_normalizer)
                score += self._idf.get(term, 0.0) * tf * min(query_count, frequency)
            coverage = matched_groups / len(term_groups)
            score *= 1.0 + 0.75 * coverage
            # Phrase matching uses the original query, not the expanded alias
            # stream, so exact signs/subtitles receive the intended boost.
            document_text = self._normalized_text[index]
            if len(normalized_query) >= 6 and normalized_query in document_text:
                score *= 2.0
            elif len(compact_query) >= 6 and compact_query in document_text:
                score *= 1.65
            scored.append((score, matched_groups, index))
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
