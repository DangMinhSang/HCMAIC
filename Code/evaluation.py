"""AIC 2026 preliminary scoring, implemented from the official PDF."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from progress import track
from retrieval import normalize_text


RANK_CUTOFFS = (1, 5, 20, 50, 100)
NUMBER_ALIASES = {
    "khong": "0", "zero": "0",
    "mot": "1", "one": "1",
    "hai": "2", "two": "2",
    "ba": "3", "three": "3",
    "bon": "4", "tu": "4", "four": "4",
    "nam": "5", "five": "5",
    "sau": "6", "six": "6",
    "bay": "7", "seven": "7",
    "tam": "8", "eight": "8",
    "chin": "9", "nine": "9",
    "muoi": "10", "ten": "10",
}


@dataclass(frozen=True)
class Interval:
    start: int
    end: int

    def contains(self, frame_id: int) -> bool:
        return self.start <= frame_id <= self.end


@dataclass(frozen=True)
class KISGroundTruth:
    video_id: str
    interval: Interval


@dataclass(frozen=True)
class QAGroundTruth(KISGroundTruth):
    accepted_answers: tuple[str, ...]


@dataclass(frozen=True)
class TrakeGroundTruth:
    video_id: str
    intervals: tuple[Interval, ...]


def normalize_answer(value: str) -> str:
    normalized = " ".join(normalize_text(value).split())
    without_color_prefix = normalized.removeprefix("mau ").strip()
    return NUMBER_ALIASES.get(without_color_prefix, without_color_prefix)


def answer_matches(answer: str, accepted_answers: Iterable[str]) -> bool:
    candidate = normalize_answer(answer)
    return bool(candidate) and candidate in {normalize_answer(value) for value in accepted_answers}


def final_score(r_scores: Sequence[float], cutoffs: Sequence[int] = RANK_CUTOFFS) -> float:
    """Mean of the best R-Score within each official rank cutoff."""
    values = [float(value) for value in r_scores]
    if not cutoffs:
        return 0.0
    return sum(max(values[:cutoff], default=0.0) for cutoff in cutoffs) / len(cutoffs)


def score_kis(rows: Sequence[dict[str, str]], ground_truth: KISGroundTruth) -> list[float]:
    output: list[float] = []
    candidates = rows[:100]
    for row in track(
        candidates,
        desc="Chấm KIS",
        total=len(candidates),
        unit="answer",
    ):
        try:
            frame_id = int(float(row["frame_id"]))
        except (KeyError, TypeError, ValueError):
            output.append(0.0)
            continue
        correct = row.get("video_id") == ground_truth.video_id and ground_truth.interval.contains(frame_id)
        output.append(float(correct))
    return output


def score_qa(rows: Sequence[dict[str, str]], ground_truth: QAGroundTruth) -> list[float]:
    location_scores = score_kis(rows, ground_truth)
    return [
        location_score
        * float(answer_matches(row.get("answer", ""), ground_truth.accepted_answers))
        for row, location_score in track(
            zip(rows[:100], location_scores),
            desc="Chấm Q&A",
            total=len(location_scores),
            unit="answer",
        )
    ]


def score_trake(rows: Sequence[dict[str, str]], ground_truth: TrakeGroundTruth) -> list[float]:
    output: list[float] = []
    event_count = len(ground_truth.intervals)
    candidates = rows[:100]
    for row in track(
        candidates,
        desc="Chấm TRAKE",
        total=len(candidates),
        unit="answer",
    ):
        if row.get("video_id") != ground_truth.video_id or event_count == 0:
            output.append(0.0)
            continue
        matches = 0
        for index, interval in track(
            enumerate(ground_truth.intervals, start=1),
            desc="Chấm TRAKE events",
            total=len(ground_truth.intervals),
            unit="event",
            nested=True,
        ):
            try:
                frame_id = int(float(row[f"frame_id_{index}"]))
            except (KeyError, TypeError, ValueError):
                continue
            matches += int(interval.contains(frame_id))
        output.append(matches / event_count)
    return output


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def evaluate_file(predictions: Path, ground_truth_path: Path) -> dict[str, object]:
    rows = _read_rows(predictions)
    payload = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    task = str(payload.get("task") or "").lower()
    if task == "kis":
        ground_truth = KISGroundTruth(
            str(payload["video_id"]),
            Interval(int(payload["start"]), int(payload["end"])),
        )
        scores = score_kis(rows, ground_truth)
    elif task == "qa":
        answers = payload.get("answers") or [payload.get("answer", "")]
        ground_truth = QAGroundTruth(
            str(payload["video_id"]),
            Interval(int(payload["start"]), int(payload["end"])),
            tuple(str(answer) for answer in answers),
        )
        scores = score_qa(rows, ground_truth)
    elif task == "trake":
        ground_truth = TrakeGroundTruth(
            str(payload["video_id"]),
            tuple(Interval(int(start), int(end)) for start, end in payload["intervals"]),
        )
        scores = score_trake(rows, ground_truth)
    else:
        raise ValueError("Ground truth task phải là kis, qa hoặc trake.")
    return {
        "task": task,
        "answers": len(scores),
        "r_at": {str(cutoff): max(scores[:cutoff], default=0.0) for cutoff in RANK_CUTOFFS},
        "final_score": final_score(scores),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an AIC 2026 prediction CSV")
    parser.add_argument("predictions", type=Path)
    parser.add_argument("ground_truth", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(evaluate_file(arguments.predictions, arguments.ground_truth), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
