"""Pure ranking helpers shared by the AIC dashboard pipelines."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence


CandidateKey = tuple[str, int]


def candidate_key(item: Any) -> CandidateKey:
    return str(item.video_id), int(item.keyframe_number)


def normalized_scores(values: Sequence[float], *, flat_value: float = 0.5) -> list[float]:
    if not values:
        return []
    minimum, maximum = min(values), max(values)
    spread = maximum - minimum
    if spread < 1e-9:
        return [flat_value] * len(values)
    return [(value - minimum) / spread for value in values]


def select_multisource_candidates(
    visual_results: Sequence[Any],
    ocr_hits: Sequence[Any],
    combined: Mapping[CandidateKey, Any],
    budget: int,
    source_weights: Mapping[str, float] | None = None,
    max_per_video: int | None = None,
    min_frame_gap: int = 0,
) -> list[Any]:
    """Guarantee adaptive Qwen coverage from vision, OCR and metadata."""
    budget = max(1, budget)
    selected: list[Any] = []
    seen: set[CandidateKey] = set()
    video_counts: dict[str, int] = {}
    frames_by_video: dict[str, list[int]] = {}

    def add(item: Any | None, *, enforce_diversity: bool = True) -> bool:
        if item is None or len(selected) >= budget:
            return False
        key = candidate_key(item)
        if key in seen:
            return False
        video_id = str(item.video_id)
        frame_id = int(getattr(item, "frame_id", 0))
        if enforce_diversity:
            if max_per_video is not None and video_counts.get(video_id, 0) >= max_per_video:
                return False
            if min_frame_gap and any(
                abs(frame_id - previous) <= min_frame_gap
                for previous in frames_by_video.get(video_id, ())
            ):
                return False
        seen.add(key)
        selected.append(item)
        video_counts[video_id] = video_counts.get(video_id, 0) + 1
        frames_by_video.setdefault(video_id, []).append(frame_id)
        return True

    def take(items: Iterable[Any], quota: int) -> None:
        added = 0
        for item in items:
            if add(item):
                added += 1
                if added >= quota:
                    break

    weights = source_weights or {}
    if weights:
        visual_share = min(
            0.65,
            max(0.30, float(weights.get("visual", 0.0)) + 0.35 * float(weights.get("object", 0.0))),
        )
        ocr_share = min(0.55, max(0.15, float(weights.get("ocr", 0.0))))
        metadata_share = min(0.30, max(0.08, float(weights.get("metadata", 0.0))))
    else:
        visual_share, ocr_share, metadata_share = 0.55, 0.30, 0.10
    visual_quota = max(1, round(budget * visual_share))
    ocr_quota = max(1, round(budget * ocr_share))
    metadata_quota = max(1, round(budget * metadata_share))
    take((combined.get(candidate_key(item)) for item in visual_results), visual_quota)
    take(
        (combined.get((str(hit.video_id), int(hit.keyframe_number))) for hit in ocr_hits),
        ocr_quota,
    )
    metadata_ranked = sorted(
        combined.values(),
        key=lambda item: float(getattr(item, "metadata_score", 0.0)),
        reverse=True,
    )
    take(metadata_ranked, metadata_quota)
    fused_ranked = sorted(combined.values(), key=lambda value: float(value.score), reverse=True)
    take(fused_ranked, budget)
    # If a filtered query genuinely has only one relevant video or a tight
    # temporal burst, fill every remaining model slot after the diverse pass.
    for item in fused_ranked:
        add(item, enforce_diversity=False)
        if len(selected) >= budget:
            break
    return selected


def fuse_adaptive_retrieval_scores(results: Sequence[Any], profile: Any) -> None:
    """Fuse first-stage evidence using percentages chosen from the query."""
    if not results:
        return
    base_scores = normalized_scores(
        [float(getattr(result, "retrieval_score", getattr(result, "score", 0.0))) for result in results]
    )
    visual_scores = normalized_scores([float(getattr(result, "visual_score", 0.0)) for result in results])
    for index, result in enumerate(results):
        support = profile.support_score(
            visual=visual_scores[index],
            ocr=float(getattr(result, "ocr_score", 0.0)),
            metadata=float(getattr(result, "metadata_score", 0.0)),
            object_score=float(getattr(result, "object_score", 0.0)),
        )
        # Keep a small rank-preserving term so a weak router prediction cannot
        # erase strong broad-recall evidence. The adaptive signal is primary.
        result.score = 0.22 * base_scores[index] + 0.78 * support
        result.retrieval_score = result.score


def select_diverse_results(
    results: Sequence[Any],
    *,
    limit: int,
    min_frame_gap: int,
    max_per_video: int | None,
) -> list[Any]:
    """Apply submission diversity after all retrieval/reranking stages."""
    selected: list[Any] = []
    frames_by_video: dict[str, list[int]] = {}
    for result in sorted(results, key=lambda item: float(item.score), reverse=True):
        nearby = frames_by_video.setdefault(str(result.video_id), [])
        if max_per_video is not None and len(nearby) >= max_per_video:
            continue
        if any(abs(int(result.frame_id) - frame_id) <= min_frame_gap for frame_id in nearby):
            continue
        nearby.append(int(result.frame_id))
        result.rank = len(selected) + 1
        selected.append(result)
        if len(selected) >= limit:
            break
    return selected
