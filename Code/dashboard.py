"""AIC 2026 retrieval workspace for KIS, Q&A, and TRAKE."""

from __future__ import annotations

import argparse
import csv
import io
import os
import secrets
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, render_template, request, send_file, session, url_for
from werkzeug.exceptions import HTTPException

from data_paths import DatasetNotFoundError
from multimodal_reranker import (
    MultimodalRerankerUnavailableError,
    QwenVLQueryReranker,
)
from ocr_index import OCRMemoryIndex
from ocr_regions import OCR_INDEX_SCHEMA_VERSION
from progress import track
from qa import VQABaseline, split_qa_query
from query_language import normalize_query
from query_router import QueryProfile, build_query_profile
from ranking import (
    fuse_adaptive_retrieval_scores,
    normalized_scores,
    select_diverse_results,
    select_multisource_candidates,
)
from retrieval import AICRetrievalEngine, SearchResult, TrakeVideoResult


app = Flask(__name__)
app.secret_key = os.environ.get("AIC_WEB_SECRET", secrets.token_urlsafe(32))


@app.errorhandler(HTTPException)
def json_api_http_error(error: HTTPException):
    """Never make API clients parse Flask's default HTML error document."""
    if request.path.startswith("/api/"):
        return jsonify({"error": error.description, "status": error.code}), error.code
    return error

ENGINE: AICRetrievalEngine | None = None
VQA: VQABaseline | None = None
OCR_INDEX: OCRMemoryIndex | None = None
OCR_INDEX_LOADED = False
RERANKER: QwenVLQueryReranker | None = None
RERANKER_ATTEMPTED = False
RERANKER_ERROR = ""
ENGINE_LOCK = threading.RLock()
SESSIONS: dict[str, "SearchSession"] = {}
SESSIONS_LOCK = threading.RLock()
SEARCH_JOBS: dict[str, "SearchJob"] = {}
SEARCH_JOBS_LOCK = threading.RLock()


def bounded_environment_integer(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        return max(minimum, min(int(os.environ.get(name, str(default))), maximum))
    except ValueError:
        return default


# Qwen and CLIP share one Kaggle GPU. A single worker keeps memory predictable;
# advanced deployments can opt into two workers only when they have enough VRAM.
SEARCH_EXECUTOR = ThreadPoolExecutor(
    max_workers=bounded_environment_integer("AIC_SEARCH_WORKERS", 1, 1, 2),
    thread_name_prefix="aic-search",
)


@dataclass
class StoredResult:
    result: SearchResult
    group: str | None = None
    event_index: int | None = None


@dataclass
class SearchSession:
    task: str = "kis"
    results: dict[str, StoredResult] = field(default_factory=dict)
    trake_sequences: dict[str, TrakeVideoResult] = field(default_factory=dict)
    touched_at: float = field(default_factory=time.monotonic)


@dataclass
class SearchJob:
    session_id: str
    task: str
    query: str
    status: str = "queued"
    stage: str = "Đang chờ GPU…"
    created_at: float = field(default_factory=time.monotonic)
    started_at: float = 0.0
    finished_at: float = 0.0
    elapsed_ms: int = 0
    notice: str = ""
    query_profile: dict[str, float | str] = field(default_factory=dict)
    result_ids: list[str] = field(default_factory=list)
    error: str = ""


def get_engine() -> AICRetrievalEngine:
    global ENGINE
    with ENGINE_LOCK:
        if ENGINE is None:
            ENGINE = AICRetrievalEngine.from_environment()
        return ENGINE


def get_reranker() -> QwenVLQueryReranker | None:
    """Load the optional multimodal reranker once, on the first KIS query."""
    global RERANKER, RERANKER_ATTEMPTED, RERANKER_ERROR
    with ENGINE_LOCK:
        if RERANKER_ATTEMPTED:
            return RERANKER
        RERANKER_ATTEMPTED = True
        default_enabled = "1" if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") else "0"
        if os.environ.get("AIC_RERANKER", default_enabled).lower() in {"0", "false", "no"}:
            RERANKER_ERROR = "đã tắt qua AIC_RERANKER=0"
            return None
        try:
            RERANKER = QwenVLQueryReranker()
        except MultimodalRerankerUnavailableError as error:
            RERANKER_ERROR = str(error)
        return RERANKER


def warmup_reranker() -> bool:
    """Exercise a real image pair before exposing the public dashboard."""
    global RERANKER, RERANKER_ERROR
    reranker = get_reranker()
    if reranker is None:
        return False
    engine = get_engine()
    try:
        for video_id in track(
            engine._features,
            desc="Tìm ảnh warmup Qwen",
            total=len(engine._features),
            unit="video",
            force=True,
        ):
            mapping = engine._mapping(video_id)
            if not mapping:
                continue
            sample = engine.result_for_keyframe(video_id, mapping[0].keyframe_number)
            if sample is not None and sample.image_path:
                reranker.score("a representative video frame", [sample])
                build_query_profile("exact words written on a yellow warning sign", reranker)
                return True
        raise MultimodalRerankerUnavailableError("Không tìm thấy keyframe để warmup Qwen.")
    except MultimodalRerankerUnavailableError as error:
        RERANKER_ERROR = str(error)
        RERANKER = None
        return False


def get_vqa() -> VQABaseline:
    global VQA
    with ENGINE_LOCK:
        if VQA is None:
            VQA = VQABaseline()
        return VQA


def warmup_vqa() -> str:
    """Exercise the selected VQA backend on a real mounted keyframe."""
    engine = get_engine()
    vqa = get_vqa()
    for video_id in track(
        engine._features,
        desc="Tìm ảnh warmup VQA",
        total=len(engine._features),
        unit="video",
        force=True,
    ):
        mapping = engine._mapping(video_id)
        if not mapping:
            continue
        sample = engine.result_for_keyframe(video_id, mapping[0].keyframe_number)
        if sample is not None and sample.image_path:
            vqa.warmup(sample.image_path)
            return vqa.backend_name
    vqa.warmup()
    return vqa.backend_name


def get_ocr_index() -> OCRMemoryIndex | None:
    """Load text-only OCR postings into RAM once, before the first query."""
    global OCR_INDEX, OCR_INDEX_LOADED
    with ENGINE_LOCK:
        if OCR_INDEX_LOADED:
            return OCR_INDEX
        OCR_INDEX_LOADED = True
        configured = os.environ.get("AIC_OCR_INDEX")
        default_path = Path("/kaggle/working/aic_ocr_index.jsonl.gz")
        index_path = Path(configured) if configured else default_path
        if index_path.is_file():
            OCR_INDEX = OCRMemoryIndex.load(index_path)
            if OCR_INDEX.legacy_record_count:
                print(
                    "[warning] OCR index v1 không có tọa độ text; đang dùng bộ lọc ticker fallback. "
                    "Chạy lại run.py để tạo OCR index v2 chính xác hơn.",
                    flush=True,
                )
        return OCR_INDEX


def current_session_identifier() -> str:
    identifier = session.get("aic_session")
    if not identifier:
        identifier = uuid.uuid4().hex
        session["aic_session"] = identifier
    return str(identifier)


def session_state(identifier: str) -> SearchSession:
    with SESSIONS_LOCK:
        state = SESSIONS.setdefault(identifier, SearchSession())
        state.touched_at = time.monotonic()
        return state


def current_session() -> SearchSession:
    return session_state(current_session_identifier())


def prune_search_jobs() -> None:
    """Bound completed job/session state without touching active GPU work."""
    now = time.monotonic()
    ttl = bounded_environment_integer("AIC_SEARCH_JOB_TTL", 3600, 300, 86400)
    with SEARCH_JOBS_LOCK:
        expired = [
            identifier
            for identifier, job in SEARCH_JOBS.items()
            if job.status in {"complete", "error"} and now - (job.finished_at or job.created_at) > ttl
        ]
        for identifier in track(
            expired,
            desc="Dọn search jobs hết hạn",
            total=len(expired),
            unit="job",
            nested=True,
        ):
            SEARCH_JOBS.pop(identifier, None)
        # A public share URL can be scanned by bots. Keep at most 256 compact
        # job records even if their cookies are never used again.
        finished = sorted(
            (
                (identifier, job)
                for identifier, job in SEARCH_JOBS.items()
                if job.status in {"complete", "error"}
            ),
            key=lambda item: item[1].finished_at,
        )
        overflow_jobs = finished[: max(0, len(SEARCH_JOBS) - 256)]
        for identifier, _job in track(
            overflow_jobs,
            desc="Dọn search job overflow",
            total=len(overflow_jobs),
            unit="job",
            nested=True,
        ):
            SEARCH_JOBS.pop(identifier, None)
        retained_sessions = {job.session_id for job in SEARCH_JOBS.values()}

    session_ttl = bounded_environment_integer("AIC_SESSION_TTL", 7200, 600, 86400)
    with SESSIONS_LOCK:
        stale_sessions = [
            identifier
            for identifier, state in SESSIONS.items()
            if identifier not in retained_sessions and now - state.touched_at > session_ttl
        ]
        for identifier in track(
            stale_sessions,
            desc="Dọn session hết hạn",
            total=len(stale_sessions),
            unit="session",
            nested=True,
        ):
            SESSIONS.pop(identifier, None)
        removable = sorted(
            (
                (identifier, state)
                for identifier, state in SESSIONS.items()
                if identifier not in retained_sessions
            ),
            key=lambda item: item[1].touched_at,
        )
        overflow_sessions = removable[: max(0, len(SESSIONS) - 128)]
        for identifier, _state in track(
            overflow_sessions,
            desc="Dọn session overflow",
            total=len(overflow_sessions),
            unit="session",
            nested=True,
        ):
            SESSIONS.pop(identifier, None)


def time_label(seconds: float) -> str:
    minutes, rest = divmod(max(seconds, 0), 60)
    return f"{int(minutes):02d}:{rest:06.3f}"


def language_note(engine: AICRetrievalEngine) -> str:
    info = getattr(engine.encoder, "last_query", None)
    if info is None or info.language != "vi":
        return "Đã nhận dạng English."
    if info.translation_used:
        return f"VI → EN: {info.text_for_model}"
    return info.warning or "Đang dùng truy vấn tiếng Việt gốc."


def vector_count_compat(engine: AICRetrievalEngine) -> int:
    """Support a notebook process that still has an older retrieval module cached."""
    value = getattr(engine, "vector_count", None)
    if value is not None:
        return int(value)
    try:
        import numpy as np

        return sum(int(np.load(path, mmap_mode="r").shape[0]) for path in engine._features.values())
    except (AttributeError, OSError, ValueError):
        return 0


def as_payload(identifier: str, stored: StoredResult) -> dict[str, Any]:
    result = stored.result
    return {
        "id": identifier,
        "rank": result.rank,
        "video_id": result.video_id,
        "frame_id": result.frame_id,
        "pts_time": round(result.pts_time, 3),
        "time": time_label(result.pts_time),
        "score": round(result.score, 3),
        "clip": round(result.visual_score, 3),
        "retrieval_score": round(getattr(result, "retrieval_score", 0.0), 3),
        "metadata_score": round(result.metadata_score, 3),
        "object_score": round(getattr(result, "object_score", 0.0), 3),
        "keyframe_number": result.keyframe_number,
        "title": result.title,
        "objects": list(result.object_labels[:8]),
        "answer": getattr(result, "answer", ""),
        "qa_confidence": round(getattr(result, "qa_confidence", 0.0), 3),
        "ocr_score": round(getattr(result, "ocr_score", 0.0), 3),
        "ocr_quality": round(getattr(result, "ocr_quality", 1.0), 3),
        "ocr_text": getattr(result, "ocr_text", ""),
        "ai_score": round(getattr(result, "rerank_score", 0.0), 3),
        "ai_joint_score": round(getattr(result, "rerank_joint_score", 0.0), 3),
        "ai_visual_score": round(getattr(result, "rerank_visual_score", 0.0), 3),
        "ai_ocr_score": round(getattr(result, "rerank_ocr_score", 0.0), 3),
        "event": stored.event_index,
        # Include the WSGI mount prefix when the app is exposed under
        # /dashboard by the Kaggle Gradio share gateway.
        "image_url": url_for("media", identifier=identifier),
        "video_url": url_for("video", identifier=identifier) if result.video_path else "",
    }


def search_options(body: dict[str, Any]) -> tuple[int, int, int | None, str | None]:
    options = body.get("options") or {}
    top_k = max(1, min(int(options.get("top_k", 100)), 100))
    min_gap = max(0, min(int(options.get("dedupe", 0)), 600))
    max_per_video_raw = int(options.get("max_per_video", 4))
    max_per_video = max(1, min(max_per_video_raw, 100)) if max_per_video_raw else None
    video_id = str(options.get("video_id") or "").strip() or None
    return top_k, min_gap, max_per_video, video_id


def make_kis_results(
    engine: AICRetrievalEngine,
    query: str,
    body: dict[str, Any],
    *,
    profile: QueryProfile | None = None,
    reranker: QwenVLQueryReranker | None = None,
) -> tuple[list[StoredResult], str]:
    top_k, min_gap, maximum, video_id = search_options(body)
    if profile is None:
        reranker = reranker or get_reranker()
        profile = build_query_profile(query, reranker)
    default_reranker_enabled = "1" if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") else "0"
    reranker_requested = os.environ.get("AIC_RERANKER", default_reranker_enabled).lower() not in {
        "0",
        "false",
        "no",
    }
    # Recall broadly before applying the user's final Top-K/diversity rules.
    # Otherwise the expensive multimodal model would only see the first 16
    # CLIP answers and could not recover a correct frame ranked 30th or 80th.
    recall_top_k = 100 if reranker_requested else top_k
    recall_min_gap = 0 if reranker_requested else min_gap
    recall_max_per_video = None if reranker_requested else maximum
    try:
        results = engine.search(
            query,
            top_k=recall_top_k,
            min_frame_gap=recall_min_gap,
            max_per_video=recall_max_per_video,
            video_id=video_id,
            metadata_weight=profile.metadata_weight,
        )
    except TypeError as error:
        # A running Kaggle kernel can retain an older retrieval module in
        # sys.modules after git pull. Keep the dashboard usable until restart.
        if "unexpected keyword" not in str(error):
            raise
        results = engine.search(
            query,
            top_k=recall_top_k,
            min_frame_gap=recall_min_gap,
            metadata_weight=profile.metadata_weight,
        )
        if video_id:
            results = [result for result in results if result.video_id == video_id]
        if maximum and not reranker_requested:
            counts: dict[str, int] = {}
            filtered = []
            for result in track(
                results,
                desc="Lọc giới hạn mỗi video",
                total=len(results),
                unit="frame",
            ):
                counts[result.video_id] = counts.get(result.video_id, 0) + 1
                if counts[result.video_id] <= maximum:
                    filtered.append(result)
            results = filtered
    ocr_index = get_ocr_index()
    hits = ocr_index.search(query, limit=max(200, min(top_k * 4, 400)), video_id=video_id) if ocr_index else []
    combined = {(result.video_id, result.keyframe_number): result for result in results}
    for hit in track(
        hits,
        desc="Ghép OCR candidates",
        total=len(hits),
        unit="frame",
    ):
        key = (hit.video_id, hit.keyframe_number)
        text_quality = float(getattr(hit, "text_quality", 1.0))
        evidence_score = float(getattr(hit, "effective_score", hit.score))
        # A coordinate-free legacy index can contain a lower-third. Never
        # give untrusted ticker text to Qwen, or it can reinforce the same
        # false positive after first-stage OCR recall.
        trusted_text = (
            hit.text
            if text_quality >= 0.50
            and int(getattr(hit, "schema_version", 1)) >= OCR_INDEX_SCHEMA_VERSION
            else ""
        )
        existing = combined.get(key)
        if existing is None:
            existing = engine.result_for_keyframe(
                hit.video_id,
                hit.keyframe_number,
                # OCR is useful evidence, but must not overpower the visual
                # signal before the multimodal reranker gets a chance to
                # inspect the actual frame.
                score=0.55 * evidence_score,
                ocr_score=evidence_score,
                ocr_quality=text_quality,
                ocr_text=trusted_text,
            )
            if existing is None:
                continue
            combined[key] = existing
        else:
            existing.ocr_score = max(existing.ocr_score, evidence_score)
            existing.ocr_quality = text_quality
            existing.ocr_text = trusted_text
            existing.score = max(
                existing.score,
                0.40 * existing.visual_score
                + 0.45 * evidence_score
                + 0.10 * existing.metadata_score,
            )
            existing.retrieval_score = existing.score

    fuse_adaptive_retrieval_scores(list(combined.values()), profile)
    rerank_note = ""
    if reranker and combined:
        # Qwen is a cross-attention reranker, not a corpus scanner. Keep the
        # expensive pass bounded while ensuring OCR hits can enter the pool.
        try:
            # The HTTP layer is asynchronous, so the accuracy-first default
            # can inspect 32 multimodal candidates without a gateway timeout.
            rerank_budget = max(1, min(int(os.environ.get("AIC_RERANKER_CANDIDATES", "32")), 100))
        except ValueError:
            rerank_budget = 32
        rerank_limit = min(rerank_budget, len(combined))
        rerank_candidates = select_multisource_candidates(
            results,
            hits,
            combined,
            rerank_limit,
            source_weights=profile.values(),
            max_per_video=None if video_id else max(4, (rerank_limit + 5) // 6),
            min_frame_gap=30,
        )
        try:
            normalized = normalize_query(query)
            rerank_query = query
            if normalized.text_for_model.lower() != query.lower():
                rerank_query = f"{query}\nEnglish translation: {normalized.text_for_model}"
            rerank_scores = reranker.score(rerank_query, rerank_candidates)
            for result, rerank_score in track(
                zip(rerank_candidates, rerank_scores),
                desc="Fuse Qwen scores",
                total=len(rerank_candidates),
                unit="frame",
            ):
                result.rerank_joint_score = rerank_score.joint
                result.rerank_visual_score = rerank_score.visual
                result.rerank_ocr_score = rerank_score.ocr
                adaptive_support = profile.support_score(
                    visual=rerank_score.visual,
                    ocr=rerank_score.ocr,
                    metadata=result.metadata_score,
                    object_score=result.object_score,
                )
                result.score = 0.80 * rerank_score.joint + 0.20 * adaptive_support
                result.rerank_score = result.score
            # Candidates outside the reranker budget remain available as a
            # diversity fallback, but cannot outrank model-scored candidates.
            if len(rerank_candidates) < len(combined):
                candidate_ids = {id(result) for result in rerank_candidates}
                floor = min(result.score for result in rerank_candidates)
                for result in track(
                    combined.values(),
                    desc="Hạ điểm ngoài Qwen pool",
                    total=len(combined),
                    unit="frame",
                ):
                    if id(result) not in candidate_ids:
                        result.score = min(result.score, floor - 0.001)
            rerank_note = f" · AI rerank Qwen: {len(rerank_candidates)} ảnh"
        except MultimodalRerankerUnavailableError as error:
            rerank_note = f" · AI rerank lỗi, dùng fallback: {str(error)[:140]}"
    elif RERANKER_ERROR:
        rerank_note = f" · AI rerank fallback: {RERANKER_ERROR[:140]}"

    selected = select_diverse_results(
        list(combined.values()),
        limit=top_k,
        min_frame_gap=min_gap,
        max_per_video=maximum,
    )
    if ocr_index:
        ocr_note = f"OCR RAM v{ocr_index.schema_version}: {len(hits)} keyframe scene-text."
        if ocr_index.legacy_record_count:
            ocr_note += f" · legacy filter: {ocr_index.legacy_record_count:,} record"
    else:
        ocr_note = "OCR index chưa được nạp."
    return (
        [StoredResult(result) for result in selected],
        language_note(engine) + " · " + profile.summary() + " · " + ocr_note + rerank_note,
    )


def make_qa_results(
    engine: AICRetrievalEngine,
    query: str,
    body: dict[str, Any],
    *,
    profile: QueryProfile | None = None,
    reranker: QwenVLQueryReranker | None = None,
) -> tuple[list[StoredResult], str]:
    event, question = split_qa_query(query)
    stored, note = make_kis_results(
        engine,
        event,
        body,
        profile=profile,
        reranker=reranker,
    )
    results = [item.result for item in stored]
    try:
        vqa = get_vqa()
        predictions = vqa.predict(question, results)
        by_rank = {prediction.rank: prediction for prediction in predictions}
        for result in track(
            results,
            desc="Gắn đáp án VQA",
            total=len(results),
            unit="frame",
        ):
            prediction = by_rank.get(result.rank)
            if prediction:
                result.answer = prediction.answer
                result.qa_confidence = prediction.confidence
        if predictions:
            answer_scores: dict[str, float] = {}
            answer_labels: dict[str, str] = {}
            for prediction in track(
                predictions,
                desc="Tổng hợp VQA",
                total=len(predictions),
                unit="answer",
            ):
                key = " ".join(prediction.answer.lower().split())
                answer_labels.setdefault(key, prediction.answer)
                rank_weight = 1.0 / (1.0 + 0.10 * max(prediction.rank - 1, 0))
                answer_scores[key] = answer_scores.get(key, 0.0) + prediction.confidence * rank_weight
            best_key = max(answer_scores, key=answer_scores.get)
            consensus_answer = answer_labels[best_key]
            best_confidence = max(
                prediction.confidence
                for prediction in predictions
                if " ".join(prediction.answer.lower().split()) == best_key
            )
            for result in track(
                results,
                desc="Điền VQA đồng thuận",
                total=len(results),
                unit="frame",
            ):
                if not result.answer:
                    result.answer = consensus_answer
                    result.qa_confidence = best_confidence
            note += f" · VQA đồng thuận: {consensus_answer} ({best_confidence:.0%})"
        else:
            note += " · Không có ảnh để chạy VQA."
        note += f" · VQA model: {vqa.backend_name}"
        if vqa.load_error:
            note += f" ({vqa.load_error[:100]})"
    except RuntimeError as error:
        # Retrieval must remain available if the optional VQA checkpoint is offline.
        note += f" · VQA chưa sẵn sàng: {error}"
    return stored, note


def make_trake_results(
    engine: AICRetrievalEngine,
    query: str,
    body: dict[str, Any],
    *,
    profile: QueryProfile | None = None,
    reranker: QwenVLQueryReranker | None = None,
) -> tuple[list[StoredResult], dict[str, TrakeVideoResult], str]:
    events = [line.strip().lstrip("-0123456789. ") for line in query.splitlines() if line.strip()]
    top_k, _gap, _maximum, _video_id = search_options(body)
    sequences = engine.search_trake(events, top_videos=min(top_k, 100))
    rerank_note = ""
    profile = profile or build_query_profile(query, reranker)
    if reranker and sequences:
        try:
            pair_budget = max(1, min(int(os.environ.get("AIC_TRAKE_RERANK_PAIRS", "32")), 100))
        except ValueError:
            pair_budget = 32
        sequence_limit = min(len(sequences), max(1, pair_budget // len(events)))
        reranked_sequences = sequences[:sequence_limit]
        center_scored_ids = {id(sequence) for sequence in reranked_sequences}
        event_queries: list[str] = []
        event_vectors = []
        for event in track(
            events,
            desc="Chuẩn hóa TRAKE events",
            total=len(events),
            unit="event",
            force=True,
        ):
            normalized = normalize_query(event)
            event_queries.append(
                event
                if normalized.text_for_model.lower() == event.lower()
                else f"{event}\nEnglish translation: {normalized.text_for_model}"
            )
            event_vectors.append(engine.encoder.encode(event))
        pair_queries: list[str] = []
        pair_frames: list[SearchResult] = []
        for sequence in track(
            reranked_sequences,
            desc="Chuẩn bị Qwen TRAKE",
            total=len(reranked_sequences),
            unit="sequence",
        ):
            for event_query, frame in track(
                zip(event_queries, sequence.frames),
                desc=f"TRAKE {sequence.video_id}",
                total=len(sequence.frames),
                unit="event",
            ):
                pair_queries.append(event_query)
                pair_frames.append(frame)
        pair_prompt = (
            "Score whether this exact video frame is the requested semantic moment. "
            "Use the visible action, temporal stage, objects, and relevant text."
        )
        try:
            pair_scores = reranker.score_pairs(
                pair_queries,
                pair_frames,
                prompt=pair_prompt,
            )
            clip_scaled = normalized_scores([sequence.score for sequence in sequences])
            clip_by_sequence = {
                id(sequence): clip_score for sequence, clip_score in zip(sequences, clip_scaled)
            }
            offset = 0
            for index, sequence in track(
                enumerate(sequences),
                desc="Fuse TRAKE center",
                total=len(sequences),
                unit="sequence",
            ):
                if index < sequence_limit:
                    values = pair_scores[offset : offset + len(events)]
                    offset += len(events)
                    model_score = 0.72 * (sum(values) / len(values)) + 0.28 * min(values)
                    sequence.score = 0.85 * model_score + 0.15 * clip_scaled[index]
                    for frame, value in track(
                        zip(sequence.frames, values),
                        desc=f"Gắn TRAKE {sequence.video_id}",
                        total=len(sequence.frames),
                        unit="event",
                    ):
                        frame.rerank_score = value
                        frame.rerank_joint_score = value
                        frame.score = value
                else:
                    sequence.score = 0.35 * clip_scaled[index]
            sequences.sort(key=lambda item: item.score, reverse=True)

            try:
                radius = max(0, min(int(os.environ.get("AIC_TRAKE_REFINE_RADIUS", "1")), 2))
                refine_limit = max(0, min(int(os.environ.get("AIC_TRAKE_REFINE_SEQUENCES", "2")), 5))
            except ValueError:
                radius, refine_limit = 1, 2
            refinement_targets = [
                sequence for sequence in sequences if id(sequence) in center_scored_ids
            ][:refine_limit]
            refinement_frames: list[SearchResult] = []
            refinement_queries: list[str] = []
            refinement_groups: list[
                tuple[TrakeVideoResult, list[list[tuple[int, SearchResult]]]]
            ] = []
            if radius and refinement_targets:
                for sequence in track(
                    refinement_targets,
                    desc="Chuẩn bị TRAKE refine",
                    total=len(refinement_targets),
                    unit="sequence",
                    force=True,
                ):
                    groups: list[list[tuple[int, SearchResult]]] = []
                    for event_index, center in track(
                        enumerate(sequence.frames),
                        desc=f"Lấy lân cận {sequence.video_id}",
                        total=len(sequence.frames),
                        unit="event",
                        nested=True,
                    ):
                        neighbors = engine.neighboring_keyframes(
                            sequence.video_id,
                            center.keyframe_number,
                            radius=radius,
                            query_vector=event_vectors[event_index],
                        )
                        groups.append(neighbors)
                        refinement_frames.extend(frame for _index, frame in neighbors)
                        refinement_queries.extend([event_queries[event_index]] * len(neighbors))
                    refinement_groups.append((sequence, groups))
                refinement_scores = reranker.score_pairs(
                    refinement_queries,
                    refinement_frames,
                    prompt=pair_prompt,
                )
                score_offset = 0
                for sequence, groups in track(
                    refinement_groups,
                    desc="Fuse TRAKE refine",
                    total=len(refinement_groups),
                    unit="sequence",
                    force=True,
                ):
                    grouped_scores: list[list[float]] = []
                    for group in track(
                        groups,
                        desc=f"Nhóm refine {sequence.video_id}",
                        total=len(groups),
                        unit="event",
                        nested=True,
                    ):
                        grouped_scores.append(
                            refinement_scores[score_offset : score_offset + len(group)]
                        )
                        score_offset += len(group)
                    choices = engine._ordered_candidate_alignment(
                        [[feature_index for feature_index, _frame in group] for group in groups],
                        grouped_scores,
                    )
                    if choices is None:
                        continue
                    selected_frames = [
                        groups[event_index][choice][1]
                        for event_index, choice in enumerate(choices)
                    ]
                    selected_scores = [
                        grouped_scores[event_index][choice]
                        for event_index, choice in enumerate(choices)
                    ]
                    model_score = 0.72 * (sum(selected_scores) / len(selected_scores)) + 0.28 * min(selected_scores)
                    sequence.frames = selected_frames
                    sequence.score = 0.88 * model_score + 0.12 * clip_by_sequence[id(sequence)]
                    for frame, value in track(
                        zip(selected_frames, selected_scores),
                        desc=f"Gắn refine {sequence.video_id}",
                        total=len(selected_frames),
                        unit="event",
                        nested=True,
                    ):
                        frame.rerank_score = value
                        frame.rerank_joint_score = value
                        frame.score = value

            sequences.sort(key=lambda item: item.score, reverse=True)
            for rank, sequence in track(
                enumerate(sequences, start=1),
                desc="Xếp hạng TRAKE",
                total=len(sequences),
                unit="sequence",
            ):
                sequence.rank = rank
                for frame in track(
                    sequence.frames,
                    desc=f"Gắn hạng {sequence.video_id}",
                    total=len(sequence.frames),
                    unit="event",
                    nested=True,
                ):
                    frame.rank = rank
            rerank_note = (
                f" · Qwen center: {len(pair_frames)} cặp"
                + (f" · refine ±{radius}: {len(refinement_frames)} frame" if refinement_frames else "")
            )
        except MultimodalRerankerUnavailableError as error:
            rerank_note = f" · Qwen TRAKE fallback: {str(error)[:120]}"
    stored: list[StoredResult] = []
    indexed: dict[str, TrakeVideoResult] = {}
    for sequence in track(
        sequences,
        desc="Lưu TRAKE results",
        total=len(sequences),
        unit="sequence",
    ):
        group = f"trake-{sequence.rank}"
        indexed[group] = sequence
        for event_index, frame in track(
            enumerate(sequence.frames, start=1),
            desc=f"Lưu {sequence.video_id}",
            total=len(sequence.frames),
            unit="event",
            nested=True,
        ):
            stored.append(StoredResult(frame, group=group, event_index=event_index))
    return (
        stored,
        indexed,
        "TRAKE căn chỉnh có thứ tự thời gian; chọn một card để xuất cả chuỗi video. · "
        + profile.summary()
        + rerank_note,
    )


@app.get("/")
def index():
    # The dashboard is served at / locally but under /dashboard/ when it is
    # mounted by the Gradio gateway. Generate API URLs from the current WSGI
    # script root instead of relying on browser-relative URLs.
    return render_template(
        "dashboard.html",
        app_config={
            "search": url_for("search"),
            "export": url_for("export"),
            "health": url_for("health"),
        },
    )


@app.get("/api/health")
def health():
    try:
        engine = get_engine()
        ocr_index = get_ocr_index()
        return jsonify(
            {
                "ok": True,
                "video_count": engine.video_count,
                "vector_count": vector_count_compat(engine),
                "video_ids": sorted(engine._features),
                "ocr_records": ocr_index.record_count if ocr_index else 0,
                "ocr_schema": ocr_index.schema_version if ocr_index else 0,
                "ocr_legacy_records": ocr_index.legacy_record_count if ocr_index else 0,
                "feature_cache_loaded": engine.feature_cache_loaded,
                "reranker_ready": RERANKER is not None,
                "reranker_error": RERANKER_ERROR,
                "vqa_backend": VQA.backend_name if VQA is not None else "chưa tải",
                "vqa_error": VQA.load_error if VQA is not None else "",
            }
        )
    except (DatasetNotFoundError, OSError) as error:
        return jsonify({"ok": False, "error": str(error)}), 503


def run_search_job(
    job_id: str,
    session_id: str,
    task: str,
    query: str,
    body: dict[str, Any],
) -> None:
    """Run a potentially long GPU query outside the Gradio HTTP request."""
    started = time.perf_counter()
    with SEARCH_JOBS_LOCK:
        job = SEARCH_JOBS.get(job_id)
        if job is None:
            return
        job.status = "running"
        job.stage = "CLIP/OCR recall → multimodal rerank…"
        job.started_at = time.monotonic()
    try:
        engine = get_engine()
        reranker = get_reranker()
        profile_query = split_qa_query(query)[0] if task == "qa" else query
        profile = build_query_profile(profile_query, reranker)
        with ENGINE_LOCK:
            if task == "kis":
                stored, notice = make_kis_results(
                    engine,
                    query,
                    body,
                    profile=profile,
                    reranker=reranker,
                )
                sequences: dict[str, TrakeVideoResult] = {}
            elif task == "qa":
                stored, notice = make_qa_results(
                    engine,
                    query,
                    body,
                    profile=profile,
                    reranker=reranker,
                )
                sequences = {}
            else:
                stored, sequences, notice = make_trake_results(
                    engine,
                    query,
                    body,
                    profile=profile,
                    reranker=reranker,
                )
        result_ids: list[str] = []
        with SESSIONS_LOCK:
            state = SESSIONS.setdefault(session_id, SearchSession())
            state.task = task
            state.results.clear()
            state.trake_sequences = sequences
            state.touched_at = time.monotonic()
            for item in track(
                stored,
                desc="Gắn result IDs",
                total=len(stored),
                unit="result",
            ):
                identifier = uuid.uuid4().hex
                state.results[identifier] = item
                result_ids.append(identifier)
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        with SEARCH_JOBS_LOCK:
            job = SEARCH_JOBS.get(job_id)
            if job is not None:
                job.status = "complete"
                job.stage = "Hoàn thành."
                job.finished_at = time.monotonic()
                job.elapsed_ms = elapsed_ms
                job.notice = notice
                job.query_profile = profile.as_dict()
                job.result_ids = result_ids
    except (DatasetNotFoundError, ValueError, RuntimeError, OSError) as error:
        with SEARCH_JOBS_LOCK:
            job = SEARCH_JOBS.get(job_id)
            if job is not None:
                job.status = "error"
                job.stage = "Truy vấn thất bại."
                job.finished_at = time.monotonic()
                job.elapsed_ms = round((time.perf_counter() - started) * 1000)
                job.error = str(error)
    except Exception as error:  # keep a worker failure observable to the UI
        with SEARCH_JOBS_LOCK:
            job = SEARCH_JOBS.get(job_id)
            if job is not None:
                job.status = "error"
                job.stage = "Backend gặp lỗi ngoài dự kiến."
                job.finished_at = time.monotonic()
                job.elapsed_ms = round((time.perf_counter() - started) * 1000)
                job.error = f"{type(error).__name__}: {error}"


@app.post("/api/search")
def search():
    """Queue a search and return immediately, avoiding Gradio's 504 timeout."""
    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict):
        return jsonify({"error": "JSON body phải là một object."}), 400
    task = str(body.get("task") or "kis").lower()
    query = str(body.get("query") or "").strip()
    if task not in {"kis", "qa", "trake"}:
        return jsonify({"error": "Loại truy vấn không hợp lệ."}), 400
    if not query:
        return jsonify({"error": "Hãy nhập truy vấn."}), 400
    if len(query) > 4000:
        return jsonify({"error": "Query quá dài; giới hạn 4.000 ký tự."}), 400
    prune_search_jobs()
    job_id = uuid.uuid4().hex
    session_id = current_session_identifier()
    with SEARCH_JOBS_LOCK:
        duplicate = next(
            (
                (identifier, job)
                for identifier, job in SEARCH_JOBS.items()
                if job.session_id == session_id
                and job.task == task
                and job.query == query
                and job.status in {"queued", "running"}
            ),
            None,
        )
        if duplicate is not None:
            duplicate_id, duplicate_job = duplicate
            return (
                jsonify(
                    {
                        "job_id": duplicate_id,
                        "status": duplicate_job.status,
                        "stage": duplicate_job.stage,
                        "status_url": url_for("search_status", job_id=duplicate_id),
                    }
                ),
                202,
            )
        active_jobs = sum(job.status in {"queued", "running"} for job in SEARCH_JOBS.values())
        queue_limit = bounded_environment_integer("AIC_SEARCH_QUEUE_LIMIT", 8, 1, 64)
        if active_jobs >= queue_limit:
            return jsonify({"error": "GPU queue đang đầy; hãy thử lại sau khi query hiện tại hoàn tất."}), 429
        SEARCH_JOBS[job_id] = SearchJob(session_id=session_id, task=task, query=query)
    SEARCH_EXECUTOR.submit(run_search_job, job_id, session_id, task, query, body)
    return (
        jsonify(
            {
                "job_id": job_id,
                "status": "queued",
                "stage": "Đang chờ GPU…",
                "status_url": url_for("search_status", job_id=job_id),
            }
        ),
        202,
    )


@app.get("/api/search/<job_id>")
def search_status(job_id: str):
    """Return a short polling response and materialize URLs in this mount."""
    session_id = current_session_identifier()
    with SEARCH_JOBS_LOCK:
        job = SEARCH_JOBS.get(job_id)
        if job is None or job.session_id != session_id:
            return jsonify({"error": "Không tìm thấy query job trong phiên này."}), 404
        status = job.status
        response: dict[str, Any] = {
            "job_id": job_id,
            "status": status,
            "stage": job.stage,
            "elapsed_ms": job.elapsed_ms,
        }
        if status == "error":
            response["error"] = job.error or "Truy vấn thất bại."
            return jsonify(response)
        if status != "complete":
            if job.started_at:
                response["elapsed_ms"] = round((time.monotonic() - job.started_at) * 1000)
            return jsonify(response)
        result_ids = list(job.result_ids)
        task = job.task
        query = job.query
        notice = job.notice
        query_profile = dict(job.query_profile)

    state = session_state(session_id)
    with SESSIONS_LOCK:
        payload = [
            as_payload(identifier, state.results[identifier])
            for identifier in result_ids
            if identifier in state.results
        ]
    return jsonify(
        {
            **response,
            "results": payload,
            "notice": notice,
            "query_profile": query_profile,
            "task": task,
            "query": query,
        }
    )


@app.get("/media/<identifier>")
def media(identifier: str):
    stored = current_session().results.get(identifier)
    if stored is None or not stored.result.image_path:
        return "Not found", 404
    path = Path(stored.result.image_path)
    if not path.is_file():
        return "Not found", 404
    return send_file(path, conditional=True, max_age=3600)


@app.get("/video/<identifier>")
def video(identifier: str):
    stored = current_session().results.get(identifier)
    if stored is None or not stored.result.video_path:
        return "Not found", 404
    path = Path(stored.result.video_path)
    if not path.is_file():
        return "Not found", 404
    return send_file(path, conditional=True, max_age=3600)


@app.post("/api/export")
def export():
    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict):
        return jsonify({"error": "JSON body phải là một object."}), 400
    selected = [str(value) for value in body.get("selected") or []]
    raw_overrides = body.get("frame_overrides") or {}
    raw_answer_overrides = body.get("answer_overrides") or {}
    state = current_session()
    selected_entries = [(identifier, state.results[identifier]) for identifier in selected if identifier in state.results]
    if not selected_entries:
        return jsonify({"error": "Chọn ít nhất một kết quả trước khi xuất CSV."}), 400
    frame_overrides: dict[str, int] = {}
    if not isinstance(raw_overrides, dict):
        return jsonify({"error": "Frame override phải là một JSON object."}), 400
    try:
        for identifier, _entry in track(
            selected_entries,
            desc="Kiểm tra frame override",
            total=len(selected_entries),
            unit="row",
        ):
            if identifier in raw_overrides:
                frame_overrides[identifier] = max(0, int(raw_overrides[identifier]))
    except (TypeError, ValueError):
        return jsonify({"error": "Frame override phải là số nguyên không âm."}), 400
    answer_overrides: dict[str, str] = {}
    if not isinstance(raw_answer_overrides, dict):
        return jsonify({"error": "Answer override phải là một JSON object."}), 400
    for identifier, _entry in track(
        selected_entries,
        desc="Kiểm tra answer override",
        total=len(selected_entries),
        unit="row",
    ):
        if identifier not in raw_answer_overrides:
            continue
        answer = str(raw_answer_overrides[identifier]).strip()
        if not answer or len(answer) > 200:
            return jsonify({"error": "Answer override phải có 1–200 ký tự."}), 400
        answer_overrides[identifier] = answer
    override_by_result = {
        id(entry.result): frame_overrides[identifier]
        for identifier, entry in selected_entries
        if identifier in frame_overrides
    }

    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    if state.task == "trake":
        groups = {entry.group for _identifier, entry in selected_entries if entry.group}
        sequences = [state.trake_sequences[group] for group in groups if group in state.trake_sequences]
        sequences.sort(key=lambda item: item.rank)
        width = max((len(item.frames) for item in sequences), default=0)
        writer.writerow(["video_id", *[f"frame_id_{index}" for index in range(1, width + 1)]])
        for item in track(
            sequences,
            desc="Xuất TRAKE CSV",
            total=len(sequences),
            unit="row",
            force=True,
        ):
            writer.writerow(
                [item.video_id, *[override_by_result.get(id(frame), frame.frame_id) for frame in item.frames]]
            )
        name = "aic_trake.csv"
    else:
        selected_entries.sort(key=lambda item: item[1].result.rank)
        is_qa = state.task == "qa"
        writer.writerow(["video_id", "frame_id", "answer"] if is_qa else ["video_id", "frame_id"])
        for identifier, entry in track(
            selected_entries,
            desc=f"Xuất {state.task.upper()} CSV",
            total=len(selected_entries),
            unit="row",
            force=True,
        ):
            row = [entry.result.video_id, frame_overrides.get(identifier, entry.result.frame_id)]
            if is_qa:
                row.append(answer_overrides.get(identifier, entry.result.answer))
            writer.writerow(row)
        name = f"aic_{state.task}.csv"
    return Response(
        stream.getvalue().encode("utf-8-sig"),
        content_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="AIC26 custom retrieval dashboard")
    parser.add_argument("--host", default=os.environ.get("AIC_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("AIC_PORT", "7860")))
    arguments = parser.parse_args()
    app.run(host=arguments.host, port=arguments.port, debug=False, threaded=False)


if __name__ == "__main__":
    main()
