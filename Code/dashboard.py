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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, render_template, request, send_file, session, url_for

from data_paths import DatasetNotFoundError
from multimodal_reranker import (
    MultimodalRerankerUnavailableError,
    QwenVLQueryReranker,
)
from ocr_index import OCRMemoryIndex
from qa import VQABaseline, split_qa_query
from query_language import normalize_query
from ranking import normalized_scores, select_diverse_results, select_multisource_candidates
from retrieval import AICRetrievalEngine, SearchResult, TrakeVideoResult


app = Flask(__name__)
app.secret_key = os.environ.get("AIC_WEB_SECRET", secrets.token_urlsafe(32))

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
        for video_id in engine._features:
            mapping = engine._mapping(video_id)
            if not mapping:
                continue
            sample = engine.result_for_keyframe(video_id, mapping[0].keyframe_number)
            if sample is not None and sample.image_path:
                reranker.score("a representative video frame", [sample])
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
        return OCR_INDEX


def current_session() -> SearchSession:
    identifier = session.get("aic_session")
    if not identifier:
        identifier = uuid.uuid4().hex
        session["aic_session"] = identifier
    with SESSIONS_LOCK:
        return SESSIONS.setdefault(identifier, SearchSession())


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
        "ocr_text": getattr(result, "ocr_text", ""),
        "ai_score": round(getattr(result, "rerank_score", 0.0), 3),
        "ai_joint_score": round(getattr(result, "rerank_joint_score", 0.0), 3),
        "ai_visual_score": round(getattr(result, "rerank_visual_score", 0.0), 3),
        "ai_ocr_score": round(getattr(result, "rerank_ocr_score", 0.0), 3),
        "event": stored.event_index,
        # Include the WSGI mount prefix when the app is exposed under
        # /dashboard by the Kaggle Gradio share gateway.
        "image_url": url_for("media", identifier=identifier),
    }


def search_options(body: dict[str, Any]) -> tuple[int, int, int | None, str | None]:
    options = body.get("options") or {}
    top_k = max(1, min(int(options.get("top_k", 100)), 100))
    min_gap = max(0, min(int(options.get("dedupe", 0)), 600))
    max_per_video_raw = int(options.get("max_per_video", 4))
    max_per_video = max(1, min(max_per_video_raw, 100)) if max_per_video_raw else None
    video_id = str(options.get("video_id") or "").strip() or None
    return top_k, min_gap, max_per_video, video_id


def make_kis_results(engine: AICRetrievalEngine, query: str, body: dict[str, Any]) -> tuple[list[StoredResult], str]:
    top_k, min_gap, maximum, video_id = search_options(body)
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
            metadata_weight=0.10,
        )
    except TypeError as error:
        # A running Kaggle kernel can retain an older retrieval module in
        # sys.modules after git pull. Keep the dashboard usable until restart.
        if "unexpected keyword" not in str(error):
            raise
        results = engine.search(query, top_k=recall_top_k, min_frame_gap=recall_min_gap, metadata_weight=0.10)
        if video_id:
            results = [result for result in results if result.video_id == video_id]
        if maximum and not reranker_requested:
            counts: dict[str, int] = {}
            filtered = []
            for result in results:
                counts[result.video_id] = counts.get(result.video_id, 0) + 1
                if counts[result.video_id] <= maximum:
                    filtered.append(result)
            results = filtered
    ocr_index = get_ocr_index()
    hits = ocr_index.search(query, limit=max(200, min(top_k * 4, 400)), video_id=video_id) if ocr_index else []
    combined = {(result.video_id, result.keyframe_number): result for result in results}
    for hit in hits:
        key = (hit.video_id, hit.keyframe_number)
        existing = combined.get(key)
        if existing is None:
            existing = engine.result_for_keyframe(
                hit.video_id,
                hit.keyframe_number,
                # OCR is useful evidence, but must not overpower the visual
                # signal before the multimodal reranker gets a chance to
                # inspect the actual frame.
                score=0.55 * hit.score,
                ocr_score=hit.score,
                ocr_text=hit.text,
            )
            if existing is None:
                continue
            combined[key] = existing
        else:
            existing.ocr_score = max(existing.ocr_score, hit.score)
            existing.ocr_text = hit.text
            existing.score = max(
                existing.score,
                0.40 * existing.visual_score + 0.45 * hit.score + 0.10 * existing.metadata_score,
            )
            existing.retrieval_score = existing.score

    rerank_note = ""
    reranker = get_reranker()
    if reranker and combined:
        # Qwen is a cross-attention reranker, not a corpus scanner. Keep the
        # expensive pass bounded while ensuring OCR hits can enter the pool.
        try:
            # Recall still contains up to 100 candidates. A single Qwen
            # multimodal pass over 24 is accurate enough for the final Top-K
            # while keeping a synchronous /api/search request below the
            # Gradio gateway timeout. Increase explicitly when needed.
            rerank_budget = max(1, min(int(os.environ.get("AIC_RERANKER_CANDIDATES", "24")), 100))
        except ValueError:
            rerank_budget = 24
        rerank_limit = min(rerank_budget, len(combined))
        rerank_candidates = select_multisource_candidates(results, hits, combined, rerank_limit)
        try:
            normalized = normalize_query(query)
            rerank_query = query
            if normalized.text_for_model.lower() != query.lower():
                rerank_query = f"{query}\nEnglish translation: {normalized.text_for_model}"
            rerank_scores = reranker.score(rerank_query, rerank_candidates)
            for result, rerank_score in zip(rerank_candidates, rerank_scores):
                result.rerank_score = rerank_score.final
                result.rerank_joint_score = rerank_score.joint
                result.rerank_visual_score = rerank_score.visual
                result.rerank_ocr_score = rerank_score.ocr
                result.score = rerank_score.final
            # Candidates outside the reranker budget remain available as a
            # diversity fallback, but cannot outrank model-scored candidates.
            if len(rerank_candidates) < len(combined):
                candidate_ids = {id(result) for result in rerank_candidates}
                floor = min(result.score for result in rerank_candidates)
                for result in combined.values():
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
    ocr_note = f"OCR RAM: {len(hits)} keyframe khớp chữ." if ocr_index else "OCR index chưa được nạp."
    return [StoredResult(result) for result in selected], language_note(engine) + " · " + ocr_note + rerank_note


def make_qa_results(engine: AICRetrievalEngine, query: str, body: dict[str, Any]) -> tuple[list[StoredResult], str]:
    event, question = split_qa_query(query)
    stored, note = make_kis_results(engine, event, body)
    results = [item.result for item in stored]
    try:
        predictions = get_vqa().predict(question, results)
        by_rank = {prediction.rank: prediction for prediction in predictions}
        for result in results:
            prediction = by_rank.get(result.rank)
            if prediction:
                result.answer = prediction.answer
                result.qa_confidence = prediction.confidence
        if predictions:
            answer_scores: dict[str, float] = {}
            answer_labels: dict[str, str] = {}
            for prediction in predictions:
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
            for result in results:
                if not result.answer:
                    result.answer = consensus_answer
                    result.qa_confidence = best_confidence
            note += f" · VQA đồng thuận: {consensus_answer} ({best_confidence:.0%})"
        else:
            note += " · Không có ảnh để chạy VQA."
    except RuntimeError as error:
        # Retrieval must remain available if the optional VQA checkpoint is offline.
        note += f" · VQA chưa sẵn sàng: {error}"
    return stored, note


def make_trake_results(engine: AICRetrievalEngine, query: str, body: dict[str, Any]) -> tuple[list[StoredResult], dict[str, TrakeVideoResult], str]:
    events = [line.strip().lstrip("-0123456789. ") for line in query.splitlines() if line.strip()]
    top_k, _gap, _maximum, _video_id = search_options(body)
    sequences = engine.search_trake(events, top_videos=min(top_k, 100))
    rerank_note = ""
    reranker = get_reranker()
    if reranker and sequences:
        try:
            pair_budget = max(1, min(int(os.environ.get("AIC_TRAKE_RERANK_PAIRS", "24")), 100))
        except ValueError:
            pair_budget = 24
        sequence_limit = min(len(sequences), max(1, pair_budget // len(events)))
        reranked_sequences = sequences[:sequence_limit]
        pair_queries: list[str] = []
        pair_frames: list[SearchResult] = []
        for sequence in reranked_sequences:
            for event, frame in zip(events, sequence.frames):
                normalized = normalize_query(event)
                pair_queries.append(
                    event
                    if normalized.text_for_model.lower() == event.lower()
                    else f"{event}\nEnglish translation: {normalized.text_for_model}"
                )
                pair_frames.append(frame)
        try:
            pair_scores = reranker.score_pairs(
                pair_queries,
                pair_frames,
                prompt=(
                    "Score whether this exact video frame is the requested semantic moment. "
                    "Use the visible action, temporal stage, objects, and relevant text."
                ),
            )
            clip_scaled = normalized_scores([sequence.score for sequence in sequences])
            offset = 0
            for index, sequence in enumerate(sequences):
                if index < sequence_limit:
                    values = pair_scores[offset : offset + len(events)]
                    offset += len(events)
                    model_score = 0.72 * (sum(values) / len(values)) + 0.28 * min(values)
                    sequence.score = 0.85 * model_score + 0.15 * clip_scaled[index]
                    for frame, value in zip(sequence.frames, values):
                        frame.rerank_score = value
                        frame.rerank_joint_score = value
                        frame.score = value
                else:
                    sequence.score = 0.35 * clip_scaled[index]
            sequences.sort(key=lambda item: item.score, reverse=True)
            for rank, sequence in enumerate(sequences, start=1):
                sequence.rank = rank
                for frame in sequence.frames:
                    frame.rank = rank
            rerank_note = f" · Qwen: {len(pair_frames)} cặp event–frame"
        except MultimodalRerankerUnavailableError as error:
            rerank_note = f" · Qwen TRAKE fallback: {str(error)[:120]}"
    stored: list[StoredResult] = []
    indexed: dict[str, TrakeVideoResult] = {}
    for sequence in sequences:
        group = f"trake-{sequence.rank}"
        indexed[group] = sequence
        for event_index, frame in enumerate(sequence.frames, start=1):
            stored.append(StoredResult(frame, group=group, event_index=event_index))
    return (
        stored,
        indexed,
        "TRAKE căn chỉnh có thứ tự thời gian; chọn một card để xuất cả chuỗi video."
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
        return jsonify(
            {
                "ok": True,
                "video_count": engine.video_count,
                "vector_count": vector_count_compat(engine),
                "video_ids": sorted(engine._features),
                "ocr_records": get_ocr_index().record_count if get_ocr_index() else 0,
                "feature_cache_loaded": engine.feature_cache_loaded,
                "reranker_ready": RERANKER is not None,
                "reranker_error": RERANKER_ERROR,
            }
        )
    except (DatasetNotFoundError, OSError) as error:
        return jsonify({"ok": False, "error": str(error)}), 503


@app.post("/api/search")
def search():
    started = time.perf_counter()
    body = request.get_json(silent=True) or {}
    task = str(body.get("task") or "kis").lower()
    query = str(body.get("query") or "").strip()
    if task not in {"kis", "qa", "trake"}:
        return jsonify({"error": "Loại truy vấn không hợp lệ."}), 400
    if not query:
        return jsonify({"error": "Hãy nhập truy vấn."}), 400
    try:
        engine = get_engine()
        with ENGINE_LOCK:
            if task == "kis":
                stored, notice = make_kis_results(engine, query, body)
                sequences: dict[str, TrakeVideoResult] = {}
            elif task == "qa":
                stored, notice = make_qa_results(engine, query, body)
                sequences = {}
            else:
                stored, sequences, notice = make_trake_results(engine, query, body)
        state = current_session()
        state.task = task
        state.results.clear()
        state.trake_sequences = sequences
        payload: list[dict[str, Any]] = []
        for item in stored:
            identifier = uuid.uuid4().hex
            state.results[identifier] = item
            payload.append(as_payload(identifier, item))
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        return jsonify({"results": payload, "elapsed_ms": elapsed_ms, "notice": notice, "task": task})
    except (DatasetNotFoundError, ValueError, RuntimeError, OSError) as error:
        return jsonify({"error": str(error)}), 400


@app.get("/media/<identifier>")
def media(identifier: str):
    stored = current_session().results.get(identifier)
    if stored is None or not stored.result.image_path:
        return "Not found", 404
    path = Path(stored.result.image_path)
    if not path.is_file():
        return "Not found", 404
    return send_file(path, conditional=True, max_age=3600)


@app.post("/api/export")
def export():
    body = request.get_json(silent=True) or {}
    selected = [str(value) for value in body.get("selected") or []]
    state = current_session()
    entries = [state.results[item] for item in selected if item in state.results]
    if not entries:
        return jsonify({"error": "Chọn ít nhất một kết quả trước khi xuất CSV."}), 400

    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    if state.task == "trake":
        groups = {entry.group for entry in entries if entry.group}
        sequences = [state.trake_sequences[group] for group in groups if group in state.trake_sequences]
        sequences.sort(key=lambda item: item.rank)
        width = max((len(item.frames) for item in sequences), default=0)
        writer.writerow(["video_id", *[f"frame_id_{index}" for index in range(1, width + 1)]])
        for item in sequences:
            writer.writerow([item.video_id, *[frame.frame_id for frame in item.frames]])
        name = "aic_trake.csv"
    else:
        entries.sort(key=lambda item: item.result.rank)
        is_qa = state.task == "qa"
        writer.writerow(["video_id", "frame_id", "answer"] if is_qa else ["video_id", "frame_id"])
        for entry in entries:
            row = [entry.result.video_id, entry.result.frame_id]
            if is_qa:
                row.append(entry.result.answer)
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
