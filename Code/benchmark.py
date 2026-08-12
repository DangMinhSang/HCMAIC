"""Benchmark the warm AIC query path on the mounted Kaggle runtime."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from typing import Any

import dashboard
from progress import track
from qa import split_qa_query
from query_router import build_query_profile


DEFAULT_QUERIES = {
    "kis": "Biển màu vàng có nội dung cảnh báo sạt lở nguy hiểm",
    "qa": "Một nhóm người đứng trên sân khấu. Câu hỏi: Có bao nhiêu người?",
    "trake": "Người đặt laptop lên bàn\nNgười mở nắp laptop\nMàn hình laptop sáng lên",
}


def result_preview(task: str, stored: list[dashboard.StoredResult]) -> dict[str, Any] | None:
    if not stored:
        return None
    result = stored[0].result
    preview: dict[str, Any] = {
        "video_id": result.video_id,
        "frame_id": result.frame_id,
        "score": round(float(result.score), 4),
    }
    if task == "qa":
        preview["answer"] = result.answer
        preview["qa_confidence"] = round(float(result.qa_confidence), 4)
    return preview


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the prewarmed AIC 2026 pipeline")
    parser.add_argument("--task", choices=("kis", "qa", "trake"), default="kis")
    parser.add_argument("--query", help="Query text; TRAKE uses newline-separated events")
    parser.add_argument("--repeat", type=int, default=2, help="Number of warm query runs (1..10)")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--skip-model-warmup", action="store_true")
    arguments = parser.parse_args()

    query = (arguments.query or DEFAULT_QUERIES[arguments.task]).strip()
    repeat = max(1, min(arguments.repeat, 10))
    top_k = max(1, min(arguments.top_k, 100))
    body = {"options": {"top_k": top_k, "dedupe": 90, "max_per_video": 4}}

    os.environ.setdefault("AIC_DATA_ROOT", "/kaggle/input")
    os.environ.setdefault("AIC_OCR_INDEX", "/kaggle/working/aic_ocr_index.jsonl.gz")
    os.environ.setdefault("AIC_PRELOAD_FEATURES", "1")
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        os.environ.setdefault("AIC_RERANKER", "1")

    startup_started = time.perf_counter()
    engine = dashboard.get_engine()
    engine.prepare_runtime()
    ocr_index = dashboard.get_ocr_index()
    if not arguments.skip_model_warmup:
        dashboard.warmup_reranker()
        if arguments.task == "qa":
            dashboard.warmup_vqa()
    startup_seconds = time.perf_counter() - startup_started

    reranker = dashboard.get_reranker()
    if arguments.task == "qa":
        event, question = split_qa_query(query)
        profile_query = (
            event
            if event.casefold() == question.casefold()
            else f"{event}\nVisual question: {question}"
        )
    else:
        profile_query = query
    profile = build_query_profile(profile_query, reranker)
    runs: list[dict[str, Any]] = []
    run_indices = range(1, repeat + 1)
    for run_index in track(
        run_indices,
        desc=f"Benchmark {arguments.task.upper()}",
        total=len(run_indices),
        unit="query",
        force=True,
        leave=True,
    ):
        started = time.perf_counter()
        with dashboard.ENGINE_LOCK:
            if arguments.task == "kis":
                stored, notice = dashboard.make_kis_results(
                    engine, query, body, profile=profile, reranker=reranker
                )
            elif arguments.task == "qa":
                stored, notice = dashboard.make_qa_results(
                    engine, query, body, profile=profile, reranker=reranker
                )
            else:
                stored, _sequences, notice = dashboard.make_trake_results(
                    engine, query, body, profile=profile, reranker=reranker
                )
        elapsed = time.perf_counter() - started
        runs.append(
            {
                "run": run_index,
                "seconds": round(elapsed, 3),
                "result_count": len(stored),
                "top_1": result_preview(arguments.task, stored),
                "notice": notice,
            }
        )

    timings = [run["seconds"] for run in runs]
    report = {
        "task": arguments.task,
        "query": query,
        "startup_seconds": round(startup_seconds, 3),
        "warm_query_mean_seconds": round(statistics.fmean(timings), 3),
        "warm_query_min_seconds": round(min(timings), 3),
        "corpus": {
            "videos": engine.video_count,
            "vectors": engine.vector_count,
            "ocr_records": ocr_index.record_count if ocr_index else 0,
            "features_in_ram": engine.feature_cache_loaded,
        },
        "models": {
            "reranker": reranker.model_name if reranker is not None else "fallback",
            "reranker_error": dashboard.RERANKER_ERROR,
            "vqa": dashboard.VQA.backend_name if dashboard.VQA is not None else "not used",
        },
        "query_profile": profile.as_dict(),
        "runs": runs,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
