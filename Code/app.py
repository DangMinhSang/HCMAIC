"""Three-tab Gradio web demo for the AIC 2026 preliminary tasks."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Sequence

import gradio as gr

from data_paths import DatasetNotFoundError
from qa import VQABaseline, split_qa_query
from retrieval import (
    AICRetrievalEngine,
    SearchResult,
    TrakeVideoResult,
    write_kis_submission,
    write_trake_submission,
)


ENGINE: AICRetrievalEngine | None = None
VQA: VQABaseline | None = None
TOP_K = 100
MIN_FRAME_GAP = 90
METADATA_WEIGHT = 0.10


def get_engine() -> AICRetrievalEngine:
    """Create the feature reader only when a user submits a query."""
    global ENGINE
    if ENGINE is None:
        ENGINE = AICRetrievalEngine.from_environment()
    return ENGINE


def get_vqa() -> VQABaseline:
    global VQA
    if VQA is None:
        VQA = VQABaseline()
    return VQA


def _output_file(name: str) -> Path:
    directory = Path(tempfile.gettempdir()) / "aic2026_demo_submission"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name


def _gallery(results: Sequence[SearchResult], limit: int = 48) -> list[tuple[str, str]]:
    # Keep the browser responsive while CSV/table retains all 100 answers.
    return [(result.image_path, result.caption()) for result in results[:limit] if result.image_path]


def _language_note(engine: AICRetrievalEngine) -> str:
    info = getattr(engine.encoder, "last_query", None)
    if info is None or info.language != "vi":
        return "Đã nhận diện truy vấn tiếng Anh."
    if info.translation_used:
        return f"Đã nhận diện tiếng Việt và dịch cho CLIP: `{info.text_for_model}`"
    return f"Đã nhận diện tiếng Việt. {info.warning}".strip()


def _search(engine: AICRetrievalEngine, query: str) -> list[SearchResult]:
    return engine.search(
        query,
        top_k=TOP_K,
        min_frame_gap=MIN_FRAME_GAP,
        metadata_weight=METADATA_WEIGHT,
    )


def run_kis(query: str):
    try:
        engine = get_engine()
        results = _search(engine, query)
        destination = write_kis_submission(results, _output_file("aic_textual_kis.csv"))
        status = (
            f"✅ Tìm được **{len(results)}** đáp án có thứ hạng trong {engine.video_count} video. "
            f"{_language_note(engine)} CSV có `video_id, frame_id` và giữ đủ tối đa 100 kết quả."
        )
        return status, _gallery(results), [result.table_row() for result in results], str(destination)
    except (DatasetNotFoundError, ValueError, RuntimeError, OSError) as error:
        return f"❌ Không thể truy vấn: {error}", [], [], None


def run_qa(query: str):
    """Localize first; VQA is a best-effort second stage and cannot break search."""
    try:
        event, question = split_qa_query(query)
        engine = get_engine()
        results = _search(engine, event)
        vqa_message = ""
        try:
            predictions = get_vqa().predict(question, results)
            by_rank = {item.rank: item for item in predictions}
            for result in results:
                prediction = by_rank.get(result.rank)
                if prediction is not None:
                    result.answer = prediction.answer
                    result.qa_confidence = prediction.confidence
            if predictions:
                # All candidates answer one question. Reuse top localized VQA
                # answer in CSV rows without a separate user input field.
                answer = predictions[0].answer
                confidence = predictions[0].confidence
                for result in results:
                    if not result.answer:
                        result.answer = answer
                vqa_message = f" Câu trả lời VQA đề xuất: **{answer}** ({confidence:.0%}, evidence top-1)."
            else:
                vqa_message = " Không có keyframe hợp lệ để chạy VQA."
        except RuntimeError as error:
            # The user can still inspect 100 localized frames even if the VQA
            # checkpoint is unavailable in a network-restricted Kaggle run.
            vqa_message = f" VQA chưa sẵn sàng: {error}"

        destination = write_kis_submission(
            results,
            _output_file("aic_qa.csv"),
            include_answer=any(result.answer for result in results),
        )
        rows = [result.table_row() + [result.answer, round(result.qa_confidence, 4)] for result in results]
        status = (
            f"✅ Đã định vị **{len(results)}** ứng viên Q&A. {_language_note(engine)}{vqa_message}"
        )
        return status, _gallery(results), rows, str(destination)
    except (DatasetNotFoundError, ValueError, RuntimeError, OSError) as error:
        return f"❌ Không thể truy vấn Q&A: {error}", [], [], None


def _events(value: str) -> list[str]:
    return [line.strip().lstrip("-0123456789. ") for line in (value or "").splitlines() if line.strip()]


def run_trake(events_text: str):
    try:
        engine = get_engine()
        results = engine.search_trake(_events(events_text), top_videos=10)
        destination = write_trake_submission(results, _output_file("aic_trake.csv"))
        rows = [row for item in results for row in item.table_rows()]
        frames = [frame for item in results for frame in item.frames]
        status = (
            f"✅ Đã truy xuất và căn chỉnh **{len(results)}** video TRAKE. "
            "Mỗi dòng input được tự nhận diện Việt/Anh và là một mốc semantic theo thứ tự."
        )
        return status, _gallery(frames), rows, str(destination)
    except (DatasetNotFoundError, ValueError, RuntimeError, OSError) as error:
        return f"❌ Không thể truy vấn TRAKE: {error}", [], [], None


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="AIC 2026 Retrieval Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # AIC 2026 — Video Retrieval Demo

            Chọn đúng loại câu hỏi bên dưới. Mỗi tab chỉ có **một ô truy vấn**;
            hệ thống tự nhận diện tiếng Việt/Anh. Với tiếng Việt, demo tự dịch sang
            tiếng Anh để khớp CLIP ViT-B/32, và vẫn truy vấn câu gốc nếu dịch không sẵn sàng.
            Dữ liệu chỉ được đọc từ Kaggle Input — không tải dataset.
            """
        )

        with gr.Tab("1. Textual KIS"):
            kis_query = gr.Textbox(
                label="Mô tả sự kiện (Việt hoặc English)",
                lines=5,
                placeholder="Ví dụ: Tìm cảnh một người mở laptop trong văn phòng.",
            )
            kis_button = gr.Button("Tìm Textual KIS", variant="primary")
            kis_status = gr.Markdown()
            kis_gallery = gr.Gallery(label="Top keyframes", columns=4, height="auto")
            kis_table = gr.Dataframe(
                headers=["rank", "video_id", "frame_id", "keyframe", "time_s", "clip", "metadata", "score", "title", "objects"],
                interactive=False,
                label="Tối đa 100 đáp án — dùng video_id và frame_id để nộp",
            )
            kis_file = gr.File(label="CSV Textual KIS")
            kis_button.click(run_kis, inputs=kis_query, outputs=[kis_status, kis_gallery, kis_table, kis_file])

        with gr.Tab("2. Q&A"):
            qa_query = gr.Textbox(
                label="Mô tả sự kiện và câu hỏi (Việt hoặc English)",
                lines=5,
                placeholder="Cảnh lễ trao giải âm nhạc. Câu hỏi: Có bao nhiêu người trên sân khấu?",
            )
            qa_button = gr.Button("Tìm Q&A", variant="primary")
            qa_status = gr.Markdown()
            qa_gallery = gr.Gallery(label="Keyframes Q&A", columns=4, height="auto")
            qa_table = gr.Dataframe(
                headers=["rank", "video_id", "frame_id", "keyframe", "time_s", "clip", "metadata", "score", "title", "objects", "VQA answer", "VQA confidence"],
                interactive=False,
                label="Định vị sự kiện và câu trả lời VQA",
            )
            qa_file = gr.File(label="CSV Q&A")
            qa_button.click(run_qa, inputs=qa_query, outputs=[qa_status, qa_gallery, qa_table, qa_file])

        with gr.Tab("3. TRAKE"):
            trake_events = gr.Textbox(
                label="Chuỗi event TRAKE — mỗi dòng một mốc, Việt hoặc English",
                lines=7,
                placeholder="Vận động viên chạy đà\nVận động viên giậm nhảy\nVận động viên bay qua xà\nVận động viên tiếp đất trên đệm",
            )
            trake_button = gr.Button("Truy xuất & căn chỉnh TRAKE", variant="primary")
            trake_status = gr.Markdown()
            trake_gallery = gr.Gallery(label="Semantic keyframes", columns=4, height="auto")
            trake_table = gr.Dataframe(
                headers=["video_rank", "video_id", "event", "frame_id", "keyframe", "time_s", "event_clip", "video_score", "title"],
                interactive=False,
                label="Kết quả TRAKE",
            )
            trake_file = gr.File(label="CSV TRAKE")
            trake_button.click(run_trake, inputs=trake_events, outputs=[trake_status, trake_gallery, trake_table, trake_file])

        gr.Markdown(
            "CSV giữ nguyên thứ tự xếp hạng. Khi BTC phát hành template nộp chính thức, giữ nguyên "
            "`video_id` và `frame_id`, chỉ điều chỉnh header nếu được yêu cầu."
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("AIC_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("AIC_PORT", "7860")))
    parser.add_argument("--share", action="store_true", help="Tạo Gradio share link (hữu ích trên Kaggle).")
    arguments = parser.parse_args()
    build_demo().queue(default_concurrency_limit=1).launch(
        server_name=arguments.host,
        server_port=arguments.port,
        share=arguments.share,
    )


if __name__ == "__main__":
    main()
