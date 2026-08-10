"""Gradio web demo for AIC 2026 Textual KIS, Q&A, and TRAKE."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Sequence

import gradio as gr

from data_paths import DatasetNotFoundError
from retrieval import (
    AICRetrievalEngine,
    SearchResult,
    TrakeVideoResult,
    write_kis_submission,
    write_trake_submission,
)


ENGINE: AICRetrievalEngine | None = None


def get_engine() -> AICRetrievalEngine:
    """Create the engine lazily so opening the UI does not touch the dataset."""
    global ENGINE
    if ENGINE is None:
        ENGINE = AICRetrievalEngine.from_environment()
    return ENGINE


def _output_file(name: str) -> Path:
    directory = Path(tempfile.gettempdir()) / "aic2026_demo_submission"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name


def _gallery(results: Sequence[SearchResult]) -> list[tuple[str, str]]:
    return [(result.image_path, result.caption()) for result in results if result.image_path]


def run_kis_or_qa(
    query: str,
    english_expansion: str,
    answer: str,
    top_k: int,
    min_frame_gap: int,
    metadata_weight: float,
):
    try:
        engine = get_engine()
        results = engine.search(
            query,
            english_expansion,
            top_k=int(top_k),
            min_frame_gap=int(min_frame_gap),
            metadata_weight=float(metadata_weight),
        )
        destination = write_kis_submission(
            results, _output_file("aic_kis_or_qa_submission.csv"), answer
        )
        title = "Q&A" if answer.strip() else "Textual KIS"
        status = (
            f"Đã tìm **{len(results)}** ứng viên trong {engine.video_count} video "
            f"({title}). File CSV giữ nguyên thứ tự xếp hạng và có tối đa 100 đáp án."
        )
        return status, _gallery(results), [result.table_row() for result in results], str(destination)
    except (DatasetNotFoundError, ValueError, RuntimeError, OSError) as error:
        return f"❌ {error}", [], [], None


def _events(value: str) -> list[str]:
    return [line.strip().lstrip("-0123456789. ") for line in (value or "").splitlines() if line.strip()]


def run_trake(events_text: str, english_events_text: str, top_videos: int):
    try:
        engine = get_engine()
        results = engine.search_trake(
            _events(events_text),
            _events(english_events_text),
            top_videos=int(top_videos),
        )
        destination = write_trake_submission(results, _output_file("aic_trake_submission.csv"))
        rows = [row for item in results for row in item.table_rows()]
        frames = [frame for item in results for frame in item.frames]
        status = (
            f"Đã truy xuất và căn chỉnh **{len(results)}** video TRAKE. "
            "Mỗi hàng trong CSV là một video cùng frame_id của từng event theo đúng thứ tự."
        )
        return status, _gallery(frames), rows, str(destination)
    except (DatasetNotFoundError, ValueError, RuntimeError, OSError) as error:
        return f"❌ {error}", [], [], None


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="AIC 2026 Retrieval Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # AIC 2026 — Video Retrieval Demo

            Tìm **Textual KIS**, hỗ trợ xác minh **Q&A**, và căn chỉnh **TRAKE** bằng
            CLIP ViT-B/32 khớp với feature AIC đã cung cấp. Dữ liệu luôn được đọc từ
            Kaggle Input; ứng dụng không tải hay sao chép dataset.

            Với truy vấn tiếng Việt, hãy điền một bản diễn đạt tiếng Anh chính xác để
            tăng chất lượng CLIP. Kết quả được đa dạng theo thời gian để hữu ích cho
            danh sách tối đa 100 câu trả lời của vòng sơ tuyển.
            """
        )

        with gr.Tab("Textual KIS / Q&A"):
            with gr.Row():
                with gr.Column(scale=3):
                    query = gr.Textbox(
                        label="Mô tả truy vấn",
                        lines=4,
                        placeholder="Ví dụ: Tìm cảnh một người mở laptop trong văn phòng.",
                    )
                    english = gr.Textbox(
                        label="English expansion — khuyến nghị cho độ chính xác",
                        lines=3,
                        placeholder="A person opening a laptop in an office.",
                    )
                    answer = gr.Textbox(
                        label="Câu trả lời Q&A (tùy chọn)",
                        placeholder="Nhập sau khi kiểm tra keyframe/object; để trống cho Textual KIS.",
                    )
                with gr.Column(scale=1):
                    top_k = gr.Slider(1, 100, value=50, step=1, label="Số đáp án xuất")
                    frame_gap = gr.Slider(
                        0, 300, value=90, step=1, label="Khoảng cách frame tối thiểu cùng video"
                    )
                    metadata_weight = gr.Slider(
                        0, 0.35, value=0.10, step=0.01, label="Trọng số metadata"
                    )
                    search_button = gr.Button("Tìm kiếm", variant="primary")
            kis_status = gr.Markdown()
            kis_gallery = gr.Gallery(label="Keyframe xếp hạng", columns=4, height="auto")
            kis_table = gr.Dataframe(
                headers=[
                    "rank", "video_id", "frame_id", "keyframe", "time_s", "clip", "metadata",
                    "score", "title", "objects",
                ],
                datatype=["number", "str", "number", "number", "number", "number", "number", "number", "str", "str"],
                interactive=False,
                label="Kết quả — dùng video_id và frame_id để nộp",
            )
            kis_file = gr.File(label="CSV nộp bài")
            search_button.click(
                run_kis_or_qa,
                inputs=[query, english, answer, top_k, frame_gap, metadata_weight],
                outputs=[kis_status, kis_gallery, kis_table, kis_file],
            )

        with gr.Tab("TRAKE — temporal alignment"):
            gr.Markdown(
                "Nhập **mỗi mốc ngữ nghĩa một dòng**, đúng thứ tự thời gian. Hệ thống chọn video có "
                "đủ mọi mốc rồi chọn một semantic keyframe cho từng mốc."
            )
            with gr.Row():
                trake_events = gr.Textbox(
                    label="Các event của truy vấn TRAKE", lines=7,
                    placeholder="1. Athlete runs toward the high-jump bar\n2. Athlete takes off\n3. Athlete clears the bar\n4. Athlete lands on the mat",
                )
                trake_english = gr.Textbox(
                    label="English expansion của từng event (cùng số dòng, tùy chọn)", lines=7,
                    placeholder="Có thể để trống nếu event đã viết bằng tiếng Anh.",
                )
            with gr.Row():
                trake_top = gr.Slider(1, 50, value=10, step=1, label="Số video TRAKE xuất")
                trake_button = gr.Button("Truy xuất & căn chỉnh", variant="primary")
            trake_status = gr.Markdown()
            trake_gallery = gr.Gallery(label="Semantic keyframes", columns=4, height="auto")
            trake_table = gr.Dataframe(
                headers=["video_rank", "video_id", "event", "frame_id", "keyframe", "time_s", "event_clip", "video_score", "title"],
                datatype=["number", "str", "number", "number", "number", "number", "number", "number", "str"],
                interactive=False,
                label="Kết quả TRAKE",
            )
            trake_file = gr.File(label="CSV nộp TRAKE")
            trake_button.click(
                run_trake,
                inputs=[trake_events, trake_english, trake_top],
                outputs=[trake_status, trake_gallery, trake_table, trake_file],
            )

        gr.Markdown(
            """
            **Lưu ý chấm điểm:** CSV chỉ là định dạng làm việc. Hãy đối chiếu template nộp bài
            chính thức khi BTC phát hành. Với Q&A, demo định vị sự kiện và hiển thị object/keyframe;
            người dùng nhập câu trả lời ngữ nghĩa vào ô Q&A trước khi tải CSV.
            """
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
