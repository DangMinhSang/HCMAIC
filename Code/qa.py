"""A lazy visual-QA baseline for the AIC Q&A tab."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from query_language import normalize_query
from retrieval import SearchResult


QUESTION_RE = re.compile(r"(?:câu\s*hỏi|question)\s*[:\-]\s*(.+)$", flags=re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class QAPrediction:
    rank: int
    answer: str
    confidence: float


def split_qa_query(value: str) -> tuple[str, str]:
    """Split one Q&A field into event description and question.

    The marker makes retrieval stable: CLIP receives the event, while the VQA
    model receives only the question. Example: ``Cảnh trao giải. Câu hỏi: Có
    bao nhiêu người trên sân khấu?``
    """
    value = (value or "").strip()
    match = QUESTION_RE.search(value)
    if not match:
        # Do not reject a natural one-field question. Using the whole sentence
        # is less precise than the explicit marker, but it keeps the demo
        # usable for queries such as "Trong cảnh ... có bao nhiêu người?".
        if value:
            return value, value
        raise ValueError("Nhập mô tả và câu hỏi Q&A.")
    question = match.group(1).strip()
    event = value[: match.start()].strip(" .:-\n")
    if not event or not question:
        raise ValueError("Q&A cần cả mô tả sự kiện và câu hỏi.")
    return event, question


class VQABaseline:
    """Run ViLT VQA only after retrieval has found candidate keyframes."""

    def __init__(self, model_name: str = "dandelin/vilt-b32-finetuned-vqa") -> None:
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._torch = None
        self.device = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # type: ignore
            from transformers import ViltForQuestionAnswering, ViltProcessor  # type: ignore
        except ImportError as error:
            raise RuntimeError("Thiếu transformers/Pillow cho Q&A. Chạy lại cell cài requirements.") from error
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self._processor = ViltProcessor.from_pretrained(self.model_name)
            self._model = ViltForQuestionAnswering.from_pretrained(self.model_name).to(self.device).eval()
        except Exception as error:
            raise RuntimeError(
                "Không tải được model VQA. Bật Internet lần đầu hoặc cache model trước; "
                "truy xuất keyframe vẫn hoạt động."
            ) from error
        self._torch = torch

    def predict(self, question: str, results: Sequence[SearchResult], limit: int = 8) -> list[QAPrediction]:
        self._load()
        try:
            from PIL import Image  # type: ignore
        except ImportError as error:
            raise RuntimeError("Thiếu Pillow cho Q&A.") from error
        normalized = normalize_query(question)
        predictions: list[QAPrediction] = []
        for result in results[:limit]:
            if not result.image_path or not Path(result.image_path).is_file():
                continue
            with Image.open(result.image_path) as image:
                inputs = self._processor(images=image.convert("RGB"), text=normalized.text_for_model, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self._torch.no_grad():
                logits = self._model(**inputs).logits[0]
                probabilities = logits.softmax(dim=-1)
                index = int(probabilities.argmax())
            predictions.append(
                QAPrediction(
                    rank=result.rank,
                    answer=str(self._model.config.id2label[index]),
                    confidence=float(probabilities[index]),
                )
            )
        return predictions
