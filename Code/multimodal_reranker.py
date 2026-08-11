"""Optional Qwen3-VL reranking for query/image/OCR candidates.

The CLIP index is still used for fast recall.  This module only runs on a
small candidate set after recall, so the 2B multimodal model does not need to
scan the whole AIC corpus for every query.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Sequence


DEFAULT_MODEL = "Qwen/Qwen3-VL-Reranker-2B"
VISUAL_PROMPT = "Retrieve video frames whose visual scene matches the user's query."
OCR_PROMPT = "Retrieve video frames whose visible or transcribed text matches the user's query. Ignore generic unrelated text."
JOINT_PROMPT = "Retrieve video frames that match the query in both visual context and relevant visible text. Penalize generic scenes or unrelated OCR."


class MultimodalRerankerUnavailableError(RuntimeError):
    """Raised when the optional GPU reranker cannot be loaded."""


@dataclass(frozen=True)
class RerankScore:
    visual: float
    ocr: float
    joint: float

    @property
    def final(self) -> float:
        """Fuse calibrated model scores while keeping visual context primary."""
        return 0.63 * self.joint + 0.22 * self.visual + 0.15 * self.ocr


class QwenVLQueryReranker:
    """Score a query against candidate frames with Qwen3-VL-Reranker-2B."""

    def __init__(self, model_name: str | None = None) -> None:
        try:
            import torch  # type: ignore
            from sentence_transformers import CrossEncoder  # type: ignore
        except ImportError as error:
            raise MultimodalRerankerUnavailableError(
                "Thiếu sentence-transformers/qwen-vl-utils cho Qwen3-VL-Reranker. "
                "Chạy lại requirements.txt hoặc tắt AIC_RERANKER."
            ) from error

        if not torch.cuda.is_available() and os.environ.get("AIC_RERANKER_CPU", "0").lower() not in {
            "1",
            "true",
            "yes",
        }:
            raise MultimodalRerankerUnavailableError(
                "Qwen3-VL-Reranker cần GPU. Bật Kaggle Accelerator = GPU "
                "hoặc đặt AIC_RERANKER_CPU=1 để chạy thử bằng CPU."
            )

        self._torch = torch
        self.model_name = model_name or os.environ.get("AIC_RERANKER_MODEL", DEFAULT_MODEL)
        device = os.environ.get("AIC_RERANKER_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model = CrossEncoder(self.model_name, device=device)
        except Exception as error:
            raise MultimodalRerankerUnavailableError(
                f"Không tải được Qwen reranker `{self.model_name}`: {error}"
            ) from error

    @staticmethod
    def _image_document(result: Any) -> dict[str, str] | None:
        image_path = str(getattr(result, "image_path", "") or "")
        return {"image": image_path} if image_path else None

    @classmethod
    def _joint_document(cls, result: Any) -> dict[str, str] | None:
        document: dict[str, str] = {}
        image = cls._image_document(result)
        ocr_text = str(getattr(result, "ocr_text", "") or "").strip()
        if image:
            document.update(image)
        if ocr_text:
            document["text"] = ocr_text
        return document or None

    def _predict(self, query: str, documents: list[Any], prompt: str) -> list[float]:
        pairs = [(query, document) for document in documents]
        if not pairs:
            return []
        try:
            values = self.model.predict(
                pairs,
                batch_size=1,
                show_progress_bar=False,
                activation_fn=self._torch.nn.Sigmoid(),
                prompt=prompt,
            )
        except Exception as error:
            raise MultimodalRerankerUnavailableError(
                f"Qwen reranker không chấm được candidate: {error}"
            ) from error
        return [max(0.0, min(1.0, float(value))) for value in values]

    def score(self, query: str, results: Sequence[Any]) -> list[RerankScore]:
        """Return visual/OCR/joint probabilities for each result.

        The three passes intentionally isolate evidence types.  The joint
        pass sees both image and OCR and receives the largest final weight.
        """
        image_documents = [self._image_document(result) for result in results]
        joint_documents = [self._joint_document(result) for result in results]
        visual_documents = [document or {"text": "No image available."} for document in image_documents]
        joint_documents = [document or {"text": "No image or OCR evidence available."} for document in joint_documents]

        visual_scores = self._predict(query, visual_documents, VISUAL_PROMPT)
        joint_scores = self._predict(query, joint_documents, JOINT_PROMPT)

        ocr_indices = [index for index, result in enumerate(results) if str(getattr(result, "ocr_text", "") or "").strip()]
        ocr_scores = [0.0] * len(results)
        if ocr_indices:
            ocr_documents = [{"text": str(getattr(results[index], "ocr_text", "")).strip()} for index in ocr_indices]
            values = self._predict(query, ocr_documents, OCR_PROMPT)
            for index, value in zip(ocr_indices, values):
                ocr_scores[index] = value

        return [
            RerankScore(visual_scores[index], ocr_scores[index], joint_scores[index])
            for index in range(len(results))
        ]
