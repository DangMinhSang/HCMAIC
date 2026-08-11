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
            import qwen_vl_utils  # type: ignore  # noqa: F401
            from sentence_transformers import CrossEncoder  # type: ignore
        except Exception as error:
            raise MultimodalRerankerUnavailableError(
                "Không import được Qwen reranker stack "
                f"({type(error).__name__}: {error}). "
                "Chạy lại run.py để cài đúng transformers/sentence-transformers; "
                "muốn tắt thì đặt AIC_RERANKER=0."
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
            batch_size = max(1, min(int(os.environ.get("AIC_RERANKER_BATCH_SIZE", "1")), 4))
        except ValueError:
            batch_size = 1
        try:
            values = self.model.predict(
                pairs,
                batch_size=batch_size,
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

        Use one multimodal pass per candidate.  Three independent Qwen passes
        (image, OCR, and joint) made a synchronous Gradio request exceed the
        gateway timeout.  The joint score is the model score; visual/OCR are
        calibrated first-stage signals retained as cheap supporting evidence.
        """
        joint_documents = [
            document or {"text": "No image or OCR evidence available."}
            for document in (self._joint_document(result) for result in results)
        ]
        joint_scores = self._predict(query, joint_documents, JOINT_PROMPT)

        def bounded(value: Any) -> float:
            return max(0.0, min(1.0, float(value or 0.0)))

        return [
            RerankScore(
                visual=bounded(getattr(result, "visual_score", 0.0)),
                ocr=bounded(getattr(result, "ocr_score", 0.0)),
                joint=joint_scores[index],
            )
            for index, result in enumerate(results)
        ]
