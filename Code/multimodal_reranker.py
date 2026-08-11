"""Optional Qwen3-VL reranking for query/image/OCR candidates.

The CLIP index is still used for fast recall.  This module only runs on a
small candidate set after recall, so the 2B multimodal model does not need to
scan the whole AIC corpus for every query.
"""

from __future__ import annotations

import os
from collections import OrderedDict
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
        """Fuse the multimodal model with calibrated recall evidence."""
        return 0.84 * self.joint + 0.10 * self.visual + 0.06 * self.ocr


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
        model_kwargs: dict[str, Any] = {}
        if device.startswith("cuda"):
            # T4 has fast native FP16 tensor cores. Loading the 2B model in
            # FP16 also leaves enough VRAM for CLIP and request activations.
            model_kwargs["torch_dtype"] = torch.float16
        try:
            self.model = CrossEncoder(self.model_name, device=device, model_kwargs=model_kwargs)
        except Exception as error:
            raise MultimodalRerankerUnavailableError(
                f"Không tải được Qwen reranker `{self.model_name}`: {error}"
            ) from error
        self.device = device
        self._score_cache: OrderedDict[tuple[str, str, str, str], float] = OrderedDict()
        try:
            self._cache_size = max(0, min(int(os.environ.get("AIC_RERANKER_CACHE", "512")), 4096))
        except ValueError:
            self._cache_size = 512

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

    def _predict_pairs(self, pairs: list[tuple[str, Any]], prompt: str) -> list[float]:
        if not pairs:
            return []
        try:
            batch_size = max(1, min(int(os.environ.get("AIC_RERANKER_BATCH_SIZE", "2")), 4))
        except ValueError:
            batch_size = 2
        try:
            values = self.model.predict(
                pairs,
                batch_size=batch_size,
                show_progress_bar=False,
                activation_fn=self._torch.nn.Sigmoid(),
                prompt=prompt,
            )
        except Exception as error:
            if batch_size > 1 and "out of memory" in str(error).lower():
                if self._torch.cuda.is_available():
                    self._torch.cuda.empty_cache()
                try:
                    values = self.model.predict(
                        pairs,
                        batch_size=1,
                        show_progress_bar=False,
                        activation_fn=self._torch.nn.Sigmoid(),
                        prompt=prompt,
                    )
                except Exception as retry_error:
                    raise MultimodalRerankerUnavailableError(
                        f"Qwen reranker không chấm được candidate: {retry_error}"
                    ) from retry_error
            else:
                raise MultimodalRerankerUnavailableError(
                    f"Qwen reranker không chấm được candidate: {error}"
                ) from error
        return [max(0.0, min(1.0, float(value))) for value in values]

    @staticmethod
    def _cache_key(query: str, result: Any, prompt: str) -> tuple[str, str, str, str]:
        return (
            query,
            str(getattr(result, "image_path", "") or ""),
            str(getattr(result, "ocr_text", "") or "").strip(),
            prompt,
        )

    def score_pairs(
        self,
        queries: Sequence[str],
        results: Sequence[Any],
        *,
        prompt: str = JOINT_PROMPT,
    ) -> list[float]:
        """Score aligned query/result pairs with one batched model call."""
        if len(queries) != len(results):
            raise ValueError("Số query và candidate Qwen phải bằng nhau.")
        output: list[float | None] = [None] * len(results)
        missing_pairs: list[tuple[str, Any]] = []
        missing_positions: list[int] = []
        missing_keys: list[tuple[str, str, str, str]] = []
        for index, (query, result) in enumerate(zip(queries, results)):
            key = self._cache_key(query, result, prompt)
            cached = self._score_cache.get(key)
            if cached is not None:
                self._score_cache.move_to_end(key)
                output[index] = cached
                continue
            document = self._joint_document(result) or {"text": "No image or OCR evidence available."}
            missing_pairs.append((query, document))
            missing_positions.append(index)
            missing_keys.append(key)
        if missing_pairs:
            values = self._predict_pairs(missing_pairs, prompt)
            for position, key, value in zip(missing_positions, missing_keys, values):
                output[position] = value
                if self._cache_size:
                    self._score_cache[key] = value
                    self._score_cache.move_to_end(key)
                    while len(self._score_cache) > self._cache_size:
                        self._score_cache.popitem(last=False)
        if any(value is None for value in output):
            raise MultimodalRerankerUnavailableError(
                "Qwen reranker trả thiếu điểm cho một hoặc nhiều candidate."
            )
        return [float(value) for value in output]

    def score(self, query: str, results: Sequence[Any]) -> list[RerankScore]:
        """Return visual/OCR/joint probabilities for each result.

        Use one multimodal pass per candidate.  Three independent Qwen passes
        (image, OCR, and joint) made a synchronous Gradio request exceed the
        gateway timeout.  The joint score is the model score; visual/OCR are
        calibrated first-stage signals retained as cheap supporting evidence.
        """
        joint_scores = self.score_pairs([query] * len(results), results)

        def bounded(value: Any) -> float:
            return max(0.0, min(1.0, float(value or 0.0)))

        raw_visual = [float(getattr(result, "visual_score", 0.0) or 0.0) for result in results]
        if raw_visual:
            minimum, maximum = min(raw_visual), max(raw_visual)
            spread = maximum - minimum
            visual_scores = [0.5] * len(raw_visual) if spread < 1e-9 else [
                (value - minimum) / spread for value in raw_visual
            ]
        else:
            visual_scores = []

        return [
            RerankScore(
                visual=bounded(visual_scores[index]),
                ocr=bounded(getattr(result, "ocr_score", 0.0)),
                joint=joint_scores[index],
            )
            for index, result in enumerate(results)
        ]
