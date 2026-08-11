"""A lazy visual-QA baseline for the AIC Q&A tab."""

from __future__ import annotations

import gc
import math
import os
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
    """Accuracy-first Qwen3-VL VQA with a compact ViLT fallback."""

    def __init__(self, model_name: str | None = None) -> None:
        default_backend = "qwen" if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") else "vilt"
        self.requested_backend = os.environ.get("AIC_VQA_BACKEND", default_backend).strip().lower()
        self.model_name = model_name or os.environ.get(
            "AIC_VQA_MODEL",
            "Qwen/Qwen3-VL-2B-Instruct"
            if self.requested_backend == "qwen"
            else "dandelin/vilt-b32-finetuned-vqa",
        )
        self._model = None
        self._processor = None
        self._torch = None
        self.device = None
        self.backend_name = "unloaded"
        self.load_error = ""

    def _load(self) -> None:
        if self._model is not None:
            return
        if self.requested_backend == "qwen":
            try:
                self._load_qwen()
                return
            except Exception as error:
                self.load_error = f"Qwen VQA fallback: {type(error).__name__}: {error}"
                self._release_model()
        self._load_vilt()

    def _load_qwen(self) -> None:
        try:
            import torch  # type: ignore
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # type: ignore
        except ImportError as error:
            raise RuntimeError("Thiếu Transformers v5 cho Qwen3-VL VQA.") from error
        self._torch = torch
        if not torch.cuda.is_available() and os.environ.get("AIC_VQA_CPU", "0").lower() not in {
            "1", "true", "yes",
        }:
            raise RuntimeError("Qwen VQA cần GPU; dùng AIC_VQA_BACKEND=vilt khi chạy CPU.")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_name,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self._model = self._model.to(self.device).eval()
        self.backend_name = "Qwen3-VL-2B-Instruct"

    def _load_vilt(self) -> None:
        try:
            import torch  # type: ignore
            from transformers import ViltForQuestionAnswering, ViltProcessor  # type: ignore
        except ImportError as error:
            raise RuntimeError("Thiếu transformers/Pillow cho Q&A. Chạy lại cell cài requirements.") from error
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        fallback_model = os.environ.get("AIC_VILT_MODEL", "dandelin/vilt-b32-finetuned-vqa")
        try:
            self._processor = ViltProcessor.from_pretrained(fallback_model)
            self._model = ViltForQuestionAnswering.from_pretrained(fallback_model)
            if self.device == "cuda":
                self._model = self._model.half()
            self._model = self._model.to(self.device).eval()
        except Exception as error:
            raise RuntimeError(
                "Không tải được Qwen hoặc ViLT VQA. Bật Internet lần đầu hoặc cache model trước; "
                "truy xuất keyframe vẫn hoạt động."
            ) from error
        self._torch = torch
        self.backend_name = "ViLT fallback"

    def _release_model(self) -> None:
        self._model = None
        self._processor = None
        gc.collect()
        try:
            if self._torch is not None and self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
        except Exception:
            pass

    def warmup(self, image_path: str | None = None) -> None:
        """Load weights and optionally execute one real-image inference."""
        self._load()
        if image_path and Path(image_path).is_file():
            try:
                self._predict_one("What is visible in this image?", Path(image_path))
            except Exception as error:
                if not self.backend_name.startswith("Qwen"):
                    raise
                self.load_error = f"Qwen VQA warmup fallback: {type(error).__name__}: {error}"
                self.requested_backend = "vilt"
                self._release_model()
                self._load_vilt()
                self._predict_one("What is visible in this image?", Path(image_path))

    def predict(
        self,
        question: str,
        results: Sequence[SearchResult],
        limit: int | None = None,
    ) -> list[QAPrediction]:
        self._load()
        if limit is None:
            try:
                limit = max(1, min(int(os.environ.get("AIC_VQA_CANDIDATES", "6")), 12))
            except ValueError:
                limit = 6
        normalized = normalize_query(question)
        candidates = [
            result
            for result in results[:limit]
            if result.image_path and Path(result.image_path).is_file()
        ]
        try:
            return [
                QAPrediction(result.rank, *self._predict_one(normalized.text_for_model, Path(result.image_path)))
                for result in candidates
            ]
        except Exception as error:
            if self.backend_name.startswith("Qwen"):
                self.load_error = f"Qwen VQA inference fallback: {type(error).__name__}: {error}"
                self.requested_backend = "vilt"
                self._release_model()
                self._load_vilt()
                return [
                    QAPrediction(result.rank, *self._predict_one(normalized.text_for_model, Path(result.image_path)))
                    for result in candidates
                ]
            raise

    def _predict_one(self, question: str, image_path: Path) -> tuple[str, float]:
        if self.backend_name.startswith("Qwen"):
            return self._predict_qwen(question, image_path)
        return self._predict_vilt(question, image_path)

    def _predict_qwen(self, question: str, image_path: Path) -> tuple[str, float]:
        try:
            from PIL import Image  # type: ignore
        except ImportError as error:
            raise RuntimeError("Thiếu Pillow cho Q&A.") from error
        with Image.open(image_path) as source:
            image = source.convert("RGB").copy()
        prompt = (
            "Answer the visual question using only the image. Return only the shortest direct answer, "
            f"without explanation. Question: {question}"
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        try:
            max_tokens = max(2, min(int(os.environ.get("AIC_VQA_MAX_TOKENS", "16")), 40))
        except ValueError:
            max_tokens = 16
        with self._torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
        input_length = int(inputs["input_ids"].shape[-1])
        token_ids = generated.sequences[0, input_length:]
        answer = self._processor.batch_decode(
            [token_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        answer = re.sub(r"^(?:answer|đáp\s*án)\s*[:\-]\s*", "", answer, flags=re.IGNORECASE).strip()
        probabilities: list[float] = []
        for logits, token_id in zip(generated.scores, token_ids):
            probability = logits[0].float().softmax(dim=-1)[int(token_id)]
            probabilities.append(max(1e-8, float(probability)))
        confidence = math.exp(sum(math.log(value) for value in probabilities) / len(probabilities)) if probabilities else 0.0
        return answer or "unknown", confidence

    def _predict_vilt(self, question: str, image_path: Path) -> tuple[str, float]:
        try:
            from PIL import Image  # type: ignore
        except ImportError as error:
            raise RuntimeError("Thiếu Pillow cho Q&A.") from error
        with Image.open(image_path) as image:
            inputs = self._processor(images=image.convert("RGB"), text=question, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            logits = self._model(**inputs).logits[0]
            probabilities = logits.softmax(dim=-1)
            index = int(probabilities.argmax())
        return str(self._model.config.id2label[index]), float(probabilities[index])
