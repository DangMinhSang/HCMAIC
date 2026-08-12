"""A lazy visual-QA baseline for the AIC Q&A tab."""

from __future__ import annotations

import gc
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ocr_regions import prepare_reranker_image
from progress import track
from query_language import normalize_query
from retrieval import SearchResult


QUESTION_RE = re.compile(r"(?:câu\s*hỏi|question)\s*[:\-]\s*(.+)$", flags=re.IGNORECASE | re.DOTALL)
COUNT_RE = re.compile(
    r"\b(?:bao\s*nhiêu|bao\s*nhieu|how\s*many|number\s+of|count|"
    r"số\s*lượng|so\s*luong|dem\s*so)\b",
    re.IGNORECASE,
)
HOLDING_RE = re.compile(
    r"\b(?:cầm|cam|holding|hold|carrying|carry|nắm|nam)\b|\bwhat\s+is\s+.+\s+holding\b",
    re.IGNORECASE,
)
APPEARANCE_RE = re.compile(
    r"\b(?:màu|mau|color|colour|mặc|mac|wearing|áo|ao)\b",
    re.IGNORECASE,
)
NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "khong": "0", "mot": "1", "hai": "2", "ba": "3",
    "bon": "4", "tu": "4", "nam": "5", "sau": "6", "bay": "7",
    "tam": "8", "chin": "9", "muoi": "10",
}


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


def question_kind(question: str) -> str:
    """Classify the small set of question forms needing stricter grounding."""
    if COUNT_RE.search(question):
        return "count"
    if HOLDING_RE.search(question):
        return "holding"
    if APPEARANCE_RE.search(question):
        return "appearance"
    return "general"


def _question_prompt(question: str, kind: str) -> str:
    instruction = (
        "Inspect the entire visible scene and answer only from visual evidence. Ignore TV lower-thirds, "
        "scrolling tickers, subtitles, logos, clocks, and words on presentation slides. Do not infer from "
        "the event description or common sense. Return only the shortest direct answer. "
    )
    if kind == "count":
        instruction += (
            "This is a counting question: count distinct, clearly visible instances of the requested "
            "subject in the scene, not people or objects printed on a screen, poster, or photograph. "
            "Return one Arabic integer only; if the subject cannot be seen clearly, return 0. "
        )
    elif kind == "holding":
        instruction += (
            "This asks what a person is holding: first locate the person using the clothing/color clue, "
            "then inspect the hands. Return only the object name; if no object is visibly held, return "
            "'nothing' or 'unclear', never an inferred activity. "
        )
    elif kind == "appearance":
        instruction += "Answer the requested visible color or clothing attribute, not a background object. "
    return instruction + f"Question: {question}"


def normalize_answer(answer: str, kind: str = "general") -> str:
    cleaned = re.sub(r"^(?:answer|đáp\s*án)\s*[:\-]\s*", "", answer, flags=re.IGNORECASE).strip()
    if kind == "count":
        match = re.search(r"\b\d+\b", cleaned)
        if match:
            return match.group(0)
        words = re.findall(r"[\wÀ-ỹĐđ]+", cleaned.lower())
        for word in words:
            normalized = "".join(
                character
                for character in unicodedata.normalize("NFD", word)
                if unicodedata.category(character) != "Mn"
            )
            normalized = normalized.replace("đ", "d")
            if normalized in NUMBER_WORDS:
                return NUMBER_WORDS[normalized]
    if kind == "holding":
        cleaned = re.sub(
            r"^(?:the\s+)?(?:red[- ]shirt(?:ed)?\s+)?(?:person|man|woman)\s+"
            r"(?:is\s+)?(?:holding|carrying)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip(" .,:;")
        cleaned = re.sub(r"^(?:a|an|the)\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned or "unknown"


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
                self._predict_one("What is visible in this image?", Path(image_path), "general")
            except Exception as error:
                if not self.backend_name.startswith("Qwen"):
                    raise
                self.load_error = f"Qwen VQA warmup fallback: {type(error).__name__}: {error}"
                self.requested_backend = "vilt"
                self._release_model()
                self._load_vilt()
                self._predict_one("What is visible in this image?", Path(image_path), "general")

    def predict(
        self,
        question: str,
        results: Sequence[SearchResult],
        limit: int | None = None,
    ) -> list[QAPrediction]:
        self._load()
        if limit is None:
            try:
                limit = max(1, min(int(os.environ.get("AIC_VQA_CANDIDATES", "8")), 12))
            except ValueError:
                limit = 8
        normalized = normalize_query(question)
        kind = question_kind(normalized.text_for_model)
        source_results = results[:limit]
        candidates = [
            result
            for result in track(
                source_results,
                desc="Kiểm tra ảnh VQA",
                total=len(source_results),
                unit="frame",
            )
            if result.image_path and Path(result.image_path).is_file()
        ]
        try:
            predictions = []
            for result in track(
                candidates,
                desc=f"VQA {self.backend_name}",
                total=len(candidates),
                unit="frame",
                force=True,
                leave=True,
            ):
                predictions.append(
                    QAPrediction(
                        result.rank,
                        *self._predict_one(normalized.text_for_model, Path(result.image_path), kind),
                    )
                )
            return predictions
        except Exception as error:
            if self.backend_name.startswith("Qwen"):
                self.load_error = f"Qwen VQA inference fallback: {type(error).__name__}: {error}"
                self.requested_backend = "vilt"
                self._release_model()
                self._load_vilt()
                predictions = []
                for result in track(
                    candidates,
                    desc="VQA ViLT fallback",
                    total=len(candidates),
                    unit="frame",
                    force=True,
                    leave=True,
                ):
                    predictions.append(
                        QAPrediction(
                            result.rank,
                            *self._predict_one(normalized.text_for_model, Path(result.image_path), kind),
                        )
                    )
                return predictions
            raise

    def _predict_one(self, question: str, image_path: Path, kind: str = "general") -> tuple[str, float]:
        if self.backend_name.startswith("Qwen"):
            return self._predict_qwen(question, image_path, kind)
        return self._predict_vilt(question, image_path)

    def _predict_qwen(self, question: str, image_path: Path, kind: str = "general") -> tuple[str, float]:
        try:
            from PIL import Image  # type: ignore
        except ImportError as error:
            raise RuntimeError("Thiếu Pillow cho Q&A.") from error
        prepared = prepare_reranker_image(image_path)
        if isinstance(prepared, (str, Path)):
            with Image.open(prepared) as source:
                image = source.convert("RGB").copy()
        else:
            image = prepared
        prompt = _question_prompt(question, kind)
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
        answer = normalize_answer(answer, kind)
        probabilities: list[float] = []
        for logits, token_id in track(
            zip(generated.scores, token_ids),
            desc="Tính VQA confidence",
            total=len(generated.scores),
            unit="token",
            nested=True,
        ):
            probability = logits[0].float().softmax(dim=-1)[int(token_id)]
            probabilities.append(max(1e-8, float(probability)))
        confidence = math.exp(sum(math.log(value) for value in probabilities) / len(probabilities)) if probabilities else 0.0
        close = getattr(image, "close", None)
        if callable(close):
            close()
        return answer, confidence

    def _predict_vilt(self, question: str, image_path: Path) -> tuple[str, float]:
        try:
            from PIL import Image  # type: ignore
        except ImportError as error:
            raise RuntimeError("Thiếu Pillow cho Q&A.") from error
        prepared = prepare_reranker_image(image_path)
        if isinstance(prepared, (str, Path)):
            with Image.open(prepared) as image:
                inputs = self._processor(images=image.convert("RGB"), text=question, return_tensors="pt")
        else:
            try:
                inputs = self._processor(images=prepared, text=question, return_tensors="pt")
            finally:
                close = getattr(prepared, "close", None)
                if callable(close):
                    close()
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            logits = self._model(**inputs).logits[0]
            probabilities = logits.softmax(dim=-1)
            index = int(probabilities.argmax())
        return str(self._model.config.id2label[index]), float(probabilities[index])
