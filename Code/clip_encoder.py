"""OpenAI CLIP ViT-B/32 text encoder compatible with the supplied features."""

from __future__ import annotations

import os
from pathlib import Path


class ClipModelUnavailableError(RuntimeError):
    """Raised with an actionable message when the CLIP model is unavailable."""


class ClipTextEncoder:
    """Encode text with the exact ViT-B/32 CLIP family used by AIC features.

    Model weights are cached in ``AIC_CLIP_CACHE`` (or ``~/.cache/clip``) by
    the CLIP package. This is a model cache, not a copy of any AIC dataset.
    """

    def __init__(self, model_name: str = "ViT-B/32", device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._clip = None
        self._torch = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import clip  # type: ignore
            import torch  # type: ignore
        except ImportError as error:
            raise ClipModelUnavailableError(
                "Thiếu CLIP/Torch. Chạy `pip install -r Code/requirements.txt` "
                "(notebook Kaggle đã có cell này)."
            ) from error

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        cache = Path(os.environ.get("AIC_CLIP_CACHE", "~/.cache/clip")).expanduser()
        try:
            model, _ = clip.load(self.model_name, device=device, download_root=str(cache))
        except Exception as error:  # package-specific model/cache errors
            raise ClipModelUnavailableError(
                "Không tải được trọng số CLIP ViT-B/32. Bật Internet cho Kaggle "
                "lần đầu hoặc mount sẵn cache qua AIC_CLIP_CACHE. Ứng dụng không "
                "tải dataset nào."
            ) from error
        model.eval()
        self.device = device
        self._model, self._clip, self._torch = model, clip, torch

    def encode(self, query: str, english_expansion: str = ""):
        """Return one L2-normalized NumPy query vector.

        CLIP ViT-B/32 is strongest in English. An accurate English expansion
        is deliberately weighted more when supplied; prompt ensembling reduces
        sensitivity to small wording changes.
        """
        self._load()
        query = (query or "").strip()
        english_expansion = (english_expansion or "").strip()
        if not query and not english_expansion:
            raise ValueError("Nhập mô tả truy vấn.")

        base = english_expansion or query
        prompts = [base]
        if english_expansion:
            prompts.extend([f"a video frame of {base}", f"a photograph of {base}"])
            if query and query != english_expansion:
                prompts.append(query)
        elif all(ord(char) < 128 for char in base):
            prompts.extend([f"a video frame of {base}", f"a photograph of {base}"])

        tokens = self._clip.tokenize(prompts, truncate=True).to(self.device)
        with self._torch.no_grad():
            embeddings = self._model.encode_text(tokens).float()
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            query_embedding = embeddings.mean(dim=0)
            query_embedding = query_embedding / query_embedding.norm().clamp_min(1e-12)
        return query_embedding.cpu().numpy().astype("float32", copy=False)
