"""Find AIC datasets that are already available on disk.

The public assets can be mounted in several layouts: directly below
``/kaggle/input`` (normal Kaggle inputs) or below
``input/datasets/doanminhtuan`` (the supplied dataset tree).  This module
only resolves paths and never downloads, copies, or indexes the dataset.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from progress import track


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the required Kaggle inputs have not been attached."""


def _first_existing(candidates: Iterable[Path], label: str) -> Path:
    checked = list(dict.fromkeys(Path(path) for path in candidates))
    for path in track(
        checked,
        desc=f"Tìm {label}",
        total=len(checked),
        unit="path",
        nested=True,
    ):
        if path.is_dir():
            return path
    preview = "\n".join(f"  - {path}" for path in checked[:8])
    raise DatasetNotFoundError(
        f"Không tìm thấy {label}. Hãy Add Input các dataset AIC vào Kaggle "
        f"hoặc đặt biến môi trường tương ứng. Các đường dẫn đã thử:\n{preview}"
    )


def _env_path(variable: str) -> Path | None:
    value = os.environ.get(variable)
    return Path(value).expanduser() if value else None


@dataclass(frozen=True)
class AICPaths:
    """Locations of the mounted AIC assets.

    Attributes point to the small directory metadata of mounted inputs. No
    dataset content is copied into the repository or into a cache.
    """

    input_root: Path
    features_dir: Path
    mapping_dir: Path
    metadata_dir: Path | None
    keyframe_roots: tuple[Path, ...]
    objects_dir: Path | None
    video_roots: tuple[Path, ...]

    @classmethod
    def from_environment(cls, input_root: str | Path | None = None) -> "AICPaths":
        root = Path(input_root or os.environ.get("AIC_DATA_ROOT") or "/kaggle/input").expanduser()

        # Each base supports normal Kaggle inputs and the supplied tree layout.
        bases = (root, root / "datasets" / "doanminhtuan", root / "doanminhtuan")

        def component_dirs(dataset: str, child: str) -> list[Path]:
            return [base / dataset / child for base in bases]

        features_dir = _env_path("AIC_FEATURES_DIR") or _first_existing(
            component_dirs("clip-features-32-aic25-b1", "clip-features-32"),
            "CLIP features",
        )
        mapping_dir = _env_path("AIC_MAPPING_DIR") or _first_existing(
            component_dirs("map-keyframes-aic25-b1", "map-keyframes"),
            "mapping keyframe-to-frame",
        )

        metadata_candidates = component_dirs("media-info-aic25-b1", "media-info")
        metadata_env = _env_path("AIC_METADATA_DIR")
        metadata_dir = metadata_env or next(
            (path for path in metadata_candidates if path.is_dir()), None
        )

        keyframe_base_candidates = [base / "dataset-ai-challenge-keyframe" for base in bases]
        keyframe_base = _env_path("AIC_KEYFRAMES_DIR") or next(
            (path for path in keyframe_base_candidates if path.is_dir()), None
        )
        if keyframe_base is None:
            raise DatasetNotFoundError(
                "Không tìm thấy keyframes. Hãy Add Input dataset "
                "dataset-ai-challenge-keyframe vào notebook."
            )
        keyframe_roots = tuple(
            sorted(
                path for path in keyframe_base.glob("Keyframes_*") if (path / "keyframes").is_dir()
            )
        )
        if not keyframe_roots:
            raise DatasetNotFoundError(f"Không có thư mục Keyframes_* trong {keyframe_base}")

        objects_candidates = component_dirs("objects-aic25-b1-zip", "objects")
        objects_env = _env_path("AIC_OBJECTS_DIR")
        objects_dir = objects_env or next(
            (path for path in objects_candidates if path.is_dir()), None
        )

        video_base_candidates = [base / "video-aic" for base in bases]
        video_base = _env_path("AIC_VIDEOS_DIR") or next(
            (path for path in video_base_candidates if path.is_dir()), None
        )
        video_roots: tuple[Path, ...] = ()
        if video_base is not None:
            video_roots = tuple(
                sorted(path for path in video_base.glob("Videos_*") if (path / "video").is_dir())
            )

        return cls(
            input_root=root,
            features_dir=features_dir,
            mapping_dir=mapping_dir,
            metadata_dir=metadata_dir,
            keyframe_roots=keyframe_roots,
            objects_dir=objects_dir,
            video_roots=video_roots,
        )

    def image_path(self, video_id: str, keyframe_number: int) -> Path | None:
        filename = f"{keyframe_number:03d}.jpg"
        for root in track(
            self.keyframe_roots,
            desc="Tìm keyframe root",
            total=len(self.keyframe_roots),
            unit="root",
            nested=True,
        ):
            candidate = root / "keyframes" / video_id / filename
            if candidate.is_file():
                return candidate
        return None

    def object_path(self, video_id: str, keyframe_number: int) -> Path | None:
        if self.objects_dir is None:
            return None
        candidate = self.objects_dir / video_id / f"{keyframe_number:03d}.json"
        return candidate if candidate.is_file() else None

    def video_path(self, video_id: str) -> Path | None:
        for root in track(
            self.video_roots,
            desc="Tìm video root",
            total=len(self.video_roots),
            unit="root",
            nested=True,
        ):
            candidate = root / "video" / f"{video_id}.mp4"
            if candidate.is_file():
                return candidate
        return None
