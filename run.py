"""Stable Kaggle launcher for the AIC dashboard.

Keep ``run_kaggle.ipynb`` unchanged.  This script pulls the latest source,
re-executes itself when it changed, clears source-level Python caches, and
starts the dashboard in a fresh process.  It never downloads or copies the
AIC dataset: data is read only from the Kaggle Inputs mount.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path


REPO = Path(__file__).resolve().parent
CODE = REPO / "Code"
DEFAULT_OCR_INDEX = Path("/kaggle/working/aic_ocr_index.jsonl.gz")
RUNTIME_DIR = Path(os.environ.get("AIC_RUNTIME_DIR", "/kaggle/working"))
# v2 intentionally avoids an earlier venv created without pip by Kaggle's
# Python image. Keep it alongside the old directory; users need not delete it.
OCR_VENV = RUNTIME_DIR / "aic_paddle_ocr_venv_v2"
PADDLE_GPU_INDEX = "https://www.paddlepaddle.org.cn/packages/stable/cu118/"
STALE_MODULES = (
    "dashboard",
    "share_dashboard",
    "retrieval",
    "ocr_index",
    "data_paths",
    "clip_encoder",
    "qa",
    "query_language",
)


def command(arguments: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(arguments), flush=True)
    subprocess.run(arguments, check=True, env=env)


def update_source(skip_update: bool) -> None:
    """Fast-forward the clean Kaggle clone and restart with new ``run.py``."""
    if skip_update:
        return
    before = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    try:
        command(["git", "-C", str(REPO), "pull", "--ff-only"])
    except subprocess.CalledProcessError:
        # A notebook user may intentionally have local source edits. The
        # launcher remains usable instead of discarding those edits.
        print("[warning] Không thể git pull; dùng source hiện có.", file=sys.stderr, flush=True)
        return
    after = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    if after != before:
        os.execv(sys.executable, [sys.executable, str(REPO / "run.py"), "--skip-update", *sys.argv[1:]])


def clear_source_cache() -> None:
    """Remove only disposable source bytecode; preserve model/OCR caches."""
    shutil.rmtree(REPO / "__pycache__", ignore_errors=True)
    for cache_dir in CODE.rglob("__pycache__"):
        shutil.rmtree(cache_dir, ignore_errors=True)
    for module in STALE_MODULES:
        sys.modules.pop(module, None)
    importlib.invalidate_caches()
    gc.collect()


def install_if_changed(requirements: Path, marker_name: str) -> None:
    """Install only when this requirements file changed in a later git pull."""
    marker = RUNTIME_DIR / marker_name
    digest = hashlib.sha256(requirements.read_bytes()).hexdigest()
    try:
        installed_digest = marker.read_text(encoding="utf-8").strip()
    except OSError:
        installed_digest = ""
    if digest == installed_digest:
        print(f"Dependency đã sẵn sàng: {requirements.name}", flush=True)
        return
    command([sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)])
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    marker.write_text(digest + "\n", encoding="utf-8")


def install_requirements(build_ocr: bool) -> None:
    install_if_changed(CODE / "requirements.txt", ".aic_requirements.sha256")


def ensure_paddle_ocr_venv() -> Path:
    """Install Paddle GPU separately so it cannot break dashboard PyTorch."""
    gpu_probe = subprocess.run(["nvidia-smi", "-L"], text=True, capture_output=True, check=False)
    if gpu_probe.returncode != 0:
        raise RuntimeError(
            "Kaggle chưa bật GPU Accelerator. Vào Notebook settings → Accelerator → GPU, "
            "restart session rồi Run all. Không pre-OCR toàn bộ dataset bằng CPU."
        )
    python = OCR_VENV / "bin" / "python"
    if not python.is_file():
        # Kaggle's CPython omits ensurepip. Including its system packages makes
        # the mandatory sitecustomize dependency (wrapt) visible at startup;
        # PaddleOCR itself is still installed in this venv first.
        command([sys.executable, "-m", "venv", "--system-site-packages", str(OCR_VENV)])
    venv_env = os.environ.copy()
    venv_env.pop("PYTHONPATH", None)
    venv_env.pop("PYTHONHOME", None)
    venv_env["VIRTUAL_ENV"] = str(OCR_VENV)
    venv_env["PATH"] = f"{OCR_VENV / 'bin'}:{venv_env.get('PATH', '')}"
    digest = hashlib.sha256((CODE / "requirements-ocr.txt").read_bytes()).hexdigest()
    marker = OCR_VENV / ".requirements.sha256"
    try:
        installed_digest = marker.read_text(encoding="utf-8").strip()
    except OSError:
        installed_digest = ""
    if digest != installed_digest:
        print("Đang cài PaddleOCR GPU trong virtualenv riêng…", flush=True)
        command(
            [
                sys.executable,
                "-m",
                "pip",
                "--python",
                str(python),
                "install",
                "-q",
                "--no-cache-dir",
                "paddlepaddle-gpu==3.0.0",
                "-i",
                PADDLE_GPU_INDEX,
            ],
        )
        command(
            [
                sys.executable,
                "-m",
                "pip",
                "--python",
                str(python),
                "install",
                "-q",
                "-r",
                str(CODE / "requirements-ocr.txt"),
            ],
        )
        marker.write_text(digest + "\n", encoding="utf-8")
    probe = "import paddle; print(int(paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0))"
    result = subprocess.run(
        [str(python), "-c", probe], text=True, capture_output=True, check=False, env=venv_env
    )
    if result.returncode != 0 or result.stdout.strip() != "1":
        raise RuntimeError("PaddleOCR virtualenv không thấy CUDA GPU. Hãy restart Kaggle session rồi Run all.")
    print("PaddleOCR GPU virtualenv đã sẵn sàng.", flush=True)
    return python


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the AIC dashboard on Kaggle")
    parser.add_argument("--skip-update", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--build-ocr", action="store_true", help="Pre-OCR mounted keyframes before launch")
    parser.add_argument("--no-build-ocr", action="store_true", help="Launch with CLIP only when no OCR index exists")
    parser.add_argument("--ocr-index", type=Path, default=DEFAULT_OCR_INDEX)
    parser.add_argument("--data-root", default="/kaggle/input")
    parser.add_argument("--ocr-device", default="gpu:0", help="OCR device; full pre-OCR requires gpu:0")
    parser.add_argument("--no-preload-features", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    update_source(arguments.skip_update)
    complete_marker = Path(f"{arguments.ocr_index}.complete")
    index_ready = arguments.ocr_index.is_file() and complete_marker.is_file()
    forced_ocr = arguments.build_ocr or os.environ.get("AIC_BUILD_OCR", "0").lower() in {"1", "true", "yes"}
    # Accuracy-first default: a fresh Kaggle session pre-OCRs once. Subsequent
    # Run all executions reuse the compact text index and launch immediately.
    build_ocr = forced_ocr or (not arguments.no_build_ocr and not index_ready)
    clear_source_cache()
    install_requirements(build_ocr)

    os.environ["AIC_DATA_ROOT"] = str(Path(arguments.data_root).expanduser())
    os.environ["AIC_OCR_INDEX"] = str(arguments.ocr_index)
    os.environ["AIC_PRELOAD_FEATURES"] = "0" if arguments.no_preload_features else "1"
    sys.path.insert(0, str(CODE))

    if build_ocr:
        if not arguments.ocr_device.startswith("gpu"):
            raise ValueError("Full pre-OCR chỉ hỗ trợ GPU. Dùng --no-build-ocr nếu không cần OCR.")
        ocr_python = ensure_paddle_ocr_venv()
        os.environ["AIC_OCR_DEVICE"] = arguments.ocr_device
        print("Chưa có OCR index hợp lệ; đang pre-OCR keyframe đã mount.", flush=True)
        command(
            [
                str(ocr_python),
                str(CODE / "build_ocr_index.py"),
                "--output",
                str(arguments.ocr_index),
                "--device",
                arguments.ocr_device,
            ]
        )

    from data_paths import AICPaths
    from share_dashboard import launch_dashboard

    paths = AICPaths.from_environment()
    print("CLIP features:", paths.features_dir, flush=True)
    print("Keyframe roots:", len(paths.keyframe_roots), flush=True)
    print("OCR index:", arguments.ocr_index if arguments.ocr_index.is_file() else "chưa build — CLIP-only", flush=True)
    dashboard_url = launch_dashboard(share=True)
    print("Open dashboard:", dashboard_url or "không tạo được share URL", flush=True)
    print("Giữ cell này chạy để dashboard hoạt động. Dừng cell để tắt dashboard.", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("Đã dừng dashboard.", flush=True)


if __name__ == "__main__":
    main()
