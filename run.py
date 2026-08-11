"""Stable Kaggle launcher for the AIC dashboard.

Keep ``run_kaggle.ipynb`` unchanged.  This script pulls the latest source,
re-executes itself when it changed, clears source-level Python caches, and
starts the dashboard in a fresh process.  It never downloads or copies the
AIC dataset: data is read only from the Kaggle Inputs mount.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import importlib
import json
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
RERANKER_REQUIREMENTS = CODE / "requirements-reranker.txt"
# Kaggle's CPython image has no ``ensurepip``, so a virtualenv cannot be
# bootstrapped reliably there.  Keep Paddle packages in this private directory
# and expose it only to the pre-OCR subprocess via PYTHONPATH.
OCR_PACKAGES = RUNTIME_DIR / "aic_paddle_ocr_packages"
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
    "multimodal_reranker",
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


def install_reranker_requirements() -> None:
    """Install and probe the Kaggle-only Qwen reranker stack.

    Keep this separate from the lightweight local requirements. The probe is
    intentional: a stale marker must not silently produce the fallback path
    after a Kaggle kernel was restarted or packages were removed.
    """
    marker_name = ".aic_reranker_requirements.sha256"
    install_if_changed(RERANKER_REQUIREMENTS, marker_name)
    probe = (
        "import torch, qwen_vl_utils, sentence_transformers, transformers; "
        "from sentence_transformers import CrossEncoder; "
        "from transformers import Qwen3VLForConditionalGeneration; "
        "print(transformers.__version__, sentence_transformers.__version__)"
    )
    result = subprocess.run([sys.executable, "-c", probe], text=True, capture_output=True, check=False)
    if result.returncode == 0:
        print(f"Qwen reranker đã sẵn sàng (transformers {result.stdout.strip().splitlines()[-1]}).", flush=True)
        return
    print("Dependency Qwen reranker chưa hợp lệ; đang cài lại trong Kaggle kernel…", flush=True)
    marker = RUNTIME_DIR / marker_name
    marker.unlink(missing_ok=True)
    install_if_changed(RERANKER_REQUIREMENTS, marker_name)
    verify = subprocess.run([sys.executable, "-c", probe], text=True, capture_output=True, check=False)
    if verify.returncode != 0:
        detail = (verify.stderr or verify.stdout).strip().splitlines()[-1:]
        raise RuntimeError(
            "Không thể kích hoạt Qwen reranker trong Kaggle. "
            + (detail[0] if detail else "Kiểm tra Internet và GPU rồi restart kernel.")
        )
    print(f"Qwen reranker đã sẵn sàng (transformers {verify.stdout.strip().splitlines()[-1]}).", flush=True)


def install_requirements(build_ocr: bool, enable_reranker: bool = True) -> None:
    install_if_changed(CODE / "requirements.txt", ".aic_requirements.sha256")
    if enable_reranker:
        install_reranker_requirements()


def _open_ocr_text(path: Path, mode: str):
    return gzip.open(path, mode, encoding="utf-8") if path.suffix == ".gz" else path.open(mode, encoding="utf-8")


def import_ocr_index(source: Path, destination: Path) -> int:
    """Validate and install a previously exported text-only OCR index."""
    source = source.expanduser().resolve()
    destination = destination.expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Không tìm thấy OCR index để import: {source}")
    records = 0
    try:
        with _open_ocr_text(source, "rt") as stream:
            for line_number, line in enumerate(stream, 1):
                payload = json.loads(line)
                if not payload.get("video_id") or "keyframe_number" not in payload:
                    raise ValueError(f"Dòng OCR {line_number} thiếu video_id/keyframe_number")
                records += 1
    except (OSError, EOFError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"OCR index không hợp lệ hoặc bị thiếu: {source} ({exc})") from exc
    if records == 0:
        raise ValueError(f"OCR index rỗng: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source != destination.resolve():
        shutil.copyfile(source, destination)
    marker = Path(f"{destination}.complete")
    marker.write_text(
        json.dumps({"records": records, "imported_from": str(source)}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Đã import OCR index: {records:,} records → {destination}", flush=True)
    return records


def ensure_paddle_ocr_packages() -> dict[str, str]:
    """Install Paddle GPU privately, without changing Kaggle's dashboard env."""
    gpu_probe = subprocess.run(["nvidia-smi", "-L"], text=True, capture_output=True, check=False)
    if gpu_probe.returncode != 0:
        raise RuntimeError(
            "Kaggle chưa bật GPU Accelerator. Vào Notebook settings → Accelerator → GPU, "
            "restart session rồi Run all. Không pre-OCR toàn bộ dataset bằng CPU."
        )
    ocr_env = os.environ.copy()
    ocr_env.pop("PYTHONHOME", None)
    inherited_path = ocr_env.get("PYTHONPATH", "")
    ocr_env["PYTHONPATH"] = f"{OCR_PACKAGES}:{inherited_path}" if inherited_path else str(OCR_PACKAGES)
    ocr_env["PYTHONNOUSERSITE"] = "1"
    digest = hashlib.sha256((CODE / "requirements-ocr.txt").read_bytes()).hexdigest()
    marker = OCR_PACKAGES / ".requirements.sha256"
    try:
        installed_digest = marker.read_text(encoding="utf-8").strip()
    except OSError:
        installed_digest = ""
    if digest != installed_digest:
        print("Đang cài PaddleOCR GPU vào thư mục cách ly…", flush=True)
        OCR_PACKAGES.mkdir(parents=True, exist_ok=True)
        command(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--no-cache-dir",
                "--upgrade",
                "--target",
                str(OCR_PACKAGES),
                "paddlepaddle-gpu==3.0.0",
                "-i",
                PADDLE_GPU_INDEX,
            ], env=ocr_env,
        )
        command(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--upgrade",
                "--target",
                str(OCR_PACKAGES),
                "-r",
                str(CODE / "requirements-ocr.txt"),
            ], env=ocr_env,
        )
        marker.write_text(digest + "\n", encoding="utf-8")
    probe = "import paddle; print(int(paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0))"
    result = subprocess.run(
        [sys.executable, "-c", probe], text=True, capture_output=True, check=False, env=ocr_env
    )
    if result.returncode != 0 or result.stdout.strip().splitlines()[-1:] != ["1"]:
        raise RuntimeError("PaddleOCR packages cách ly không thấy CUDA GPU. Hãy restart Kaggle session rồi Run all.")
    print("PaddleOCR GPU packages cách ly đã sẵn sàng.", flush=True)
    return ocr_env


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the AIC dashboard on Kaggle")
    parser.add_argument("--skip-update", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--build-ocr", action="store_true", help="Pre-OCR mounted keyframes before launch")
    parser.add_argument("--pre-ocr", action="store_true", help="Build OCR index rồi thoát, không mở dashboard")
    parser.add_argument("--import-ocr", type=Path, metavar="PATH", help="Import OCR index đã pre-OCR từ PATH")
    parser.add_argument("--no-build-ocr", action="store_true", help="Launch with CLIP only when no OCR index exists")
    parser.add_argument("--ocr-index", type=Path, default=DEFAULT_OCR_INDEX)
    parser.add_argument("--data-root", default="/kaggle/input")
    parser.add_argument("--ocr-device", default="gpu:0", help="OCR device; full pre-OCR requires gpu:0")
    parser.add_argument("--no-preload-features", action="store_true")
    parser.add_argument("--no-reranker", action="store_true", help="Không cài/chạy Qwen multimodal reranker")
    return parser.parse_args()


def warmup_dashboard(reranker_enabled: bool) -> None:
    """Load query-time resources before Gradio can receive HTTP requests."""
    import dashboard

    print("Đang khởi tạo feature engine và OCR index trước khi mở dashboard…", flush=True)
    dashboard.get_engine()
    dashboard.get_ocr_index()
    if reranker_enabled:
        print("Đang tải Qwen reranker trước query đầu tiên…", flush=True)
        try:
            import torch

            print(
                f"PyTorch {torch.__version__}; CUDA available={torch.cuda.is_available()}"
                + (f"; GPU={torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""),
                flush=True,
            )
        except Exception as error:
            print(f"Không kiểm tra được CUDA từ PyTorch: {error}", flush=True)
        if dashboard.get_reranker() is None:
            detail = dashboard.RERANKER_ERROR or "không có thông tin lỗi"
            print(f"Qwen reranker không sẵn sàng: {detail}", flush=True)
            print("Dashboard sẽ dùng CLIP/OCR fallback.", flush=True)


def main() -> None:
    arguments = parse_arguments()
    if arguments.pre_ocr and arguments.import_ocr:
        raise ValueError("Không dùng đồng thời --pre-ocr và --import-ocr.")
    if arguments.pre_ocr and arguments.no_build_ocr:
        raise ValueError("--pre-ocr cần chạy OCR; không dùng cùng --no-build-ocr.")
    update_source(arguments.skip_update)
    if arguments.import_ocr:
        import_ocr_index(arguments.import_ocr, arguments.ocr_index)
    complete_marker = Path(f"{arguments.ocr_index}.complete")
    index_ready = arguments.ocr_index.is_file() and complete_marker.is_file()
    forced_ocr = (
        arguments.build_ocr
        or arguments.pre_ocr
        or os.environ.get("AIC_BUILD_OCR", "0").lower() in {"1", "true", "yes"}
    )
    # Accuracy-first default: a fresh Kaggle session pre-OCRs once. Subsequent
    # Run all executions reuse the compact text index and launch immediately.
    build_ocr = forced_ocr or (not arguments.no_build_ocr and not index_ready)
    clear_source_cache()
    reranker_enabled = (
        not arguments.no_reranker
        and not arguments.pre_ocr
        and os.environ.get("AIC_RERANKER", "1").lower() not in {"0", "false", "no"}
    )
    install_requirements(build_ocr, reranker_enabled)

    os.environ["AIC_DATA_ROOT"] = str(Path(arguments.data_root).expanduser())
    os.environ["AIC_OCR_INDEX"] = str(arguments.ocr_index)
    os.environ["AIC_PRELOAD_FEATURES"] = "0" if arguments.no_preload_features else "1"
    os.environ["AIC_RERANKER"] = "1" if reranker_enabled else "0"
    # Keep site-packages ahead of the app directory. The project has a
    # compatibility module named ``Code/datasets.py``; putting Code first
    # shadows Hugging Face's external ``datasets`` package when
    # sentence-transformers imports it.
    code_path = str(CODE)
    while code_path in sys.path:
        sys.path.remove(code_path)
    sys.path.append(code_path)

    if build_ocr:
        if not arguments.ocr_device.startswith("gpu"):
            raise ValueError("Full pre-OCR chỉ hỗ trợ GPU. Dùng --no-build-ocr nếu không cần OCR.")
        ocr_env = ensure_paddle_ocr_packages()
        os.environ["AIC_OCR_DEVICE"] = arguments.ocr_device
        print("Chưa có OCR index hợp lệ; đang pre-OCR keyframe đã mount.", flush=True)
        command(
            [
                sys.executable,
                str(CODE / "build_ocr_index.py"),
                "--output",
                str(arguments.ocr_index),
                "--device",
                arguments.ocr_device,
            ],
            env=ocr_env,
        )
        if arguments.pre_ocr:
            print("Pre-OCR hoàn tất; index đã lưu, không khởi động dashboard.", flush=True)
            return

    warmup_dashboard(reranker_enabled)
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
