"""Stable Kaggle launcher for the AIC dashboard.

Keep ``run_kaggle.ipynb`` unchanged. This script pulls the latest source,
re-executes itself when it changed, clears source-level Python caches, and
starts the dashboard in a fresh process. The default feature path reads only
Kaggle Inputs; the explicit ``--direct-video`` path may resolve KaggleHub only
when its video mount is absent.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from Code.ocr_regions import OCR_INDEX_SCHEMA_VERSION
from Code.progress import track


REPO = Path(__file__).resolve().parent
CODE = REPO / "Code"
DEFAULT_OCR_INDEX = Path("/kaggle/working/aic_ocr_index.jsonl.gz")
RUNTIME_DIR = Path(os.environ.get("AIC_RUNTIME_DIR", "/kaggle/working"))
RERANKER_REQUIREMENTS = CODE / "requirements-reranker.txt"
QUERY_ANALYZER_REQUIREMENTS = CODE / "requirements-query-analyzer.txt"
DIRECT_VIDEO_REQUIREMENTS = CODE / "requirements-direct-video.txt"
DIRECT_VIDEO_PREPROCESS_REQUIREMENTS = CODE / "requirements-direct-video-preprocess.txt"
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
    "ocr_regions",
    "data_paths",
    "clip_encoder",
    "qa",
    "query_language",
    "query_router",
    "query_analyzer",
    "multimodal_reranker",
    "direct_video_retrieval",
    "preprocess_direct_video",
    "ranking",
    "progress",
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
    cache_dirs = list(CODE.rglob("__pycache__"))
    for cache_dir in track(cache_dirs, desc="Xóa Python cache", unit="dir"):
        shutil.rmtree(cache_dir, ignore_errors=True)
    for module in track(STALE_MODULES, desc="Gỡ module cũ", unit="module"):
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


def install_query_analyzer_requirements() -> None:
    """Install only the small text encoder used to split query evidence."""
    install_if_changed(QUERY_ANALYZER_REQUIREMENTS, ".aic_query_analyzer_requirements.sha256")


def install_direct_video_requirements() -> None:
    """Install KaggleHub only for the explicit raw-video mode."""
    install_if_changed(DIRECT_VIDEO_REQUIREMENTS, ".aic_direct_video_requirements.sha256")


def install_direct_video_preprocess_requirements() -> None:
    """Install the object detector only for the offline raw-video job."""
    install_if_changed(
        DIRECT_VIDEO_PREPROCESS_REQUIREMENTS,
        ".aic_direct_video_preprocess_requirements.sha256",
    )


def install_requirements(
    build_ocr: bool,
    enable_qwen_stack: bool = True,
    enable_query_analyzer: bool = True,
    enable_direct_video: bool = False,
    enable_direct_preprocess: bool = False,
) -> None:
    install_if_changed(CODE / "requirements.txt", ".aic_requirements.sha256")
    if enable_query_analyzer:
        install_query_analyzer_requirements()
    if enable_direct_video:
        install_direct_video_requirements()
    if enable_direct_preprocess:
        install_direct_video_preprocess_requirements()
    if enable_qwen_stack:
        install_reranker_requirements()


def _open_ocr_text(path: Path, mode: str):
    return gzip.open(path, mode, encoding="utf-8") if path.suffix == ".gz" else path.open(mode, encoding="utf-8")


def ocr_index_schema(index_path: Path, marker_path: Path) -> int:
    """Read an OCR schema cheaply from its marker, then first record."""
    try:
        marker_payload = json.loads(marker_path.read_text(encoding="utf-8"))
        marker_schema = int(marker_payload.get("ocr_schema") or 0)
        if marker_schema:
            return marker_schema
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass
    try:
        with _open_ocr_text(index_path, "rt") as stream:
            for line in track(
                stream,
                desc="Đọc OCR schema",
                unit="record",
                nested=True,
            ):
                if line.strip():
                    return int(json.loads(line).get("ocr_schema") or 1)
    except (OSError, EOFError, TypeError, ValueError, json.JSONDecodeError):
        return 0
    return 0


def import_ocr_index(source: Path, destination: Path) -> int:
    """Validate and install a previously exported text-only OCR index."""
    source = source.expanduser().resolve()
    destination = destination.expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Không tìm thấy OCR index để import: {source}")
    records = 0
    schema_version = OCR_INDEX_SCHEMA_VERSION
    try:
        with _open_ocr_text(source, "rt") as stream:
            lines = enumerate(stream, 1)
            for line_number, line in track(
                lines,
                desc="Kiểm tra OCR index",
                unit="record",
                force=True,
                leave=True,
            ):
                payload = json.loads(line)
                if not payload.get("video_id") or "keyframe_number" not in payload:
                    raise ValueError(f"Dòng OCR {line_number} thiếu video_id/keyframe_number")
                schema_version = min(schema_version, int(payload.get("ocr_schema") or 1))
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
        json.dumps(
            {
                "ocr_schema": schema_version,
                "records": records,
                "imported_from": str(source),
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Đã import OCR index v{schema_version}: {records:,} records → {destination}",
        flush=True,
    )
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
    parser.add_argument(
        "--direct-video",
        action="store_true",
        help=(
            "Dùng video-aic raw và tự tạo/cache CLIP index; mặc định tắt, "
            "không dùng feature/mapping BTC"
        ),
    )
    parser.add_argument(
        "--pre-direct-video",
        type=int,
        choices=(0, 1),
        default=0,
        metavar="{0,1}",
        help="1: cắt PNG + CLIP NPY + object + OCR theo shard rồi thoát; mặc định 0",
    )
    parser.add_argument(
        "--start-pre-video",
        type=int,
        default=1,
        help="Video đầu tiên trong thứ tự video_id, 1-based và inclusive",
    )
    parser.add_argument(
        "--end-pre-video",
        type=int,
        default=0,
        help="Video cuối, 1-based và inclusive; 0 nghĩa là tới video cuối corpus",
    )
    parser.add_argument(
        "--pre-direct-fps",
        type=float,
        default=0.0,
        help="Số frame/giây cần lưu; mặc định 0 = tiền xử lý mọi frame",
    )
    parser.add_argument(
        "--pre-direct-gpus",
        default=os.environ.get("AIC_PRE_DIRECT_GPUS", "auto"),
        metavar="GPUS",
        help="GPU cho pre-direct; mặc định auto dùng cả T4x2, hoặc chỉ định 0,1",
    )
    parser.add_argument(
        "--pre-direct-workers",
        type=int,
        default=int(os.environ.get("AIC_PRE_DIRECT_WORKERS", "0")),
        metavar="N",
        help="Số GPU worker; 0 = một worker/GPU đã chọn",
    )
    parser.add_argument(
        "--direct-frame-steps",
        default=os.environ.get("AIC_DIRECT_FRAME_STEPS", "4,2,1"),
        metavar="STEPS",
        help="Các tầng query frame_idx %% step, mặc định 4,2,1",
    )
    parser.add_argument(
        "--pre-direct-output",
        type=Path,
        default=Path(
            os.environ.get(
                "AIC_DIRECT_PREPROCESSED_ROOT",
                "/kaggle/working/aic_direct_preprocessed",
            )
        ),
    )
    parser.add_argument(
        "--pre-direct-max-side",
        type=int,
        default=0,
        help="Resize PNG theo cạnh dài; 0 giữ nguyên độ phân giải",
    )
    parser.add_argument("--force-pre-direct", action="store_true", help="Tạo lại video đã có marker hoàn tất")
    parser.add_argument("--no-reranker", action="store_true", help="Không cài/chạy Qwen multimodal reranker")
    parser.add_argument("--vilt-vqa", action="store_true", help="Dùng ViLT nhẹ thay cho Qwen3-VL VQA")
    return parser.parse_args()


def warmup_dashboard(reranker_enabled: bool, direct_video_enabled: bool = False) -> None:
    """Load query-time resources before Gradio can receive HTTP requests."""
    import dashboard

    if direct_video_enabled:
        print("Đang khởi tạo direct-video engine và local CLIP index trước khi mở dashboard…", flush=True)
    else:
        print("Đang khởi tạo feature engine và OCR index trước khi mở dashboard…", flush=True)
    engine = dashboard.get_engine()
    engine.prepare_runtime()
    ocr_index = dashboard.get_ocr_index()
    if direct_video_enabled:
        direct_ocr = (
            Path(os.environ.get("AIC_DIRECT_PREPROCESSED_ROOT", "")) / "ocr_index.jsonl.gz"
        )
        print(
            "Direct-video mode: bỏ qua features/OCR BTC; "
            f"đã sẵn sàng {engine.vector_count:,} indexed frames · "
            f"direct OCR={'đã nạp' if ocr_index else ('chưa có' if not direct_ocr.is_file() else 'rỗng')}.",
            flush=True,
        )
    print("Đang tải query analyzer nhẹ trước query đầu tiên…", flush=True)
    if dashboard.warmup_query_analyzer():
        analyzer = dashboard.get_query_analyzer()
        print(
            f"Query analyzer đã sẵn sàng ({analyzer.model_name}, device={analyzer.device}).",
            flush=True,
        )
    else:
        detail = dashboard.QUERY_ANALYZER_ERROR or "không có thông tin lỗi"
        print(f"Query analyzer model không sẵn sàng: {detail}", flush=True)
        print("Dashboard sẽ dùng lexical/structural query fallback.", flush=True)
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
        if not dashboard.warmup_reranker():
            detail = dashboard.RERANKER_ERROR or "không có thông tin lỗi"
            print(f"Qwen reranker không sẵn sàng: {detail}", flush=True)
            print("Dashboard sẽ dùng CLIP/OCR fallback.", flush=True)
        else:
            print("Qwen reranker đã warmup bằng một keyframe thật.", flush=True)
    if os.environ.get("AIC_PRELOAD_VQA", "1").lower() not in {"0", "false", "no"}:
        try:
            print("Đang tải VQA model trước query đầu tiên…", flush=True)
            backend = dashboard.warmup_vqa()
            print(f"VQA đã warmup bằng keyframe thật: {backend}.", flush=True)
            if dashboard.get_vqa().load_error:
                print(dashboard.get_vqa().load_error, flush=True)
        except Exception as error:
            print(f"VQA preload bỏ qua: {error}", flush=True)


def main() -> None:
    arguments = parse_arguments()
    pre_direct_enabled = arguments.pre_direct_video == 1
    if pre_direct_enabled:
        if arguments.start_pre_video < 1:
            raise ValueError("--start-pre-video dùng chỉ số 1-based và phải >= 1.")
        if arguments.end_pre_video != 0 and arguments.end_pre_video < arguments.start_pre_video:
            raise ValueError(
                "--end-pre-video phải >= --start-pre-video; dùng 0 để chạy tới video cuối."
            )
        if not math.isfinite(arguments.pre_direct_fps) or arguments.pre_direct_fps < 0:
            raise ValueError("--pre-direct-fps phải >= 0; dùng 0 để lấy tất cả frame.")
        if arguments.pre_direct_max_side < 0:
            raise ValueError("--pre-direct-max-side phải >= 0.")
        if arguments.pre_direct_workers < 0:
            raise ValueError("--pre-direct-workers phải >= 0; 0 nghĩa là một worker/GPU.")
    direct_video_enabled = (
        arguments.direct_video
        or pre_direct_enabled
        or os.environ.get("AIC_DIRECT_VIDEO", "0").lower() in {"1", "true", "yes", "on"}
    )
    if direct_video_enabled:
        os.environ["AIC_DIRECT_VIDEO"] = "1"
        if arguments.pre_ocr or arguments.build_ocr:
            raise ValueError(
                "--direct-video dùng raw video và local CLIP index; "
                "không dùng đồng thời pre-OCR/build-ocr."
            )
        if arguments.import_ocr:
            # The protected Kaggle notebook always passes --import-ocr. Keep
            # that notebook reusable: raw-video mode must not mix the BTC OCR
            # keyframe ids, so skip the import instead of forcing a notebook edit.
            print("Direct-video: bỏ qua --import-ocr vì OCR BTC đang bị tắt.", flush=True)
            arguments.import_ocr = None
        if os.environ.get("AIC_BUILD_OCR", "0").lower() in {"1", "true", "yes", "on"}:
            raise ValueError("AIC_BUILD_OCR không áp dụng cho --direct-video; hãy bỏ biến này.")
    else:
        os.environ["AIC_DIRECT_VIDEO"] = "0"
    if arguments.pre_ocr and arguments.import_ocr:
        raise ValueError("Không dùng đồng thời --pre-ocr và --import-ocr.")
    if arguments.pre_ocr and arguments.no_build_ocr:
        raise ValueError("--pre-ocr cần chạy OCR; không dùng cùng --no-build-ocr.")
    update_source(arguments.skip_update)
    if arguments.import_ocr:
        import_ocr_index(arguments.import_ocr, arguments.ocr_index)
    complete_marker = Path(f"{arguments.ocr_index}.complete")
    index_schema = (
        ocr_index_schema(arguments.ocr_index, complete_marker)
        if arguments.ocr_index.is_file() and complete_marker.is_file()
        else 0
    )
    index_ready = index_schema >= OCR_INDEX_SCHEMA_VERSION
    if index_schema and not index_ready:
        print(
            f"OCR index v{index_schema} còn chứa subtitle/ticker; cần rebuild v{OCR_INDEX_SCHEMA_VERSION}.",
            flush=True,
        )
    forced_ocr = (
        arguments.build_ocr
        or arguments.pre_ocr
        or os.environ.get("AIC_BUILD_OCR", "0").lower() in {"1", "true", "yes"}
    )
    # Accuracy-first default: a fresh Kaggle session pre-OCRs once. Subsequent
    # Run all executions reuse the compact text index and launch immediately.
    build_ocr = not direct_video_enabled and (forced_ocr or (not arguments.no_build_ocr and not index_ready))
    if arguments.no_build_ocr and index_schema and not index_ready:
        print(
            "[warning] --no-build-ocr giữ index cũ; runtime chỉ có thể dùng ticker fallback, "
            "không chính xác bằng pre-OCR v2.",
            flush=True,
        )
    clear_source_cache()
    reranker_enabled = (
        not arguments.no_reranker
        and not arguments.pre_ocr
        and not pre_direct_enabled
        and os.environ.get("AIC_RERANKER", "1").lower() not in {"0", "false", "no"}
    )
    if arguments.vilt_vqa:
        os.environ["AIC_VQA_BACKEND"] = "vilt"
    preload_vqa = os.environ.get("AIC_PRELOAD_VQA", "1").lower() not in {"0", "false", "no"}
    default_vqa_backend = "qwen" if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") else "vilt"
    qwen_vqa_enabled = (
        not arguments.pre_ocr
        and not pre_direct_enabled
        and preload_vqa
        and os.environ.get("AIC_VQA_BACKEND", default_vqa_backend).lower() == "qwen"
    )
    install_requirements(
        build_ocr,
        reranker_enabled or qwen_vqa_enabled,
        enable_query_analyzer=not arguments.pre_ocr and not pre_direct_enabled,
        enable_direct_video=direct_video_enabled,
        enable_direct_preprocess=pre_direct_enabled,
    )

    os.environ["AIC_DATA_ROOT"] = str(Path(arguments.data_root).expanduser())
    os.environ["AIC_OCR_INDEX"] = str(arguments.ocr_index)
    os.environ["AIC_DIRECT_PREPROCESSED_ROOT"] = str(arguments.pre_direct_output.expanduser())
    os.environ["AIC_DIRECT_FRAME_STEPS"] = arguments.direct_frame_steps
    os.environ["AIC_PRELOAD_FEATURES"] = "0" if arguments.no_preload_features or direct_video_enabled else "1"
    os.environ["AIC_RERANKER"] = "1" if reranker_enabled else "0"
    # Keep site-packages ahead of the app directory. The project has a
    # compatibility module named ``Code/datasets.py``; putting Code first
    # shadows Hugging Face's external ``datasets`` package when
    # sentence-transformers imports it.
    code_path = str(CODE)
    while code_path in sys.path:
        sys.path.remove(code_path)
    sys.path.append(code_path)

    if direct_video_enabled and not pre_direct_enabled:
        from direct_video_retrieval import parse_frame_steps

        parsed_steps = parse_frame_steps(arguments.direct_frame_steps)
        print(
            "Direct query hierarchy: "
            + " → ".join(f"frame_id % {step}" for step in parsed_steps),
            flush=True,
        )

    if pre_direct_enabled:
        common = [
            sys.executable,
            str(CODE / "preprocess_direct_video.py"),
            "--output",
            str(arguments.pre_direct_output.expanduser()),
            "--start-video",
            str(arguments.start_pre_video),
            "--end-video",
            str(arguments.end_pre_video),
            "--sample-fps",
            str(arguments.pre_direct_fps),
            "--max-side",
            str(arguments.pre_direct_max_side),
            "--gpus",
            arguments.pre_direct_gpus,
            "--workers",
            str(arguments.pre_direct_workers),
        ]
        if arguments.force_pre_direct:
            common.append("--force")
        frame_plan = (
            "mọi decoded frame"
            if arguments.pre_direct_fps == 0
            else f"{arguments.pre_direct_fps:g} frame/giây"
        )
        print(
            f"Direct preprocess GPU={arguments.pre_direct_gpus}, "
            f"workers={arguments.pre_direct_workers or 'auto'}.",
            flush=True,
        )
        print(
            f"Direct preprocess 1/3 ({frame_plan}): cắt PNG, CLIP embedding và YOLO object…",
            flush=True,
        )
        command([*common, "--stage", "visual"])
        print("Direct preprocess 2/3: PaddleOCR scene-text, loại TV overlay…", flush=True)
        ocr_env = ensure_paddle_ocr_packages()
        command(
            [*common, "--stage", "ocr", "--ocr-device", arguments.ocr_device],
            env=ocr_env,
        )
        print("Direct preprocess 3/3: hợp nhất manifest/OCR/object index…", flush=True)
        command([*common, "--stage", "finalize"])
        print(
            f"Pre-direct-video hoàn tất đoạn [{arguments.start_pre_video}, "
            f"{arguments.end_pre_video or 'cuối'}] → {arguments.pre_direct_output.expanduser()}",
            flush=True,
        )
        return

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

    warmup_dashboard(reranker_enabled, direct_video_enabled)
    import dashboard
    from share_dashboard import launch_dashboard

    if direct_video_enabled:
        engine = dashboard.get_engine()
        print("Direct video root:", engine.dataset_root, flush=True)
        print("Direct indexed frames:", f"{engine.vector_count:,}", flush=True)
        print(
            "Direct query steps:",
            (
                " → ".join(map(str, engine.frame_steps))
                if engine._preprocessed_video_count
                else f"fallback stride {engine.sample_stride}"
            ),
            flush=True,
        )
        print("BTC features/mapping:", "tắt trong direct-video mode", flush=True)
        direct_ocr_path = Path(os.environ["AIC_DIRECT_PREPROCESSED_ROOT"]) / "ocr_index.jsonl.gz"
        print("Direct OCR index:", direct_ocr_path if direct_ocr_path.is_file() else "chưa có", flush=True)
    else:
        from data_paths import AICPaths

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
