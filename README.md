# HCMAIC — AIC 2026 Retrieval Workspace

Hệ thống tìm kiếm video cho vòng sơ tuyển AIC 2026, ưu tiên **độ chính xác của từng query trước**, sau đó mới tối ưu thời gian phản hồi. Dashboard hỗ trợ đủ ba bài toán trong tài liệu BTC:

| Bài toán | Kết quả làm việc | Điều kiện đúng |
| --- | --- | --- |
| Textual KIS | Danh sách có hạng `video_id, frame_id` | Đúng video và frame nằm trong đoạn GT |
| Q&A | `video_id, frame_id, answer` | Đúng vị trí và answer đúng ngữ nghĩa |
| TRAKE | Một video cùng chuỗi `frame_id_1..N` có thứ tự | Sai video được 0; mỗi event đúng nhận điểm thành phần |

Mỗi query được phép tối đa 100 đáp án. Final score là trung bình của best R-score tại `k = 1, 5, 20, 50, 100`, vì vậy hệ thống vừa tập trung Top-1/Top-5 vừa giữ độ phủ tới Top-100.

## Pipeline hiện tại

Trước khi mở dashboard, launcher làm trước toàn bộ phần không phụ thuộc query:

1. Fast-forward source và tự restart process nếu `run.py` vừa thay đổi.
2. Cài/probe đúng stack Qwen với `transformers>=5,<6`, `sentence-transformers>=5.4` và `qwen-vl-utils`.
3. Import hoặc tạo OCR index v2 một lần bằng PaddleOCR GPU; tọa độ box loại lower-third, subtitle, ticker, logo và đồng hồ TV khỏi scene-text.
4. Nạp và chuẩn hóa toàn bộ CLIP feature vào RAM; cache mapping frame, thư mục keyframe/video và metadata.
5. Precompute BM25/IDF cho 164 nghìn OCR record và toàn bộ metadata video.
6. Nạp CLIP, Qwen3-VL-Reranker và Qwen3-VL VQA; chạy warmup bằng keyframe thật.
7. Chỉ sau đó mới mở Gradio share URL.

Hot path của KIS/Q&A:

1. Qwen đọc query ở chế độ text-only và trả tỷ trọng **Hình ảnh / OCR / Metadata / Object**; lexical Việt–Anh là lớp hiệu chỉnh/fallback.
2. CLIP ViT-B/32 recall toàn corpus bằng một matrix multiplication; OCR BM25 và metadata BM25 bổ sung candidate mà CLIP có thể bỏ sót.
3. Tỷ trọng query điều khiển score fusion và quota candidate của từng nguồn.
4. Pool Qwen được khử trùng mềm theo video/thời gian để không lãng phí 32 slot vào các frame gần giống nhau.
5. `Qwen/Qwen3-VL-Reranker-2B` chấm query + ảnh đã làm mờ lower-third trong RAM + scene OCR. Qwen joint chiếm 80% điểm cuối; 20% còn lại dùng đúng tỷ trọng bằng chứng do query router chọn.
6. Kết quả cuối mới áp frame gap, giới hạn mỗi video và Top-K.

TRAKE dùng một phép nhân ma trận cho tất cả event, Viterbi để bắt buộc semantic frame tăng theo thời gian, Qwen center-rerank theo đúng vị trí của từng event để chọn video, rồi soi lân cận quanh các chuỗi tốt nhất và dynamic-align lại. Event “tiếp đất” được soi rộng hơn một keyframe để tìm lần tiếp xúc đầu tiên sau trạng thái trên không. Q&A dùng `Qwen/Qwen3-VL-2B-Instruct` để trả lời ngắn trên các frame đã được chọn theo cả sự kiện và câu hỏi; nếu T4 không đủ VRAM hoặc model lỗi, nó tự giải phóng model và chuyển sang ViLT.

Các model Qwen được dùng theo API chính thức: [Qwen3-VL-Reranker-2B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B) và [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct).

## Dataset Kaggle

Project chỉ đọc dữ liệu đã mount, không có code tải/copy AIC dataset. Thêm các input sau vào notebook:

| Mức độ | Kaggle dataset | Công dụng |
| --- | --- | --- |
| Bắt buộc | `doanminhtuan/clip-features-32-aic25-b1` | CLIP recall |
| Bắt buộc | `doanminhtuan/map-keyframes-aic25-b1` | Đổi keyframe sang frame ID chính thức |
| Bắt buộc | `doanminhtuan/dataset-ai-challenge-keyframe` | Ảnh gallery và Qwen rerank |
| Nên có | `doanminhtuan/media-info-aic25-b1` | Metadata BM25 |
| Nên có | `doanminhtuan/objects-aic25-b1-zip` | Object support score |
| Nên có | `doanminhtuan/video-aic` | Mở video tại timestamp để xác minh frame |

Nếu mount có bố cục khác, dùng `AIC_FEATURES_DIR`, `AIC_MAPPING_DIR`, `AIC_KEYFRAMES_DIR`, `AIC_METADATA_DIR`, `AIC_OBJECTS_DIR`, `AIC_VIDEOS_DIR` hoặc đặt root qua `AIC_DATA_ROOT`.

## Chạy trên Kaggle

1. Bật **Accelerator = GPU** và Internet ở lần chạy đầu.
2. Add Input các dataset ở bảng trên.
3. Mở `run_kaggle.ipynb` rồi Run all. Notebook là bootstrap cố định; các lần sau `run.py` tự pull code mới.
4. Chờ prewarm hoàn tất, sau đó mở đúng URL được in ở dòng `Open dashboard: .../dashboard/`.

Một startup thành công cần có các dòng tương tự:

```text
Qwen reranker đã sẵn sàng (transformers 5.x ...).
PyTorch ...; CUDA available=True; GPU=Tesla T4
Qwen reranker đã warmup bằng một keyframe thật.
VQA đã warmup bằng keyframe thật: Qwen3-VL-2B-Instruct.
Open dashboard: https://....gradio.live/dashboard/
```

Lần đầu có thể lâu vì tải khoảng 4.3 GB cho reranker, thêm checkpoint VQA và pre-OCR corpus. Đây là startup/precompute; query không OCR ảnh lại và không nạp model lại.

Mọi phase có vòng lặp dài đều hiện `tqdm` ngay trong output cell Kaggle, gồm kiểm tra/import OCR, dựng postings/IDF, preload CLIP, recall, Qwen batch, VQA và TRAKE refinement. Bar có nhãn phase, số item đã xử lý, tốc độ và ETA nên có thể phân biệt model đang tính với process bị lỗi.

### Dùng OCR index đã tạo sẵn

```bash
# Chỉ tạo OCR index rồi thoát
python /kaggle/working/HCMAIC/run.py --pre-ocr \
  --ocr-index /kaggle/working/aic_ocr_index.jsonl.gz

# Phiên sau import text index đã lưu/mount
python /kaggle/working/HCMAIC/run.py \
  --import-ocr /kaggle/input/my-ocr/aic_ocr_index.jsonl.gz
```

`--import-ocr` đọc và kiểm tra toàn bộ JSONL trước khi tạo marker hoàn tất. Muốn mở nhanh mà chưa có OCR dùng `--no-build-ocr`; độ chính xác truy vấn chữ sẽ giảm.

OCR index v2 chỉ giữ chữ thuộc vật thể/cảnh. Mặc định các OCR box ở 18% đáy màn hình và logo/đồng hồ ở góc bị loại. Nếu launcher thấy index v1 cũ không có tọa độ, nó tự rebuild v2; `--no-build-ocr` chỉ dùng bộ lọc câu ticker dự phòng và kém chính xác hơn. Sau khi build xong, hãy xuất/mount file v2 mới thay cho dataset OCR cũ nếu muốn tái sử dụng ở session khác.

## Sử dụng dashboard

- KIS: mô tả một khoảnh khắc bằng Việt hoặc Anh.
- Q&A: nhập `mô tả sự kiện. Câu hỏi: ...` để phần trước tìm frame và phần sau đi vào VQA.
- TRAKE: mỗi dòng là một event, theo đúng thứ tự thời gian.
- Thanh tỷ trọng ngay trên kết quả cho biết model đã chọn bao nhiêu phần trăm Hình/OCR/Metadata/Object.
- Nhấp card để chọn; nhấp đúp để mở inspector.
- Inspector cho mở video gốc đúng timestamp, sửa `frame_id` nộp và sửa answer Q&A.
- Chọn Top 5/20/100 hoặc từng card, sau đó xuất CSV. Override được gửi về backend và ghi thật vào file, không chỉ đổi nhãn trên trình duyệt.

Giao diện sáng, dạng workspace ngang, responsive; thang Top-1/5/20/50/100 luôn hiển thị để nhắc đúng cách chấm của BTC.

## Vì sao không còn HTTP 504

`POST /dashboard/api/search` chỉ tạo GPU job và trả `202 JSON` ngay. Trình duyệt polling một status URL bằng các request ngắn trong lúc Qwen chạy nền. GPU queue mặc định chỉ có một worker để tránh hai model inference đồng thời gây OOM trên T4. Vì vậy có thể dùng accuracy-first rerank lâu hơn giới hạn request của Gradio mà không nhận HTML 504.

Mọi lỗi nằm trong Flask API cũng trả JSON, không trả error page HTML. Frontend vẫn kiểm tra `content-type` để báo rõ nếu người dùng mở sai mount hoặc share gateway bên ngoài app gặp lỗi.

## Cấu hình accuracy / latency

| Biến | Mặc định Kaggle | Ý nghĩa |
| --- | --- | --- |
| `AIC_PRELOAD_FEATURES` | `1` | Giữ ma trận CLIP đã chuẩn hóa trong RAM |
| `AIC_RERANKER` | `1` | Bật Qwen3-VL-Reranker |
| `AIC_RERANKER_CANDIDATES` | `32` | Số ảnh KIS/Q&A Qwen chấm; tăng tối đa 100 |
| `AIC_RERANKER_BATCH_SIZE` | `2` | Batch T4; OOM tự retry bằng 1 |
| `AIC_RERANKER_CACHE` | `512` | LRU cache cặp query/frame |
| `AIC_TRAKE_RERANK_PAIRS` | `32` | Ngân sách center event–frame |
| `AIC_TRAKE_REFINE_RADIUS` | `2` | Bán kính keyframe refinement; event tiếp đất tự rộng thêm 1, tối đa 5 |
| `AIC_TRAKE_REFINE_SEQUENCES` | `3` | Số chuỗi được refinement |
| `AIC_VQA_BACKEND` | `qwen` | Dùng `vilt` nếu cần nhẹ VRAM |
| `AIC_VQA_CANDIDATES` | `6` | Số frame Qwen VQA trả lời, tối đa 12 |
| `AIC_PRELOAD_VQA` | `1` | Nạp/warmup VQA trước khi mở web |
| `AIC_SEARCH_WORKERS` | `1` | Số GPU job đồng thời; giữ 1 trên T4 |
| `AIC_SEARCH_QUEUE_LIMIT` | `8` | Chặn public queue tăng vô hạn |
| `AIC_TRANSLATE_VI` | `1` | Dịch VI→EN có cache cho CLIP/VQA |
| `AIC_OCR_BOTTOM_OVERLAY_START` | `0.82` | Bỏ OCR box từ 82% chiều cao ảnh trở xuống |
| `AIC_OCR_LEGACY_MIN_QUALITY` | `0.20` | Ngưỡng loại ticker cho OCR index v1 không có tọa độ |
| `AIC_RERANKER_MASK_OVERLAYS` | `1` | Làm mờ lower-third trong RAM trước khi Qwen chấm ảnh |
| `AIC_PROGRESS` | `1` | Hiện progress bar cho startup và từng query; đặt `0` để tắt |
| `AIC_PROGRESS_MIN_ITEMS` | `20` | Ẩn các bar cực nhỏ để output dễ đọc |
| `AIC_PROGRESS_ALL` | `0` | Đặt `1` để hiện cả mọi vòng lặp nhỏ/lồng nhau khi debug |

Muốn soi chi tiết tuyệt đối một query, chạy trước launcher:

```bash
export AIC_PROGRESS_ALL=1
python run.py
```

Preset nhanh khi cần kiểm tra UI:

```bash
export AIC_RERANKER_CANDIDATES=12
export AIC_VQA_BACKEND=vilt
python run.py --no-build-ocr
```

Accuracy cao hơn, chấp nhận query lâu hơn:

```bash
export AIC_RERANKER_CANDIDATES=48
export AIC_TRAKE_RERANK_PAIRS=48
export AIC_TRAKE_REFINE_SEQUENCES=3
python run.py
```

### Đo latency thật trên Kaggle T4

Dừng cell dashboard trước để không giữ hai bản model trong VRAM, rồi chạy benchmark trong repo. Lần chạy đầu trong report là warm query sau startup; các lần sau còn cho thấy hiệu quả LRU cache:

```bash
cd /kaggle/working/HCMAIC
PYTHONPATH=Code python Code/benchmark.py --task kis --repeat 3

PYTHONPATH=Code python Code/benchmark.py --task trake --repeat 2 --query $'Event 1\nEvent 2\nEvent 3'
```

Output JSON tách `startup_seconds`, mean/min query time, kích thước corpus, model backend, tỷ trọng query và Top-1 của từng run. Đây là số cần dùng để chọn `AIC_RERANKER_CANDIDATES` trên chính session/dataset của bạn; benchmark local không đại diện cho Tesla T4.

## Troubleshooting

### `The 'any-to-any' transformer task requires transformers v5+`

Dừng cell cũ rồi Run all lại. `requirements-reranker.txt` hiện khóa `transformers>=5.0.0,<6.0.0`; launcher dùng hash marker nên sẽ cài lại khi file dependency đổi. Không tiếp tục dùng process đã import Transformers 4.x.

### `cannot import name 'Dataset' from .../Code/datasets.py`

Đây không phải thiếu Kaggle Input. File compatibility nội bộ `Code/datasets.py` từng che package Hugging Face `datasets`. `run.py` hiện luôn để site-packages đứng trước `Code` trong `sys.path` và restart sau pull. Dừng kernel/cell cũ rồi chạy lại launcher từ repo root.

### API 404 hoặc `Unexpected token '<'`

Mở đúng URL có dấu slash cuối: `https://...gradio.live/dashboard/`. Nếu Network vẫn cho thấy một `POST /dashboard/api/search` trả 504 trên bản mới, hard refresh để bỏ JavaScript cache; endpoint mới phải trả `202` cùng `job_id/status_url` gần như ngay lập tức.

### Qwen không sẵn sàng

Kiểm tra log phải có `CUDA available=True`, Transformers 5.x và keyframe thật khi warmup. Nếu lỗi VRAM, thử:

```bash
export AIC_RERANKER_BATCH_SIZE=1
export AIC_RERANKER_CANDIDATES=20
python run.py --vilt-vqa
```

Muốn giữ dashboard hoạt động không Qwen dùng `AIC_RERANKER=0` hoặc `--no-reranker`; app sẽ fallback CLIP/OCR và hiển thị rõ trạng thái.

### Top-1 là cảnh không liên quan nhưng ticker có đúng query

Health/API và notice phải hiển thị `OCR RAM v2`. Nếu vẫn là v1, chạy lại launcher không dùng `--no-build-ocr` để hệ thống pre-OCR lại. V2 loại chữ theo tọa độ trước khi index; ảnh đưa vào Qwen cũng được làm mờ phần lower-third nên model không thể đọc lại ticker từ pixel gốc.

## Kiểm thử và evaluator theo PDF

Chạy regression suite:

```bash
python -m pip install -r Code/requirements.txt
PYTHONPATH=Code python -m unittest discover -s tests -v
```

`Code/evaluation.py` cài đúng KIS, Q&A, TRAKE partial credit và Final score trong PDF. Ground truth local dạng JSON, ví dụ KIS:

```json
{"task":"kis","video_id":"L01_V001","start":500,"end":510}
```

```bash
PYTHONPATH=Code python Code/evaluation.py predictions.csv ground_truth.json
```

Tests có các ví dụ công thức chính thức: TRAKE đúng 3/4 event nhận `0.75`, và dãy R-score mẫu cho Final score `0.74`.

## Chạy local

```bash
python -m pip install -r Code/requirements.txt
export AIC_DATA_ROOT=/duong-dan/toi/input
cd Code
python app.py --host 0.0.0.0 --port 7860
```

Local mặc định nhẹ: không bật Qwen reranker và dùng ViLT. Dataset vẫn phải tồn tại ở disk. Không chạy script tải dataset cũ; `.gitignore` chặn feature, video và keyframe khỏi Git.

## Giới hạn thực tế

- Không có model nào bảo đảm đúng 100%; hãy dùng video inspector và override trước khi nộp các rank quan trọng.
- CSV là format làm việc minh bạch theo trường trong PDF. Khi BTC công bố submission template/API cuối cùng, giữ nguyên thứ tự và đổi header/transport theo template đó.
- Local test xác minh logic, API, JavaScript, scoring và fallback; hiệu năng/Qwen CUDA cuối cùng phải được benchmark trên chính Kaggle T4 cùng dataset đã mount.
