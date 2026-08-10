# AIC 2026 Video Retrieval Demo

Web demo cho vòng sơ tuyển AIC 2026, bám theo ba dạng truy vấn trong tài liệu đề bài:

- Textual KIS: trả về danh sách có thứ tự `video_id, frame_id`.
- Q&A: định vị keyframe và sinh câu trả lời VQA baseline từ cùng một ô truy vấn.
- TRAKE: truy xuất **một** video có toàn bộ chuỗi event và căn chỉnh một semantic frame cho mỗi event.

Ưu tiên chính xác truy vấn:

1. Dùng đúng họ model **OpenAI CLIP ViT-B/32**, tương thích feature 512 chiều được BTC cung cấp.
2. Tính cosine similarity trực tiếp trên toàn bộ 873 file feature bằng `numpy.memmap`; không làm giảm dữ liệu, không tạo bản sao index lớn.
3. Tự nhận diện Việt/Anh, dịch tiếng Việt sang tiếng Anh cho CLIP khi dịch vụ sẵn sàng, rồi prompt ensemble.
4. Rerank nhẹ bằng YouTube metadata/object labels và loại các keyframe quá sát nhau để tăng độ phủ cho tối đa 100 đáp án.
5. OCR index tùy chọn: OCR keyframe một lần, nạp inverted index text vào RAM, ưu tiên biển báo/subtitle có chữ khớp truy vấn.

## Không tải dataset

`Code/datasets.py` và toàn bộ app chỉ đọc dữ liệu **đã mount**. Không có API tải dataset, thao tác copy hay build index từ dataset. File [`.gitignore`](.gitignore) chặn mọi feature, video và keyframe khỏi Git.

Trên Kaggle, vào **Add Input** và gắn các public dataset sau trước khi chạy notebook:

| Bắt buộc | Kaggle dataset | Được dùng để |
| --- | --- | --- |
| Có | `doanminhtuan/clip-features-32-aic25-b1` | Tìm kiếm vector CLIP |
| Có | `doanminhtuan/map-keyframes-aic25-b1` | Đổi keyframe sang `frame_id` chuẩn |
| Có | `doanminhtuan/dataset-ai-challenge-keyframe` | Hiển thị gallery kết quả |
| Nên có | `doanminhtuan/media-info-aic25-b1` | Rerank theo tên, mô tả, keyword |
| Nên có | `doanminhtuan/objects-aic25-b1-zip` | Hiển thị/rerank object detection |
| Tùy chọn | `doanminhtuan/video-aic` | Mở đường dẫn video để kiểm tra |

Notebook [run_kaggle.ipynb](run_kaggle.ipynb) là bootstrap cố định. Lần đầu nó clone repo, rồi chỉ chạy [run.py](run.py). Launcher này tự `git pull --ff-only`, tự khởi động lại nếu source vừa đổi, cài dependency khi cần, xóa bytecode/module state cũ và mở dashboard tối màu dạng keyframe-card trong process mới. Vì vậy các cập nhật sau này chỉ sửa source trong repo, **không cần sửa notebook**. Gradio chỉ tạo share link cho Kaggle; toàn bộ giao diện và API tìm kiếm là dashboard riêng. Lần đầu cần bật Internet để cài package/trọng số CLIP; đó là **model cache**, không phải tải dataset. Khi cần chạy offline, mount/cache sẵn checkpoint ViT-B/32 và đặt `AIC_CLIP_CACHE`.

Kaggle đã có sẵn PyTorch đúng CUDA của image notebook. `requirements.txt` cố ý không nâng cấp Torch, vì cài Torch từ PyPI có thể thay NCCL và làm hỏng `libtorch_cuda.so`.

## Chạy trên Kaggle

1. Tạo notebook mới, Add Input các dataset trong bảng trên.
2. Upload/mở `run_kaggle.ipynb`, bật Internet lần đầu rồi Run all.
3. Mở URL dashboard được in ở cell cuối. Chọn KIS, Q&A hoặc TRAKE; mỗi tab chỉ có một ô truy vấn, hỗ trợ Việt/Anh tự động.
4. Chọn các keyframe muốn nộp trong lưới ảnh, sau đó bấm **Xuất CSV**. Tab Q&A dùng ViLT để đề xuất answer; hãy kiểm tra keyframe trước khi nộp.

### OCR cho biển báo và chữ trong video

Để truy vấn như “biển cảnh báo sạt lở nguy hiểm” ưu tiên ảnh có đúng nội dung chữ thay vì chỉ cảnh sạt lở, `run.py` tự pre-OCR ở lần đầu khi chưa có `aic_ocr_index.jsonl.gz`. Nó dùng PaddleOCR `lang="vi"` để tạo file text-only; dashboard nạp file đó vào RAM khi khởi động. Query không chạy OCR trên ảnh và không tải/copy AIC dataset. Một từ điển nhỏ cho biển báo Việt–Anh (ví dụ `dangerous landslide warning` ↔ `cảnh báo sạt lở nguy hiểm`) cũng chạy hoàn toàn trong RAM.

Pre-OCR toàn bộ keyframe có thể mất đáng kể thời gian, vì vậy **phải bật Kaggle Accelerator = GPU** trước khi Run all. `run.py` phát hiện GPU, thay Paddle CPU bằng wheel Paddle GPU CUDA 11.8 khi cần, rồi ép OCR chạy tại `gpu:0`; nếu chưa bật GPU, nó dừng ngay thay vì mất hàng trăm giờ trên CPU. Sau khi hoàn tất, lưu phiên bản Kaggle có file index hoặc đính kèm index ở lần chạy sau; `run.py` tự nhận `/kaggle/working/aic_ocr_index.jsonl.gz` và không build lại nếu index hợp lệ. Để bỏ qua OCR ở một phiên mới, chạy `python /kaggle/working/HCMAIC/run.py --no-build-ocr`. Launcher cũng đặt `AIC_PRELOAD_FEATURES=1`, giữ feature CLIP đã chuẩn hóa trong RAM để giảm thời gian query lặp lại mà không ghi cache feature ra đĩa.

Kaggle thường mount dữ liệu ngay tại `/kaggle/input`. Nếu tên mount khác bố cục chuẩn, đặt các biến: `AIC_FEATURES_DIR`, `AIC_MAPPING_DIR`, `AIC_KEYFRAMES_DIR`, `AIC_METADATA_DIR`, `AIC_OBJECTS_DIR`, `AIC_VIDEOS_DIR`.

## Chạy cục bộ

```bash
python -m pip install -r Code/requirements.txt
export AIC_DATA_ROOT=/duong-dan/toi/input/datasets/doanminhtuan
cd Code
python app.py --host 0.0.0.0 --port 7860
```

Mở `http://localhost:7860`. Giao diện có sidebar, ba tab, Top-K, lọc video, khử trùng theo frame, chọn card và xuất CSV trực tiếp — theo layout dashboard tham chiếu.

Không chạy script tải dataset cũ: nó tạo bản sao dataset rất lớn và không cần thiết cho demo này.

## Ghi chú về đáp án

Theo PDF, một đáp án Textual KIS chỉ đúng khi cả video và `frame_id` nằm trong khoảng GT; Q&A còn cần `answer` đúng ngữ nghĩa; TRAKE nhận 0 nếu sai video. App vì vậy xuất `frame_id` từ CSV mapping của BTC, không dùng số thứ tự keyframe làm đáp án. CSV là format làm việc minh bạch; khi BTC công bố template nộp chính thức, giữ nguyên thứ tự kết quả và đổi header theo template đó nếu cần.
