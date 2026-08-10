# AIC 2026 Video Retrieval Demo

Web demo cho vòng sơ tuyển AIC 2026, bám theo ba dạng truy vấn trong tài liệu đề bài:

- Textual KIS: trả về danh sách có thứ tự `video_id, frame_id`.
- Q&A: định vị keyframe, hiển thị object/keyframe để người dùng kiểm tra rồi nhập `answer`.
- TRAKE: truy xuất **một** video có toàn bộ chuỗi event và căn chỉnh một semantic frame cho mỗi event.

Ưu tiên chính xác truy vấn:

1. Dùng đúng họ model **OpenAI CLIP ViT-B/32**, tương thích feature 512 chiều được BTC cung cấp.
2. Tính cosine similarity trực tiếp trên toàn bộ 873 file feature bằng `numpy.memmap`; không làm giảm dữ liệu, không tạo bản sao index lớn.
3. Prompt ensemble và ô English expansion cho CLIP (rất nên dùng với câu hỏi tiếng Việt).
4. Rerank nhẹ bằng YouTube metadata/object labels và loại các keyframe quá sát nhau để tăng độ phủ cho tối đa 100 đáp án.

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

Notebook [run_kaggle.ipynb](run_kaggle.ipynb) thực hiện `git clone` (lần đầu), `git pull --ff-only`, cài dependency và mở Gradio. Lần đầu cần bật Internet để cài package/trọng số CLIP; đó là **model cache**, không phải tải dataset. Khi cần chạy offline, mount/cache sẵn checkpoint ViT-B/32 và đặt `AIC_CLIP_CACHE`.

## Chạy trên Kaggle

1. Tạo notebook mới, Add Input các dataset trong bảng trên.
2. Upload/mở `run_kaggle.ipynb`, bật Internet lần đầu rồi Run all.
3. Vào link Gradio, nhập mô tả. Với truy vấn tiếng Việt, điền bản diễn đạt tiếng Anh chính xác ở ô kế bên.
4. Kiểm tra gallery, sau đó tải CSV có thứ tự xếp hạng. Với Q&A, nhập answer sau khi đối chiếu kết quả.

Kaggle thường mount dữ liệu ngay tại `/kaggle/input`. Nếu tên mount khác bố cục chuẩn, đặt các biến: `AIC_FEATURES_DIR`, `AIC_MAPPING_DIR`, `AIC_KEYFRAMES_DIR`, `AIC_METADATA_DIR`, `AIC_OBJECTS_DIR`, `AIC_VIDEOS_DIR`.

## Chạy cục bộ

```bash
python -m pip install -r Code/requirements.txt
export AIC_DATA_ROOT=/duong-dan/toi/input/datasets/doanminhtuan
cd Code
python app.py --share
```

Không chạy script tải dataset cũ: nó tạo bản sao dataset rất lớn và không cần thiết cho demo này.

## Ghi chú về đáp án

Theo PDF, một đáp án Textual KIS chỉ đúng khi cả video và `frame_id` nằm trong khoảng GT; Q&A còn cần `answer` đúng ngữ nghĩa; TRAKE nhận 0 nếu sai video. App vì vậy xuất `frame_id` từ CSV mapping của BTC, không dùng số thứ tự keyframe làm đáp án. CSV là format làm việc minh bạch; khi BTC công bố template nộp chính thức, giữ nguyên thứ tự kết quả và đổi header theo template đó nếu cần.
