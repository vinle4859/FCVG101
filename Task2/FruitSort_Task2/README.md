# FCVG101 – Task 2: Data Pipeline & Deep-Learning Model

**Dự án mẫu có thể chạy bằng Python/PyTorch.** Bài toán ví dụ: phân loại ảnh **apple, banana, orange**.
Không có sẵn dữ liệu ảnh thật trong gói. **Không coi accuracy thu được từ `make_demo_data.py` là kết quả thực nghiệm coursework.**

## 1. Chuẩn bị môi trường

Mở PowerShell/Terminal tại thư mục chứa `train.py`.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu PowerShell chặn Activate.ps1, dùng `.venv\Scripts\python.exe` trực tiếp để chạy thay vì đổi chính sách toàn máy.

## 2. Chuẩn bị dữ liệu thật

```text
data/raw/
  apple/
    item001__view1.jpg
    item002__view1.jpg
    ...
  banana/
    item001__view1.jpg
    ...
  orange/
    item001__view1.jpg
    ...
```

Nếu có nhiều ảnh của cùng một vật/phiên chụp, dùng chung tiền tố `item001` trước `__` để những ảnh ấy nằm cùng tập train/val/test. Cần **ít nhất 3 nhóm độc lập/lớp** cho script chạy; trong thực tế nên có nhiều ảnh và nhóm hơn đáng kể. Cần kiểm tra thủ công near-duplicates, nguồn ảnh, license và nhãn.

## 3. Chạy tuần tự

```bash
python prepare_dataset.py
python train.py --epochs 10 --batch-size 32 --lr 0.001
python evaluate.py
python predict.py --image data/splits/test/apple/ten_file_thuc_te.jpg
```

Đổi đường dẫn ở lệnh `predict.py` thành **một tệp thực tế có trong máy**.

Để thử nhanh sự thông suốt của code khi **chưa có ảnh thật**, có thể chạy:

```bash
python make_demo_data.py
python prepare_dataset.py
python train.py --epochs 2
python evaluate.py
```

Sau khi kiểm tra, xóa dữ liệu tổng hợp trong `data/raw/` và thay bằng dữ liệu thật; **tạo lại split và huấn luyện lại**.

## 4. Đầu ra

- `artifacts/data_manifest.csv`: danh sách ảnh, group và split.
- `artifacts/rejected_images.csv`: nhật ký ảnh hỏng hoặc trùng byte.
- `artifacts/classes.json`: thứ tự class-to-index.
- `artifacts/model.pth`: checkpoint tốt nhất theo validation loss.
- `results/train_history.csv`, `results/learning_curves.png`: kết quả train/val.
- `results/test_metrics.json`, `results/confusion_matrix.png`, `results/test_predictions.csv`: đánh giá test.

## 5. Hạn chế và lưu ý

- Tỷ lệ 70/15/15 được chia **theo nhóm trong từng lớp**, tỷ lệ ảnh có thể lệch nếu nhóm có số ảnh khác nhau.
- Kiểm tra ảnh gần trùng và ảnh cùng phiên chụp bằng mắt hoặc perceptual hash; mã ví dụ chỉ tự bắt **trùng byte** và nhóm theo tiền tố tên tệp.
- Augmentation chỉ chạy trong train loader, không áp dụng lên validation/test.
- Chỉ đánh giá final test **sau khi hoàn tất lựa chọn mô hình/hyperparameters**.
- Softmax output chỉ là **model score**, không tự động là xác suất đã calibration.
- `artifacts/model.pth` phải đi cùng định nghĩa `FruitCNN` và metadata preprocessing; Task 3 gọi `src.inference.load_model` + `predict_pil`.
- Dùng dữ liệu có quyền sử dụng, trích dẫn nguồn; không sử dụng số liệu demo như số liệu thực nghiệm.
