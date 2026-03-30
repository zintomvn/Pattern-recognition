# Pipeline Thực Nghiệm ANRL với 3 Datasets (CASIA, MSU -> NUAA)

Chào bạn, mình đã refactor (sửa đổi) lại bộ source code của ANRL trong thư mục `src/` để máy tính của bạn hoàn toàn chạy được cấu hình **Trọng số cho 2 Tập Huấn Luyện (CASIA và MSU)** và **Test trên 1 Tập (NUAA)**. Đây là phiên bản custom so với bản gốc 3-tập-train của paper.

## 1. Những file mình đã code & sửa trong thư mục `src/`:
1. **[src/preprocessing.py](file:///d:/%C4%90%E1%BA%A1i%20h%E1%BB%8Dc/K%E1%BB%B3%208/Class%20-%20Pattern%20recognition/Project/Pattern-recognition/src/preprocessing.py) [MỚI TẠO]**: Code All-in-one. Chạy MTCNN (crop khít khuôn mặt từ Video/Ảnh), chạy MiDaS (Sinh Depth 3D Map cho mặt Real), tự động quét tạo File text `.txt` chứa danh sách đường dẫn ném vào `data/processed/meta_lists`.
2. **[src/datasets/DG_dataset.py](file:///d:/%C4%90%E1%BA%A1i%20h%E1%BB%8Dc/K%E1%BB%B3%208/Class%20-%20Pattern%20recognition/Project/Pattern-recognition/src/datasets/DG_dataset.py) [VIẾT LẠI]**: Rút gọn class Dataset chỉ load 2 files text (từ 2 source dataset) thay vì 3 files, xử lý gộp thành Stack 2 tensor. Tích hợp đọc `NUAA_depth`.
3. **[src/train.py](file:///d:/%C4%90%E1%BA%A1i%20h%E1%BB%8Dc/K%E1%BB%B3%208/Class%20-%20Pattern%20recognition/Project/Pattern-recognition/src/train.py) [CẬP NHẬT]**:
   - Khởi tạo Center Embeddings (`center_real`, `center_fake`) từ size 3 xuống size 2.
   - Sửa logic thuật toán Meta-learning tính Loss Khoảng Cực (`Euclidean Discriminative Loss`) loại trừ miền dư. Nó sẽ tự chia (1 tập làm meta-train, 1 tập làm meta-test trong suốt vòng lặp).
4. **[src/configs/CM2N.yaml](file:///d:/%C4%90%E1%BA%A1i%20h%E1%BB%8Dc/K%E1%BB%B3%208/Class%20-%20Pattern%20recognition/Project/Pattern-recognition/src/configs/CM2N.yaml) [MỚI TẠO]**: Định tuyến sẵn đường dẫn `*_list1_path` sang CASIA, `*_list2_path` sang MSU, và list test sang NUAA. Định cỡ `metasize: 1` cho Dataloader 2-chiều.

---

## 2. Quy Trình Chạy (How to Run)

### BƯỚC 1: XẾP THƯ MỤC RAW
Giải nén 3 thư mục tải về vào `data/raw/`. Cấu trúc phải y như thế này (tránh đặt sai tên):
```
Pattern-recognition/
└── data/
    └── raw/
        ├── CASIA_database/
        ├── MSU-MFSD/
        └── NUAA/            (Vì NUAA chứa nhiều ảnh Imposter/Real)
```

### BƯỚC 2: CÀI ĐẶT THƯ VIỆN ĐỘT KÍCH
Mở Terminal gõ lệh này để cài đủ thư viện chạy preprocessing:
```bash
pip install torch torchvision facenet-pytorch opencv-python Pillow numpy omegaconf albumentations wandb
```

### BƯỚC 3: TIỀN XỬ LÝ DỮ LIỆU (PREPROCESSING)
Đây là công đoạn nặng nhất, nhưng mình đã gộp vào 1 file duy nhất. Chuyển terminal vào `src/` và chạy:
```bash
cd src
python preprocessing.py
```
> **Chờ đợi**: Quá trình này sẽ tốn khoảng vài giờ tùy CPU/GPU. Code sẽ quét Video, trích xuất 5 frame/lần, cưa cắt đúng cái mặt, quăng ảnh qua MiDaS giả mạo 3D depth, và in text đường dẫn. Thành phẩm thu được nằm trọn trong `data/processed/`.

### BƯỚC 4: THỰC NGHIỆM TRAINING MODEL
Không cần chần chừ, bạn chỉ việc gõ lệh này để huấn luyện:
```bash
# Ở Windows:
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1234 train.py -c configs/CM2N.yaml
```
(*Lưu ý: tham số `-c configs/CM2N.yaml` trỏ thẳng tới file config mình build sẵn. `nproc_per_node=1` nghĩa là train trên 1 GPU mượt mà.*)

Cứ để máy cắm qua đêm là xong vòng Model Meta-Learning! Chúc bạn chạy ra paper nha.
