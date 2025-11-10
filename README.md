# 🎯 Face Recognition System

Hệ thống nhận dạng khuôn mặt với InsightFace - Professional & Performance Testing Ready

## ⚡ Quick Start

```bash
source .face/bin/activate
python3 face_capture.py --name TenBan
python3 recognize.py
```

**Done!** Nhấn `q` để thoát.

---

## 📝 Main Commands

### 1. Face Capture (AUTO)
```bash
python3 face_capture.py --name TenNguoi
```
**Tự động:** Chụp ảnh → Extract embedding → Lưu database

### 2. Face Recognition
```bash
python3 recognize.py
```
**Làm gì:** Mở camera → Nhận dạng → Hiển thị tên + FPS

### 3. **NEW** Performance Testing
```bash
python3 performance_testing_suite.py
```
**Features:** 
- Generate 100 synthetic people for testing
- Compare database performance (8 vs 108 people) 
- Quality testing of synthetic embeddings
- Interactive testing menu

---

## 🧪 Performance Testing Features

### Generate Synthetic Database
```bash
python3 generate_duplicate_embeddings.py --source db_embedding/embed_s.h5 --num 100
```
Creates 108-person database (8 real + 100 synthetic) for performance testing

### Run Benchmarks
```bash
python3 benchmark_performance.py
```
Compare loading and search performance between databases

### Test Embedding Quality
```bash
python3 test_synthetic_quality.py
```
Verify synthetic embeddings are unique and properly generated

### Camera Test with Large Database
```bash
python3 test_duplicate_recognition.py
```
Real-time recognition test with 108-person database

---

## 📁 Project Structure



## ⚙️ Config nhanh- `--model antelopev2` - Dùng model antelopev2



**File:** `config/default.json`- `--rtsp "rtsp://..."` - Từ RTSP camera



```json

{

  "recognition": {---```bash```

    "threshold": 0.3,    // Giảm = dễ nhận dạng

    "gpu_id": 0          // -1 = CPU only

  }

}### #2: Nhận dạng khuôn mặt (Simple)# Chụp ảnh + Extract embeddings + Cập nhật DB (Tất cả tự động!)Face_Recognition/

```

```bash

---

python3 run_recognize.py --model buffalo_lpython3 capture_face.py --name TenNguoi├── face_capture.py          # Capture and enroll faces

## 🎮 Controls

```

- `q` - Thoát

**Làm gì:** Mở camera → Nhận dạng → Hiển thị tên + FPS├── face_extract.py          # Extract embeddings from images

---



## 📚 Docs

**Options:**# Chụp nhiều ảnh hơn├── recognize.py             # Main recognition program

📖 **Đọc CHEAT_SHEET.md** - Tất cả lệnh + examples + troubleshooting

- `--model antelopev2` - Dùng antelopev2

Hoặc chạy: `./commands.sh`

- `--threshold 0.25` - Thay đổi độ chính xácpython3 capture_face.py --name TenNguoi --num 30├── recognize_optimized.py   # Recognition with optimized models

---

- `--rtsp "rtsp://..."` - Từ RTSP

## 🔧 Troubleshooting

├── face_recognition.py      # Enhanced recognition system

| Vấn đề | Giải pháp |

|--------|-----------|---

| FPS thấp | `det_size: [320, 320]` trong config |

| Không nhận dạng | `threshold: 0.25` hoặc chụp thêm ảnh |# Dùng model antelopev2├── check_embedding.py       # Check database quality

| Camera lỗi | `--camera 1` |

### #3: Nhận dạng (Detailed - có timing)

---

```bashpython3 capture_face.py --name TenNguoi --model antelopev2├── optimize_onnx.py         # ONNX model optimization tool

## 💡 Tips

python3 run_recognize_detailed.py --model buffalo_l

- Chụp 20-30 ảnh/người

- Nhiều góc độ + ánh sáng``````├── benchmark_compare.py     # Performance benchmark tool

- Khoảng cách: 1-3m

**Thêm:** Thống kê thời gian Detection/Recognition/Identification

---

├── README.md                # This guide

**⚡ Xem chi tiết: CHEAT_SHEET.md hoặc ./commands.sh**

---

**Chế độ TỰ ĐỘNG:**├── face_db/                 # Directory for face images

## ⚙️ Config nhanh

- 🤖 Tự động chụp mỗi 1 giây khi phát hiện khuôn mặt├── optimized_models/        # Optimized ONNX models

**File:** `config/default.json`

- 💾 Tự động lưu ảnh vào `face_db/`└── embeddings_db.h5         # Face database file

```json

{- 🔄 Tự động extract embeddings```

  "recognition": {

    "threshold": 0.3,    // Giảm = dễ nhận dạng- 📊 Tự động cập nhật database

    "gpu_id": 0          // -1 = CPU only

  }- ✅ Sẵn sàng dùng ngay!## 🔧 Installation

}

```



---### 3. Chạy nhận dạng### 1. Requirements



## 🎮 Controls```bash



- `q` - Thoát```bashpip install opencv-python numpy insightface h5py scikit-learn onnxruntime-gpu



---# Simple mode (chỉ FPS)```



## 📚 Docspython3 run_recognize.py --model buffalo_l



| File | Nội dung |For TensorRT acceleration (optional):

|------|----------|

| **CHEAT_SHEET.md** | Tất cả lệnh + examples |# Detailed mode (có timing breakdown)```bash

| **QUICK_REFERENCE.md** | Tóm tắt ngắn gọn |

| **USER_GUIDE.md** | Hướng dẫn chi tiết |python3 run_recognize_detailed.py --model buffalo_lpip install nvidia-tensorrt

| **ARCHITECTURE.md** | Developer guide |

```

---

# Từ RTSP camera

## 🔧 Troubleshooting

python3 run_recognize.py --model buffalo_l --rtsp "rtsp://192.168.1.100:554/stream"### 2. Setup

| Vấn đề | Giải pháp |

|--------|-----------|The InsightFace model will be automatically downloaded on first run.

| FPS thấp | Sửa `det_size: [320, 320]` trong config |

| Không nhận dạng | Giảm threshold: 0.25 hoặc chụp thêm ảnh |# Model antelopev2

| Camera lỗi | Thử `--camera 1` |

python3 run_recognize.py --model antelopev2### 3. Directory Structure

---

```Create a directory for storing face images:

## 💡 Tips

```bash

- Chụp 20-30 ảnh/người

- Nhiều góc độ: thẳng, trái, phải, lên, xuống## 📂 Cấu trúcmkdir -p face_db

- Nhiều ánh sáng: sáng, tối, vừa

- Khoảng cách: 1-3m```



---```



**⚡ Xem CHEAT_SHEET.md để có tất cả lệnh!**Face_Recognition/## 🎯 Usage


├── capture_face.py              # 📸 Chụp ảnh TỰ ĐỘNG + Extract

├── run_recognize.py             # 🚀 Recognition (simple)### 1. Enroll Faces

├── run_recognize_detailed.py    # 📊 Recognition (detailed)```bash

├── extract_face_embeddings.py   # 🔄 Extract (thủ công nếu cần)python face_capture.py

├── config/default.json          # ⚙️ Cấu hình```

├── face_db/                     # 📁 Ảnh người đăng kýEnter the person's name when prompted. The system will capture multiple images of the face.

└── db_embedding/                # 💾 Embeddings database

```### 2. Process Face Images

```bash

## 🎯 Workflow hoàn chỉnhpython face_extract.py

```

```bashCreates the database from face images in the `face_db` directory.

# Bước 1: Thêm người (TỰ ĐỘNG)

python3 capture_face.py --name NguyenVanA### 3. Run Recognition

```bash

# Bước 2: Chạy recognition (NGAY LẬP TỨC)python recognize.py

python3 run_recognize.py --model buffalo_l```

```Arguments:

- `--camera 0` - Select camera (default: 0)

**Chỉ 2 lệnh là xong!** 🎉- `--threshold 0.4` - Recognition threshold

- `--gpu 0` - GPU device ID

## ⚙️ Models

### 4. RTSP Camera

| Model | Accuracy | Speed | Use Case |```bash

|-------|----------|-------|----------|python recognize.py --rtsp "rtsp://your-camera-url"

| **buffalo_l** | Very High | ~35-40 FPS | High accuracy apps |```

| **antelopev2** | Very High | ~40-45 FPS | Balance speed/accuracy |

### 5. Check Database

## 🔧 Parameters quan trọng```bash

python check_embedding.py

Sửa trong `config/default.json`:```



```json### 6. Optimize Models

{```bash

  "recognition": {# Optimize recognition model

    "threshold": 0.3,    // Thấp = dễ nhận dạng, Cao = chính xác hơnpython optimize_onnx.py --model w600k_mbf

    "gpu_id": 0          // 0 = GPU, -1 = CPU only

  }# Optimize detection model

}python optimize_onnx.py --model det_500m

``````



## 📝 Tips### 7. Run Recognition with Optimized Models

```bash

### Chụp ảnh tốtpython recognize_optimized.py --rec-model optimized_models/w600k_mbf_optimized.onnx

- 📸 Nhiều góc độ: thẳng, trái, phải, lên, xuống```

- 💡 Nhiều ánh sáng: sáng, tối, vừa

- 😊 Nhiều biểu cảm: bình thường, cười, nghiêm túc### 8. Benchmark Performance

- 🎯 Nhiều khoảng cách: 1m, 2m, 3m```bash

python benchmark_compare.py --original --optimized optimized_models/w600k_mbf_optimized.onnx --iterations 20 --warmup 5

### Troubleshooting```

- **FPS thấp**: Giảm `det_size` trong config: `[320, 320]`

- **Không nhận dạng**: Giảm threshold: `0.25` hoặc `0.2`## 🛠️ Tips

- **Camera lỗi**: Thử `--camera 1` hoặc `--camera 2`

### Recognition Threshold

## 📚 Documentation- `0.3` - More sensitive (may cause false positives)

- `0.4` - Recommended default

- **[USER_GUIDE.md](USER_GUIDE.md)** - Hướng dẫn chi tiết cho user- `0.6` - Stricter recognition (reduces false matches)

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Developer guide

- **config/default.json** - Tất cả cấu hình### Improving Accuracy

- Collect 15-20 face samples per person

## 🎨 Features- Include different lighting conditions

- Vary face angles slightly

- ✅ **Modular Architecture** - Dễ maintain & extend- Use good quality cameras

- ✅ **Auto Capture** - Tự động chụp + extract + update DB

- ✅ **Dual Model Support** - Buffalo-L & Antelopev2## 📊 Performance

- ✅ **RTSP Support** - IP cameras, CCTV- **CPU**: 5-15 FPS

- ✅ **GPU Accelerated** - CUDA support- **GPU**: 20-30 FPS

- ✅ **Config-Driven** - Không hardcode- **Optimized GPU**: 

- ✅ **Production Ready** - Error handling, logging  - Original: ~600 FPS (inference only)

  - ONNX Optimized: ~609 FPS (inference only)

## 🚦 Controls  - Full TensorRT (if available): Potentially higher performance



- **'q'** - Thoát chương trình## ⚠️ Notes

- Good lighting improves accuracy

## 🔥 One-Liner Setup- The InsightFace model downloads automatically on first run

- For best results, update the database regularly

```bash- Optimized models require onnxruntime-gpu

# Từ đầu đến cuối - chỉ 3 lệnh!- TensorRT acceleration requires additional setup and compatible hardware

source .face/bin/activate

python3 capture_face.py --name TenBan## 🚀 Optimization Notes

python3 run_recognize.py --model buffalo_l

```### ONNX Runtime Optimization

The system supports running with optimized ONNX models that improve inference speed:

**Done!** Hệ thống đã chạy và nhận dạng bạn! 🎉- **Graph optimization**: Speeds up model by fusing operations and removing redundancies

- **GPU acceleration**: Uses CUDA for faster execution

---- **Provider options**: Configures execution parameters for optimal performance



💡 **Tip**: Dùng chế độ TỰ ĐỘNG (`capture_face.py`) để setup nhanh nhất!### TensorRT Support

For maximum performance with TensorRT:
- Ensure TensorRT libraries are installed (`libnvinfer.so.10`)
- Check CUDA compatibility with your GPU
- Use the `--providers` flag to specify TensorRT providers:
  ```bash
  python recognize_optimized.py --providers TensorRT CUDA
  ```

### Troubleshooting
If you encounter provider errors:
```bash
python recognize_optimized.py --providers CUDA CPUExecutionProvider
```
Hiển thị thông tin chi tiết về database embeddings.

### 6. Sửa chữa database
```bash
python3 face_database_fix_duplicates.py
```
Kiểm tra và sửa chữa các vấn đề trong database như tên trùng lặp hoặc embedding có chất lượng kém.

## ⚙️ Cấu hình

Các thông số cấu hình có thể được điều chỉnh trong các file:

### Cấu trúc thư mục
- `FACE_DB_DIR`: Thư mục chứa ảnh khuôn mặt (mặc định: "face_db")
- `DB_PATH`: Đường dẫn đến file database embeddings (mặc định: "embeddings_db.h5")

### Tham số nhận diện
- `THRESHOLD`: Ngưỡng nhận diện khuôn mặt (mặc định: 0.5)
- `GPU_ID`: ID của GPU (mặc định: 0, -1 cho CPU)

## 🔍 Mô hình được hỗ trợ

Hệ thống sử dụng mô hình ArcFace từ InsightFace, cụ thể là mô hình "buffalo_s" với các đặc điểm:
- Face Detection: SCRFD (SCR Face Detector)
- Face Recognition: ArcFace với backbone ResNet
- Độ chính xác cao với chi phí tính toán vừa phải

## 📊 Hiệu năng

Hiệu năng của hệ thống phụ thuộc vào phần cứng:
- GPU: 20-30 FPS (NVIDIA GTX 1060 trở lên)
- CPU: 5-10 FPS (Intel i5 8th gen trở lên)

## 🤝 Đóng góp

Mọi đóng góp đều được hoan nghênh! Vui lòng tạo issue hoặc pull request.

## � Giấy phép

Dự án này được phân phối dưới giấy phép MIT. Xem file LICENSE để biết thêm chi tiết.

### Buffalo Series (InsightFace):
- `buffalo_l` - Độ chính xác cao nhất
- `buffalo_m` - Cân bằng tốc độ/chính xác  
- `buffalo_s` - Nhanh nhất (default)

### Thay đổi model:
```python
app = FaceAnalysis(name='buffalo_l')  # Chọn model
```

## 📊 Performance Tips

### 1. Tối ưu GPU:
```python
GPU_ID = 0              # Sử dụng GPU
ctx_id = 0              # CUDA context
det_size = (640, 640)   # Detection size
```

### 2. Tối ưu threshold:
- `THRESHOLD = 0.3-0.4`: Nhạy, có thể false positive
- `THRESHOLD = 0.5-0.6`: Cân bằng
- `THRESHOLD = 0.7+`: Chặt chẽ, ít false positive

### 3. Cải thiện chất lượng:
- Đảm bảo ánh sáng tốt khi ghi danh
- Thu thập nhiều góc độ khác nhau
- Tăng `NUM_SAMPLES` lên 30-50 mẫu

## 🐛 Troubleshooting

### 1. Lỗi OpenCV GUI:
```bash
sudo apt-get install python3-opencv
# hoặc
pip install opencv-python-headless
```

### 2. Lỗi ONNX Runtime:
```bash
# Cho CPU:
pip install onnxruntime
# Cho GPU (cần CUDA):
pip install onnxruntime-gpu
```

### 3. Lỗi InsightFace:
```bash
pip install insightface --no-deps
pip install onnx protobuf
```

### 4. Không nhận diện được:
- Kiểm tra `THRESHOLD` (thử giảm xuống 0.3)
- Chạy `check_embedding.py` để kiểm tra database
- Đảm bảo có ghi danh trước khi nhận diện

## 📈 Advanced Features

### 1. Multiple Models:
Có thể sử dụng nhiều model cùng lúc để tăng độ chính xác.

### 2. Database Management:
```python
# Xóa person khỏi database
# Cập nhật embeddings
# Backup/restore database
```

### 3. API Integration:
Có thể tích hợp vào web API hoặc mobile app.

## 🤝 Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- GitHub Issues: [Create an issue](https://github.com/BaekMinhDuc/Face_Recognition/issues)
- Email: your.email@example.com

## 🙏 Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface) for the amazing face recognition models
- [OpenCV](https://opencv.org/) for computer vision utilities
- [scikit-learn](https://scikit-learn.org/) for machine learning tools