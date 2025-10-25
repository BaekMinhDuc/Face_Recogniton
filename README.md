# 🚀 Face Recognition System

## 📋 Introduction

A face recognition system using InsightFace with the following features:
- Face enrollment from webcam or images
- Real-time recognition from webcam or RTSP stream
- Detection of unknown faces
- Embedding database with quality checking
- Simple command-line interface
- **NEW:** Performance optimization with ONNX Runtime

## 📁 Project Structure

```
Face_Recognition/
├── face_capture.py          # Capture and enroll faces
├── face_extract.py          # Extract embeddings from images
├── recognize.py             # Main recognition program
├── recognize_optimized.py   # Recognition with optimized models
├── face_recognition.py      # Enhanced recognition system
├── check_embedding.py       # Check database quality
├── optimize_onnx.py         # ONNX model optimization tool
├── benchmark_compare.py     # Performance benchmark tool
├── README.md                # This guide
├── face_db/                 # Directory for face images
├── optimized_models/        # Optimized ONNX models
└── embeddings_db.h5         # Face database file
```

## 🔧 Installation

### 1. Requirements
```bash
pip install opencv-python numpy insightface h5py scikit-learn onnxruntime-gpu
```

For TensorRT acceleration (optional):
```bash
pip install nvidia-tensorrt
```

### 2. Setup
The InsightFace model will be automatically downloaded on first run.

### 3. Directory Structure
Create a directory for storing face images:
```bash
mkdir -p face_db
```

## 🎯 Usage

### 1. Enroll Faces
```bash
python face_capture.py
```
Enter the person's name when prompted. The system will capture multiple images of the face.

### 2. Process Face Images
```bash
python face_extract.py
```
Creates the database from face images in the `face_db` directory.

### 3. Run Recognition
```bash
python recognize.py
```
Arguments:
- `--camera 0` - Select camera (default: 0)
- `--threshold 0.4` - Recognition threshold
- `--gpu 0` - GPU device ID

### 4. RTSP Camera
```bash
python recognize.py --rtsp "rtsp://your-camera-url"
```

### 5. Check Database
```bash
python check_embedding.py
```

### 6. Optimize Models
```bash
# Optimize recognition model
python optimize_onnx.py --model w600k_mbf

# Optimize detection model
python optimize_onnx.py --model det_500m
```

### 7. Run Recognition with Optimized Models
```bash
python recognize_optimized.py --rec-model optimized_models/w600k_mbf_optimized.onnx
```

### 8. Benchmark Performance
```bash
python benchmark_compare.py --original --optimized optimized_models/w600k_mbf_optimized.onnx --iterations 20 --warmup 5
```

## 🛠️ Tips

### Recognition Threshold
- `0.3` - More sensitive (may cause false positives)
- `0.4` - Recommended default
- `0.6` - Stricter recognition (reduces false matches)

### Improving Accuracy
- Collect 15-20 face samples per person
- Include different lighting conditions
- Vary face angles slightly
- Use good quality cameras

## 📊 Performance
- **CPU**: 5-15 FPS
- **GPU**: 20-30 FPS
- **Optimized GPU**: 
  - Original: ~600 FPS (inference only)
  - ONNX Optimized: ~609 FPS (inference only)
  - Full TensorRT (if available): Potentially higher performance

## ⚠️ Notes
- Good lighting improves accuracy
- The InsightFace model downloads automatically on first run
- For best results, update the database regularly
- Optimized models require onnxruntime-gpu
- TensorRT acceleration requires additional setup and compatible hardware

## 🚀 Optimization Notes

### ONNX Runtime Optimization
The system supports running with optimized ONNX models that improve inference speed:
- **Graph optimization**: Speeds up model by fusing operations and removing redundancies
- **GPU acceleration**: Uses CUDA for faster execution
- **Provider options**: Configures execution parameters for optimal performance

### TensorRT Support
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