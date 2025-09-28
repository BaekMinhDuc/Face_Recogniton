# 🚀 Face Recognition Project

## 📋 Project Overview
Hệ thống nhận diện khuôn mặt sử dụng ArcFace và InsightFace với các tính năng:
- Ghi danh khuôn mặt từ webcam
- Nhận diện real-time
- Hỗ trợ RTSP stream
- Giao diện trực quan

## 📁 Project Structure
```
Face_Recognition/
├── data_capture.py      # Ghi danh khuôn mặt từ webcam
├── recognize.py         # Nhận diện real-time từ webcam
├── recognize_rtsp.py    # Nhận diện từ RTSP stream
├── data_video.py        # Xử lý video file
├── check_embedding.py   # Kiểm tra database embeddings
├── requirements.txt     # Dependencies
├── README.md           # Hướng dẫn này
├── .gitignore          # Git ignore file
├── face_db/            # Thư mục chứa ảnh ghi danh
└── embeddings_db.npz   # Database embeddings
```

## 🔧 Installation

### 1. Clone repository
```bash
git clone https://github.com/BaekMinhDuc/Face_Recognition.git
cd Face_Recognition
```

### 2. Create virtual environment (recommended)
```bash
python3 -m venv .face
source .face/bin/activate  # Linux/Mac
# hoặc
.face\Scripts\activate     # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download InsightFace models
Models sẽ được tự động download lần đầu chạy, hoặc tải thủ công:
```bash
# Models sẽ được lưu trong ~/.insightface/
```

## 🎯 Usage

### 1. Ghi danh khuôn mặt mới
```bash
python3 data_capture.py
```
- Nhập tên người cần ghi danh
- Chương trình sẽ chụp 20 mẫu tự động
- Ảnh được lưu trong thư mục `face_db/`

### 2. Nhận diện real-time
```bash
python3 recognize.py
```
- Sử dụng webcam để nhận diện
- Hiển thị tên và confidence score
- Nhấn 'q' để thoát

### 3. Nhận diện từ RTSP stream
```bash
python3 recognize_rtsp.py
```
- Sửa RTSP URL trong code nếu cần
- Hỗ trợ IP camera

### 4. Kiểm tra database
```bash
python3 check_embedding.py
```
- Hiển thị thông tin chi tiết về database
- Kiểm tra chất lượng embeddings
- Thống kê tổng quan

## ⚙️ Configuration

### Trong `recognize.py`:
```python
THRESHOLD = 0.4        # Ngưỡng nhận diện (0.0-1.0)
GPU_ID = 0             # GPU ID (-1 cho CPU)
SRC = 0                # Camera source
```

### Trong `data_capture.py`:
```python
NUM_SAMPLES = 20       # Số mẫu thu thập
CAPTURE_INTERVAL = 1   # Khoảng cách giữa các mẫu (giây)
```

## 🔍 Models Supported

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