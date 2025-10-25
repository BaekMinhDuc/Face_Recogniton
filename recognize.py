import os
import time
import cv2
import numpy as np
import h5py
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import argparse

DB_PATH = "embeddings_db.h5"
THRESHOLD = 0.4
GPU_ID = 0
TITLE = "Arcface"
DISPLAY_SIZE = (1580, 960)
COLOR_KNOWN = (0, 200, 0)
COLOR_UNKNOWN = (0, 0, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_FPS = (0, 255, 0)

class H5FaceRecognizer:
    def __init__(self, db_path=DB_PATH, threshold=THRESHOLD, gpu_id=GPU_ID):
        self.db_path = db_path
        self.threshold = threshold
        self.gpu_id = gpu_id
        import onnxruntime as ort
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
        self.face_app = FaceAnalysis(name='buffalo_s', providers=providers)
        self.face_app.prepare(ctx_id=self.gpu_id, det_size=(640, 640))
        self.names = []
        self.avg_embeddings = []
        self.load_database()

    def load_database(self):
        if not os.path.exists(self.db_path):
            return False
        try:
            with h5py.File(self.db_path, 'r') as f:
                if 'names' in f:
                    self.names = list(f['names'][()])
                    self.names = [name.decode('utf-8') if isinstance(name, bytes) else name for name in self.names]
                else:
                    return False
                self.avg_embeddings = []
                for name in self.names:
                    avg_path = f"avg_embeddings/{name}"
                    if avg_path in f:
                        self.avg_embeddings.append(f[avg_path][()])
                    else:
                        self.avg_embeddings.append(None)
            return True
        except Exception as e:
            return False

    def get_faces(self, frame):
        faces = self.face_app.get(frame)
        return faces

    def identify_face(self, embedding):
        embedding = np.asarray(embedding, dtype=np.float32)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-9)
        embedding = embedding.reshape(1, -1)
        best_score = -1
        best_name = "Unknown"
        valid_avg_embeddings = [e for e in self.avg_embeddings if e is not None]
        if not valid_avg_embeddings:
            return best_name, best_score
        similarities = cosine_similarity(embedding, valid_avg_embeddings)[0]
        if len(similarities) > 0:
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]
            if best_score >= self.threshold:
                valid_names = [name for i, name in enumerate(self.names) if self.avg_embeddings[i] is not None]
                if best_idx < len(valid_names):
                    best_name = valid_names[best_idx]
        return best_name, best_score

    def process_frame(self, frame):
        faces = self.get_faces(frame)
        has_unknown = False
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            name, score = self.identify_face(face.embedding)
            if name == "Unknown":
                has_unknown = True
                color = COLOR_UNKNOWN
            else:
                color = COLOR_KNOWN
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{name} ({score:.2f})"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2)
        if has_unknown:
            cv2.putText(frame, "UNKNOWN", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_UNKNOWN, 3)
        return frame, faces, has_unknown

def run_recognition(source=0, threshold=THRESHOLD, gpu_id=GPU_ID, rtsp_url=None):
    recognizer = H5FaceRecognizer(threshold=threshold, gpu_id=gpu_id)
    if rtsp_url:
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return
    frame_count = 0
    start_time = time.time()
    fps = 0
    fps_update_time = start_time
    import onnxruntime as ort
    gpu_available = 'CUDAExecutionProvider' in ort.get_available_providers()
    while True:
        ret, frame = cap.read()
        if not ret:
            if rtsp_url:
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue
            else:
                break
        frame_count += 1
        current_time = time.time()
        elapsed = current_time - start_time
        if current_time - fps_update_time >= 1.0:
            fps = frame_count / (current_time - fps_update_time)
            frame_count = 0
            fps_update_time = current_time
        processed_frame, faces, has_unknown = recognizer.process_frame(frame)
        if DISPLAY_SIZE:
            processed_frame = cv2.resize(processed_frame, DISPLAY_SIZE)
        cv2.putText(processed_frame, f"FPS: {fps:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_FPS, 2)
        cv2.imshow(TITLE, processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Recognition")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help=f"Recognition threshold (default: {THRESHOLD})")
    parser.add_argument("--gpu", type=int, default=GPU_ID, help=f"GPU ID (default: {GPU_ID})")
    parser.add_argument("--rtsp", type=str, default=None, help="RTSP URL (optional)")
    parser.add_argument("--db", type=str, default=DB_PATH, help=f"Database path (default: {DB_PATH})")
    args = parser.parse_args()
    DB_PATH = args.db
    run_recognition(source=args.camera, threshold=args.threshold, gpu_id=args.gpu, rtsp_url=args.rtsp)