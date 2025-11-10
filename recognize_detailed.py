import os
import time
import cv2
import numpy as np
import h5py
import argparse
import onnxruntime as ort
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

DB_PATH = "db_embedding/embed_antelopev2.h5"
THRESHOLD = 0.5
GPU_ID = 0
TITLE = "Face Recognition"
DISPLAY_SIZE = (640, 480)
MODEL_NAME = "antelopev2"
ALLOWED_MODULES = ['detection', 'recognition']
#ALLOWED_MODULES = None
DET_SIZE = (640, 640)
COLOR_KNOWN = (0, 200, 0)
COLOR_UNKNOWN = (0, 0, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_FPS = (0, 255, 0)
COLOR_INFO = (255, 255, 0)

class FaceRecognizer:
    def __init__(self, db_path=DB_PATH, threshold=THRESHOLD, gpu_id=GPU_ID):
        self.db_path = db_path
        self.threshold = threshold
        self.gpu_id = gpu_id
        self.names = []
        self.avg_embeddings = []
        
        print("[INFO] Loading face recognition model...")
        model_start = time.time()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
        self.face_app = FaceAnalysis(name=MODEL_NAME, allowed_modules=ALLOWED_MODULES, providers=providers)
        self.face_app.prepare(ctx_id=self.gpu_id, det_size=DET_SIZE)
        model_load_time = time.time() - model_start
        print(f"[INFO] Model loaded in {model_load_time:.2f}s")
        
        self.load_database()
        
        self.total_detect_time = 0
        self.total_recog_time = 0
        self.total_process_time = 0
        self.frame_processed = 0

    def load_database(self):
        if not os.path.exists(self.db_path):
            print(f"[WARNING] Database not found: {self.db_path}")
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
            print(f"[INFO] Loaded {len(self.names)} people from database")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load database: {e}")
            return False

    def get_faces(self, frame):
        detect_start = time.time()
        faces = self.face_app.get(frame)
        detect_time = (time.time() - detect_start) * 1000
        return faces, detect_time

    def identify_face(self, embedding):
        recog_start = time.time()
        embedding = np.asarray(embedding, dtype=np.float32)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-9)
        embedding = embedding.reshape(1, -1)
        best_score = -1
        best_name = "Unknown"
        valid_avg_embeddings = [e for e in self.avg_embeddings if e is not None]
        if not valid_avg_embeddings:
            recog_time = (time.time() - recog_start) * 1000
            return best_name, best_score, recog_time
        similarities = cosine_similarity(embedding, valid_avg_embeddings)[0]
        if len(similarities) > 0:
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]
            if best_score >= self.threshold:
                valid_names = [name for i, name in enumerate(self.names) if self.avg_embeddings[i] is not None]
                if best_idx < len(valid_names):
                    best_name = valid_names[best_idx]
        recog_time = (time.time() - recog_start) * 1000
        return best_name, best_score, recog_time

    def process_frame(self, frame):
        process_start = time.time()
        
        faces, detect_time = self.get_faces(frame)
        self.total_detect_time += detect_time
        
        total_recog_time = 0
        has_unknown = False
        
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            name, score, recog_time = self.identify_face(face.embedding)
            total_recog_time += recog_time
            
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
        
        self.total_recog_time += total_recog_time
        
        process_time = (time.time() - process_start) * 1000
        self.total_process_time += process_time
        self.frame_processed += 1
        
        return frame, faces, has_unknown, detect_time, total_recog_time, process_time

def run_recognition(source=0, threshold=THRESHOLD, gpu_id=GPU_ID, rtsp_url=None, db_path=DB_PATH):
    recognizer = FaceRecognizer(db_path=db_path, threshold=threshold, gpu_id=gpu_id)
    
    if rtsp_url:
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"[INFO] Using RTSP stream: {rtsp_url}")
    else:
        cap = cv2.VideoCapture(source)
        print(f"[INFO] Using camera: {source}")
    
    if not cap.isOpened():
        print("[ERROR] Cannot open camera/stream")
        return
    
    frame_count = 0
    fps = 0.0
    fps_update_time = time.time()
    
    print("[INFO] Starting face recognition...")
    print("[INFO] Press 'q' to quit\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if rtsp_url:
                print("[WARNING] Lost connection, reconnecting...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue
            else:
                break
        
        frame_count += 1
        current_time = time.time()
        
        if current_time - fps_update_time >= 1.0:
            fps = frame_count / (current_time - fps_update_time)
            frame_count = 0
            fps_update_time = current_time
        
        processed_frame, faces, has_unknown, detect_time, recog_time, process_time = recognizer.process_frame(frame)
        
        if DISPLAY_SIZE:
            processed_frame = cv2.resize(processed_frame, DISPLAY_SIZE)
        
        cv2.putText(processed_frame, f"FPS: {fps:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_FPS, 2)
        
        cv2.imshow(TITLE, processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if recognizer.frame_processed > 0:
        avg_detect_time = recognizer.total_detect_time / recognizer.frame_processed
        avg_recog_time = recognizer.total_recog_time / recognizer.frame_processed
        avg_process_time = recognizer.total_process_time / recognizer.frame_processed
        
        print(f"[INFO] Average detection time: {avg_detect_time:.2f}ms")
        print(f"[INFO] Average recognition time: {avg_recog_time:.2f}ms")
        print(f"[INFO] Average total processing time: {avg_process_time:.2f}ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Recognition System with Detailed Timing")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="Recognition threshold")
    parser.add_argument("--gpu", type=int, default=GPU_ID, help="GPU ID")
    parser.add_argument("--rtsp", type=str, default=None, help="RTSP URL")
    parser.add_argument("--db", type=str, default=DB_PATH, help="Database path")
    
    args = parser.parse_args()
    
    run_recognition(
        source=args.camera,
        threshold=args.threshold,
        gpu_id=args.gpu,
        rtsp_url=args.rtsp,
        db_path=args.db
    )
