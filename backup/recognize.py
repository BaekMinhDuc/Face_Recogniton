import os
import time
import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

DB_PATH = "embeddings_db.npz"
THRESHOLD = 0.4
GPU_ID = 0
SRC = 0
TITLE = "ArcFace"
MODEL_NAME = "buffalo_s"
ALLOWED_MODULES = ['detection', 'recognition']
DET_SIZE = (640, 640)
COLOR_KNOWN = (0, 200, 0)
COLOR_UNKNOWN = (0, 0, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_FPS = (0, 255, 0)
COLOR_INFO = (255, 255, 0)

# Statistics tracking
total_detect_time = 0
total_recog_time = 0
total_process_time = 0
frame_processed = 0

def load_db():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"DB not found: {DB_PATH}. Run enrollment first.")
    data = np.load(DB_PATH, allow_pickle=True)
    names = list(data["names"])
    embs = list(data["embeddings"])
    avg_embs = list(data["avg_embeddings"])
    return names, embs, avg_embs

def init_arcface():
    print("[INFO] Loading face recognition model...")
    model_start = time.time()
    
    # Check available providers
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
    
    # Load model with specified modules
    app = FaceAnalysis(name=MODEL_NAME, allowed_modules=ALLOWED_MODULES, providers=providers)
    app.prepare(ctx_id=GPU_ID, det_size=DET_SIZE)
    
    model_load_time = time.time() - model_start
    print(f"[INFO] Model loaded in {model_load_time:.2f}s")
    
    return app

def get_faces(app, frame):
    """Detect faces in frame and measure time"""
    detect_start = time.time()
    faces = app.get(frame)
    detect_time = (time.time() - detect_start) * 1000
    return faces, detect_time

def match_face(emb, db_embs, db_avg_embs, db_names, thr=THRESHOLD):
    """Match face embedding with database and measure time"""
    recog_start = time.time()
    
    emb = np.asarray(emb, dtype=np.float32).reshape(1, -1) / (np.linalg.norm(emb) + 1e-9)

    best_score = -1
    best_name = "Unknown"

    for name, avg_emb in zip(db_names, db_avg_embs):
        avg_emb = np.asarray(avg_emb, dtype=np.float32).reshape(1, -1)
        score = cosine_similarity(emb, avg_emb)[0][0]
        if score > best_score:
            best_score = score
            best_name = name if score >= thr else "Unknown"

    recog_time = (time.time() - recog_start) * 1000
    return best_name, best_score, recog_time

if __name__ == "__main__":
    names, embs, avg_embs = load_db()
    app = init_arcface()

    cap = cv2.VideoCapture(SRC)
    if not cap.isOpened():
        raise RuntimeError("Cannot open laptop camera (index 0).")

    print(f"[INFO] Running realtime | Threshold={THRESHOLD} | GPU={GPU_ID} | SRC={SRC}")
    print("[INFO] Press 'q' to quit\n")

    prev_time = time.time()
    smooth_fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Failed to read frame")
                break

            # Calculate FPS
            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                inst_fps = 1.0 / dt
                smooth_fps = (smooth_fps * 0.9) + (inst_fps * 0.1)

            # Process frame - measure time
            process_start = time.time()
            
            # Detect faces
            faces, detect_time = get_faces(app, frame)
            total_detect_time += detect_time
            
            # Recognize each face
            total_recog_time_frame = 0
            for f in faces:
                x1, y1, x2, y2 = map(int, f.bbox)
                name, score, recog_time = match_face(f.embedding, embs, avg_embs, names, thr=THRESHOLD)
                total_recog_time_frame += recog_time
                
                color = COLOR_KNOWN if name != "Unknown" else COLOR_UNKNOWN
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"{name} ({score:.2f})"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, max(25, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_TEXT, 2)
            
            total_recog_time += total_recog_time_frame
            
            process_time = (time.time() - process_start) * 1000
            total_process_time += process_time
            frame_processed += 1

            # Display FPS
            cv2.putText(frame, f"FPS: {smooth_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_FPS, 2)
            
            cv2.imshow(TITLE, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Quit requested.")
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Print statistics
        if frame_processed > 0:
            avg_detect_time = total_detect_time / frame_processed
            avg_recog_time = total_recog_time / frame_processed
            avg_process_time = total_process_time / frame_processed
            
    
            print(f"Average detection time: {avg_detect_time:.2f}ms")
            print(f"Average recognition time: {avg_recog_time:.2f}ms")
            print("=" * 60)
    cv2.destroyAllWindows()
