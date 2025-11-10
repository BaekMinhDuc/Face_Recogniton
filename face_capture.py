import os
import cv2
import numpy as np
import time
import h5py
from datetime import datetime
from insightface.app import FaceAnalysis

DB_PATH = "db_embedding/embed_l.h5"
NUM_SAMPLES = 20
CAPTURE_INTERVAL = 1
GPU_ID = 0
FACE_DB_DIR = "face_db"
MODEL_NAME = "buffalo_l"
DET_SIZE = (640, 640)

os.makedirs(FACE_DB_DIR, exist_ok=True)

class H5EmbeddingDB:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.names = []
        self.version = "1.0"

    def create_database(self):
        with h5py.File(self.db_path, 'w') as f:
            f.create_group("embeddings")
            f.create_group("avg_embeddings")
            metadata = f.create_group("metadata")
            metadata.attrs['created_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            metadata.attrs['updated_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            metadata.attrs['version'] = self.version
            metadata.attrs['description'] = "Face Recognition Embedding Database using HDF5"
            f.create_dataset("names", data=np.array([], dtype=h5py.special_dtype(vlen=str)))

    def load(self):
        if not os.path.exists(self.db_path):
            self.create_database()
            return [], []
        try:
            with h5py.File(self.db_path, 'r') as f:
                names = list(f['names'][()]) if 'names' in f else []
                avg_embeddings = [f['avg_embeddings'][name][()] for name in names if name in f['avg_embeddings']]
                return names, avg_embeddings
        except Exception as e:
            print(f"[ERROR] Failed to load database: {e}")
            self.create_database()
            return [], []

    def save(self, names, embeddings_dict, avg_embeddings_dict):
        if not os.path.exists(self.db_path):
            self.create_database()
        unique_names = list(set(names))
        try:
            with h5py.File(self.db_path, 'a') as f:
                if 'names' in f:
                    del f['names']
                f.create_dataset("names", data=np.array(unique_names, dtype=h5py.special_dtype(vlen=str)))
                if 'metadata' in f:
                    f['metadata'].attrs['updated_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f['metadata'].attrs['num_people'] = len(unique_names)
                for name in unique_names:
                    if name in embeddings_dict:
                        emb_path = f"embeddings/{name}"
                        if emb_path in f:
                            del f[emb_path]
                        embeddings = embeddings_dict[name]
                        f.create_dataset(emb_path, data=np.array(embeddings))
                        f[emb_path].attrs['num_samples'] = len(embeddings)
                        f[emb_path].attrs['updated_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if name in avg_embeddings_dict:
                        avg_path = f"avg_embeddings/{name}"
                        if avg_path in f:
                            del f[avg_path]
                        avg_emb = avg_embeddings_dict[name]
                        f.create_dataset(avg_path, data=avg_emb)
                        norm = float(np.linalg.norm(avg_emb))
                        f[avg_path].attrs['norm'] = norm
                        f[avg_path].attrs['quality'] = "good" if 0.9 <= norm <= 1.1 else "suspicious"
            print(f"[SAVED] Database updated with {len(unique_names)} people")
        except Exception as e:
            print(f"[ERROR] Failed to save database: {e}")

    def get_embeddings(self, name):
        try:
            with h5py.File(self.db_path, 'r') as f:
                return f[f"embeddings/{name}"][()] if f"embeddings/{name}" in f else None
        except Exception as e:
            return None

    def get_avg_embedding(self, name):
        try:
            with h5py.File(self.db_path, 'r') as f:
                return f[f"avg_embeddings/{name}"][()] if f"avg_embeddings/{name}" in f else None
        except Exception as e:
            return None

    def check_person_exists(self, name):
        try:
            with h5py.File(self.db_path, 'r') as f:
                return f"avg_embeddings/{name}" in f
        except Exception as e:
            return False

    def delete_person(self, name):
        try:
            with h5py.File(self.db_path, 'a') as f:
                if f"embeddings/{name}" in f:
                    del f[f"embeddings/{name}"]
                if f"avg_embeddings/{name}" in f:
                    del f[f"avg_embeddings/{name}"]
        except Exception as e:
            print(f"[ERROR] Failed to delete {name}: {e}")

def init_arcface():
    import onnxruntime as ort
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
    allowed_modules = ['detection', 'recognition']
    app = FaceAnalysis(name=MODEL_NAME, allowed_modules=allowed_modules, providers=providers)
    app.prepare(ctx_id=GPU_ID, det_size=DET_SIZE)
    return app

def enroll(person_name):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open laptop camera")
    
    collected = []
    last_capture_time = 0
    person_dir = os.path.join(FACE_DB_DIR, person_name)
    os.makedirs(person_dir, exist_ok=True)
    app = init_arcface()
    
    print(f"\n[START] Collecting {NUM_SAMPLES} samples for '{person_name}'")
    print("[INFO] Make sure your face is clearly visible in the camera")
    print("[INFO] Press 'q' to quit early\n")
    
    while len(collected) < NUM_SAMPLES:
        ret, frame = cap.read()
        if not ret:
            continue
        
        current_time = time.time()
        if (current_time - last_capture_time) >= CAPTURE_INTERVAL:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join(person_dir, f"{timestamp}.jpg")
            cv2.imwrite(image_path, frame)
            
            faces = app.get(frame)
            if faces:
                f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
                embedding = f.embedding / (np.linalg.norm(f.embedding) + 1e-9)
                collected.append(embedding)
                last_capture_time = current_time
                print(f"[CAPTURE] Sample {len(collected)}/{NUM_SAMPLES} saved")
            else:
                print(f"[WARNING] No face detected in frame")
        
        cv2.rectangle(frame, (10, 10), (300, 50), (0, 255, 0), -1)
        cv2.putText(frame, f"Samples: {len(collected)}/{NUM_SAMPLES}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        cv2.imshow("Enrollment - Press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Enrollment cancelled")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if not collected:
        raise RuntimeError("No valid face samples collected")
    
    print(f"[SUCCESS] Collected {len(collected)} samples")
    return collected

if __name__ == "__main__":
    print("=" * 50)
    print("Face Embedding Enrollment System")
    print("=" * 50)
    
    name = input("\nEnter person's name: ").strip()
    while not name:
        name = input("Name cannot be empty. Enter person's name: ").strip()
    
    db = H5EmbeddingDB()
    names, avg_embs = db.load()
    names = [n.decode('utf-8') if isinstance(n, bytes) else n for n in names]
    
    avg_embeddings_dict = {n: avg_embs[i] for i, n in enumerate(names) if i < len(avg_embs)}
    embeddings_dict = {n: db.get_embeddings(n) for n in names}
    
    if name in names:
        print(f"\n[INFO] '{name}' already exists in database")
        print("[INFO] Old embedding will be deleted and replaced with new one\n")
        db.delete_person(name)
    
    try:
        embeddings = enroll(name)
    except KeyboardInterrupt:
        print("\n[INFO] Enrollment cancelled by user")
        exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        exit(1)
    
    embeddings_dict[name] = np.array(embeddings)
    avg_embeddings_dict[name] = np.mean([e / (np.linalg.norm(e) + 1e-9) for e in embeddings], axis=0)
    
    if name not in names:
        names.append(name)
    
    db.save(names, embeddings_dict, avg_embeddings_dict)
    print(f"\n[SUCCESS] '{name}' enrollment complete!")
    print(f"[INFO] Total people in database: {len(names)}")
    print("=" * 50)
