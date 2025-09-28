import os
import cv2
import numpy as np
import time
from datetime import datetime
from insightface.app import FaceAnalysis

DB_PATH = "embeddings_db.npz"
NUM_SAMPLES = 20
CAPTURE_INTERVAL = 1  # Capture one frame per second
GPU_ID = 0
FACE_DB_DIR = "face_db"

# Ensure the face database directory exists
os.makedirs(FACE_DB_DIR, exist_ok=True)

def load_db():
    if os.path.exists(DB_PATH):
        data = np.load(DB_PATH, allow_pickle=True)
        return list(data["names"]), list(data["embeddings"])
    return [], []

def save_db(names, embs):
    avg_embs = []
    for e in embs:
        if isinstance(e, dict):
            combined_emb = np.concatenate([np.asarray(v, dtype=np.float32) for v in e.values()])
            normalized = combined_emb / (np.linalg.norm(combined_emb) + 1e-9)
            avg_embs.append(normalized)
        else:
            e = np.asarray(e, dtype=np.float32)
            if e.ndim == 1:
                e = e.reshape(1, -1)
            normalized = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
            avg_embs.append(np.mean(normalized, axis=0))

    np.savez_compressed(DB_PATH,
                        names=np.array(names, dtype=object),
                        embeddings=np.array(embs, dtype=object),
                        avg_embeddings=np.array(avg_embs, dtype=object))
    print(f"[INFO] Database saved to {DB_PATH} (people: {len(names)})")

def init_arcface():
    app = FaceAnalysis(name='buffalo_s')
    app.prepare(ctx_id=GPU_ID, det_size=(640, 640))
    return app

def enroll(person_name: str):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open laptop camera (index 0)")

    collected = []
    last_capture_time = 0
    print(f"[INFO] Enrolling '{person_name}' from webcam (0). Capturing {NUM_SAMPLES} samples...")

    person_dir = os.path.join(FACE_DB_DIR, person_name)
    os.makedirs(person_dir, exist_ok=True)

    app = init_arcface()

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

        # Always display the number of captured samples
        cv2.putText(frame, f"Samples: {len(collected)}/{NUM_SAMPLES}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

        cv2.imshow("Enrollment", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Early stop by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if not collected:
        raise RuntimeError("No valid face samples collected.")

    avg_embedding = np.mean(collected, axis=0)
    return avg_embedding

if __name__ == "__main__":
    print("=== ArcFace Enrollment ===")
    name = input("Enter person's name (label): ").strip()
    while not name:
        name = input("Name cannot be empty. Enter person's name: ").strip()

    names, embs = load_db()
    if name in names:
        print(f"[WARN] '{name}' already exists and will be overwritten.")

    avg_embedding = enroll(name)

    if name in names:
        idx = names.index(name)
        embs[idx] = avg_embedding
        print(f"[INFO] Updated embeddings for '{name}'.")
    else:
        names.append(name)
        embs.append(avg_embedding)
        print(f"[INFO] Added new person '{name}'.")

    save_db(names, embs)
    print("[DONE] Enrollment complete.")
