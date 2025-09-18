import os
import cv2
import numpy as np
import time
from insightface.app import FaceAnalysis

DB_PATH = "embeddings_db.npz"
NUM_SAMPLES = 20
GPU_ID = 0  

def load_db():
    if os.path.exists(DB_PATH):
        data = np.load(DB_PATH, allow_pickle=True)
        return list(data["names"]), list(data["embeddings"])
    return [], []

def save_db(names, embs):
    avg_embs = []
    for e in embs:
        e = np.asarray(e, dtype=np.float32)  # Đảm bảo e là mảng numpy
        if e.ndim == 1:  # Nếu e là mảng 1D, chuyển thành 2D
            e = e.reshape(1, -1)
        normalized = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
        avg_embs.append(np.mean(normalized, axis=0))

    np.savez_compressed(DB_PATH,
                        names=np.array(names, dtype=object),
                        embeddings=np.array(embs, dtype=object),
                        avg_embeddings=np.array(avg_embs, dtype=object))
    print(f"[INFO] Database saved to {DB_PATH} (people: {len(names)})")

def init_arcface():
    app = FaceAnalysis(name='buffalo_s')  # Sử dụng model nhẹ hơn
    app.prepare(ctx_id=GPU_ID, det_size=(640, 640))
    return app

def enroll(person_name: str):
    app = init_arcface()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open laptop camera (index 0)")

    collected = []
    last_capture_time = 0  # Thời gian capture cuối cùng
    print(f"[INFO] Enrolling '{person_name}' from webcam (0). Capturing {NUM_SAMPLES} samples...")

    while len(collected) < NUM_SAMPLES:
        ret, frame = cap.read()
        if not ret:
            continue

        faces = app.get(frame)
        current_time = time.time()

        if faces and (current_time - last_capture_time) >= 0.5:  # Chờ ít nhất 0.5 giây giữa các lần capture
            f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
            embedding = f.embedding / (np.linalg.norm(f.embedding) + 1e-9)  # Chuẩn hóa embedding
            collected.append(embedding)
            last_capture_time = current_time

            x1, y1, x2, y2 = map(int, f.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"Samples: {len(collected)}/{NUM_SAMPLES}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        else:
            cv2.putText(frame, "No face detected or waiting...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

        cv2.imshow("Enrollment - ArcFace", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Early stop by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if not collected:
        raise RuntimeError("No valid face samples collected.")

    # Lưu tất cả các embedding thay vì chỉ trung bình
    embeddings = np.vstack(collected)
    return embeddings

if __name__ == "__main__":
    print("=== ArcFace Enrollment ===")
    name = input("Enter person's name (label): ").strip()
    while not name:
        name = input("Name cannot be empty. Enter person's name: ").strip()

    names, embs = load_db()
    if name in names:
        print(f"[WARN] '{name}' already exists and will be overwritten.")

    embeddings = enroll(name)

    if name in names:
        idx = names.index(name)
        embs[idx] = embeddings
        print(f"[INFO] Updated embeddings for '{name}'.")
    else:
        names.append(name)
        embs.append(embeddings)
        print(f"[INFO] Added new person '{name}'.")

    save_db(names, embs)
    print("[DONE] Enrollment complete.")
