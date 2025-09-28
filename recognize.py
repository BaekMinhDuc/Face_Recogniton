import os
import time
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

DB_PATH = "embeddings_db.npz"
THRESHOLD = 0.4
GPU_ID = 0
SRC = 0
TITLE = "ArcFace"

def load_db():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"DB not found: {DB_PATH}. Run enrollment first.")
    data = np.load(DB_PATH, allow_pickle=True)
    names = list(data["names"])
    embs = list(data["embeddings"])
    avg_embs = list(data["avg_embeddings"])
    return names, embs, avg_embs

def init_arcface():
    app = FaceAnalysis(name='buffalo_s')
    app.prepare(ctx_id=GPU_ID, det_size=(640, 640))
    return app

def match_face(emb, db_embs, db_avg_embs, db_names, thr=THRESHOLD):
    emb = np.asarray(emb, dtype=np.float32).reshape(1, -1) / (np.linalg.norm(emb) + 1e-9)

    best_score = -1
    best_name = "Unknown"

    for name, avg_emb in zip(db_names, db_avg_embs):
        avg_emb = np.asarray(avg_emb, dtype=np.float32).reshape(1, -1)  # Ensure avg_emb is a 2D NumPy array
        score = cosine_similarity(emb, avg_emb)[0][0]
        if score > best_score:
            best_score = score
            best_name = name if score >= thr else "Unknown"

    return best_name, best_score

if __name__ == "__main__":
    names, embs, avg_embs = load_db()
    app = init_arcface()

    cap = cv2.VideoCapture(SRC)
    if not cap.isOpened():
        raise RuntimeError("Cannot open laptop camera (index 0).")

    print(f"[INFO] Running realtime | Threshold={THRESHOLD} | GPU={GPU_ID} | SRC={SRC}")

    prev_time = time.time()
    smooth_fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Failed to read frame"); break

        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            inst_fps = 1.0 / dt
            smooth_fps = (smooth_fps * 0.9) + (inst_fps * 0.1)

        faces = app.get(frame)
        for f in faces:
            x1, y1, x2, y2 = map(int, f.bbox)
            name, score = match_face(f.embedding, embs, avg_embs, names, thr=THRESHOLD)
            color = (0, 200, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{name} ({score:.2f})", (x1, max(25, y1-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.putText(frame, f"FPS: {smooth_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow(TITLE, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Quit requested."); break

    cap.release()
    cv2.destroyAllWindows()
