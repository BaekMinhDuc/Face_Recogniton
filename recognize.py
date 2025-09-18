import os
import time
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

DB_PATH = "embeddings_db.npz"
THRESHOLD = 0.5
GPU_ID = 0
SRC = 0
TITLE = "ArcFace"

def load_db(db_path: str = DB_PATH):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DB not found: {db_path}. Run enrollment first.")
    data = np.load(db_path, allow_pickle=True)
    names = list(data["names"])

    embs_arr = data["embeddings"]
    avg_embs = data["avg_embeddings"]

    embs = [np.asarray(e, dtype=np.float32) for e in embs_arr]
    avg_embs = [np.asarray(a, dtype=np.float32) for a in avg_embs]

    print(f"[INFO] Loaded DB: {len(names)} person(s), embeddings per person: {[len(e) for e in embs]}")
    return names, embs, avg_embs

def init_arcface():
    app = FaceAnalysis(name='buffalo_s')  # Sử dụng model buffalo_s để thống nhất
    app.prepare(ctx_id=GPU_ID, det_size=(640, 640))
    return app

def match_face(emb, db_embs, db_avg_embs, db_names, thr=THRESHOLD):
    emb = np.asarray(emb, dtype=np.float32).reshape(1, -1)  # Đảm bảo embedding là 2D
    emb = emb / (np.linalg.norm(emb) + 1e-9)

    best_score = -1
    second_best_score = -1
    best_name = "Unknown"

    for name, avg_emb, person_embs in zip(db_names, db_avg_embs, db_embs):
        avg_sim = cosine_similarity(emb, avg_emb.reshape(1, -1))[0][0]
        if avg_sim > best_score:
            second_best_score = best_score
            best_score = avg_sim
            best_name = name if avg_sim >= thr else "Unknown"

        if best_name == "Unknown":
            sims = cosine_similarity(emb, person_embs)[0]
            max_sim = float(np.max(sims))
            if max_sim > best_score:
                second_best_score = best_score
                best_score = max_sim
                best_name = name if max_sim >= thr else "Unknown"

    # Kiểm tra độ chênh lệch giữa điểm cao nhất và cao thứ hai
    if best_score - second_best_score < 0.1:
        best_name = "Unknown"

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

        # ==== FPS ====
        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            inst_fps = 1.0 / dt
            # EMA smoothing: 90% old + 10% new
            smooth_fps = (smooth_fps * 0.9) + (inst_fps * 0.1)

        faces = app.get(frame)
        for f in faces:
            x1, y1, x2, y2 = map(int, f.bbox)
            name, score = match_face(f.embedding, embs, avg_embs, names, thr=THRESHOLD)
            color = (0, 200, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{name} ({score:.2f})", (x1, max(25, y1-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Vẽ FPS luôn luôn (mượt)
        cv2.putText(frame, f"FPS: {smooth_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow(TITLE, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Quit requested."); break

    cap.release()
    cv2.destroyAllWindows()
