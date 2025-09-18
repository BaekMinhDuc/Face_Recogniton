#!/usr/bin/env python3
import os
import time
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

DB_PATH = "embeddings_db.npz"
THRESHOLD = 0.4
GPU_ID = 0

# ==== RTSP SOURCE ====
# Ví dụ: rtsp://user:pass@192.168.1.64:554/Streaming/Channels/101
RTSP_URL = "rtsp://admin:HLTHKD@192.168.0.122:554/ch1/main"   # <-- ĐỔI CHO PHÙ HỢP
WINDOW_TITLE = "ArcFace"

# Reconnect config
RECONNECT_WAIT_SEC = 2.0
MAX_RECONNECT_TRIES = 0     # 0 = retry vô hạn

# Low-latency flags (tùy hệ thống có hiệu lực hay không)
CAP_PROPS = {
    cv2.CAP_PROP_BUFFERSIZE: 1,  # giảm backlog
}

def load_db(db_path: str = DB_PATH):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DB not found: {db_path}. Run enrollment first.")
    data = np.load(db_path, allow_pickle=True)
    names = list(data["names"])

    embs_arr = data["embeddings"]
    if isinstance(embs_arr, np.ndarray) and embs_arr.dtype == object:
        embs = np.vstack([np.asarray(e, dtype=np.float32).reshape(1, -1) for e in embs_arr])
    else:
        embs = np.asarray(embs_arr, dtype=np.float32)
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)

    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    print(f"[INFO] Loaded DB: {len(names)} person(s), emb shape: {embs.shape}")
    return names, embs

def init_arcface():
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=GPU_ID, det_size=(640, 640))
    return app

def match_face(emb, db_embs, db_names, thr=THRESHOLD):
    emb = emb.astype(np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    sims = cosine_similarity([emb], db_embs)[0]
    idx = int(np.argmax(sims))
    best = float(sims[idx])
    return (db_names[idx], best) if best >= thr else ("Unknown", best)

def open_rtsp(url: str):
    """Mở RTSP với cấu hình low-latency + retry"""
    tries = 0
    while True:
        cap = cv2.VideoCapture(url)
        # set props
        for k, v in CAP_PROPS.items():
            try:
                cap.set(k, v)
            except Exception:
                pass

        if cap.isOpened():
            return cap
        cap.release()

        tries += 1
        if MAX_RECONNECT_TRIES and tries >= MAX_RECONNECT_TRIES:
            raise RuntimeError(f"Cannot open RTSP after {tries} tries.")
        print(f"[WARN] Cannot open RTSP. Retrying in {RECONNECT_WAIT_SEC}s...")
        time.sleep(RECONNECT_WAIT_SEC)

def grab_latest_frame(cap, flush_reads=0):
    """
    Đọc một frame. Nếu flush_reads>0, đọc bỏ bớt các khung cũ để giảm trễ.
    Với nhiều camera/driver, cv2 buffer nhỏ + đọc liên tục đã đủ.
    """
    ok, frame = cap.read()
    if not ok:
        return False, None
    for _ in range(flush_reads):
        cap.read()
    return True, frame

if __name__ == "__main__":
    names, embs = load_db()
    app = init_arcface()

    cap = open_rtsp(RTSP_URL)
    print(f"[INFO] Running realtime on RTSP | Threshold={THRESHOLD} | GPU={GPU_ID}")
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

    # FPS smoothing
    prev_time = time.time()
    smooth_fps = 0.0

    while True:
        ok, frame = grab_latest_frame(cap, flush_reads=0)
        if not ok:
            # thử reconnect
            print("[WARN] RTSP read failed. Reconnecting...")
            cap.release()
            cap = open_rtsp(RTSP_URL)
            continue

        # ==== FPS ====
        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            inst_fps = 1.0 / dt
            smooth_fps = smooth_fps * 0.9 + inst_fps * 0.1

        # Face detect + ArcFace
        faces = app.get(frame)
        for f in faces:
            x1, y1, x2, y2 = map(int, f.bbox)
            name, score = match_face(f.embedding, embs, names, thr=THRESHOLD)
            color = (0, 200, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{name} ({score:.2f})", (x1, max(25, y1-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Overlay FPS
        cv2.putText(frame, f"FPS: {smooth_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        cv2.imshow(WINDOW_TITLE, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] Quit requested."); break

    cap.release()
    cv2.destroyAllWindows()
