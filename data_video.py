#!/usr/bin/env python3
import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

DB_PATH = "embeddings_db.npz"
GPU_ID = 0

# Tùy chỉnh nhanh
SAMPLE_STRIDE = 1        # =1 duyệt từng frame; tăng (2/3/5) nếu video rất dài để nhanh hơn
MIN_FACE_AREA = 0.02     # tỉ lệ diện tích bbox/ảnh tối thiểu để chấp nhận (lọc mặt quá nhỏ)
SHARPNESS_MIN = 20.0     # ngưỡng độ nét (Laplacian variance) tối thiểu để chấp nhận
QUALITY_WEIGHTED = True  # True: trung bình có trọng số theo quality; False: trung bình đều

def l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x) + 1e-9)

def load_db():
    if os.path.exists(DB_PATH):
        data = np.load(DB_PATH, allow_pickle=True)
        names = list(data["names"])
        embs_arr = data["embeddings"]
        if isinstance(embs_arr, np.ndarray) and embs_arr.dtype == object:
            embs = [np.asarray(e, dtype=np.float32).reshape(-1) for e in embs_arr]
        else:
            arr = np.asarray(embs_arr, dtype=np.float32)
            if arr.ndim == 1: arr = arr.reshape(1, -1)
            embs = [e for e in arr]
        return names, embs
    return [], []

def save_db(names, embs):
    embs_2d = np.vstack([np.asarray(e, dtype=np.float32).reshape(1, -1) for e in embs])
    np.savez_compressed(DB_PATH,
                        names=np.array(names, dtype=object),
                        embeddings=embs_2d)
    print(f"[INFO] DB saved to {DB_PATH} | people={len(names)} | emb shape={embs_2d.shape}")

def init_arcface():
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=GPU_ID, det_size=(640, 640))  # sẽ fallback CPU nếu không có CUDA EP
    return app

def laplacian_sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def face_center(face):
    x1, y1, x2, y2 = face.bbox
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

def compute_quality(frame, face) -> float:
    """Heuristic quality: det_score × area × (1 + sharp/1000)"""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = map(int, face.bbox)
    area = max(1, (x2 - x1) * (y2 - y1)) / float(W * H)
    det_score = float(getattr(face, "det_score", 0.5))
    sharp = laplacian_sharpness(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    return det_score * area * (1.0 + sharp / 1000.0), area, sharp

def pick_target_face(frame, faces, ref_center=None, tol=0.35):
    """Chọn đúng người cần bám: nếu đã có ref_center → lấy mặt gần tâm đó;
       nếu chưa có → chọn mặt chất lượng cao nhất trong frame."""
    if not faces:
        return None, None, None, None
    H, W = frame.shape[:2]
    norm = float(max(W, H))

    if ref_center is not None:
        # lấy mặt gần ref_center nhất
        faces = sorted(
            faces,
            key=lambda f: (
                ((( (f.bbox[0]+f.bbox[2])*0.5 - ref_center[0])**2 +
                   ((f.bbox[1]+f.bbox[3])*0.5 - ref_center[1])**2 )**0.5) / norm
            )
        )
        cand = faces[0]
        cx, cy = face_center(cand)
        dist_norm = (((cx - ref_center[0])**2 + (cy - ref_center[1])**2)**0.5) / norm
        if dist_norm > tol:
            return None, None, None, None
        q, area, sharp = compute_quality(frame, cand)
        return cand, q, area, sharp
    else:
        # chưa có ref → chọn theo quality cao nhất
        best, best_q, best_area, best_sharp = None, -1.0, None, None
        for f in faces:
            q, area, sharp = compute_quality(frame, f)
            if q > best_q:
                best, best_q, best_area, best_sharp = f, q, area, sharp
        return best, best_q, best_area, best_sharp

if __name__ == "__main__":
    print("=== Aggregate face embeddings from ALL frames in a video ===")
    name = input("Enter person's name (label): ").strip()
    while not name:
        name = input("Name cannot be empty. Enter person's name (label): ").strip()
    vpath = input("Enter video file path (e.g., ./phone_clip.mp4): ").strip()
    while not os.path.exists(vpath):
        vpath = input("File not found. Enter video file path again: ").strip()

    names, embs = load_db()
    if name in names:
        print(f"[WARN] '{name}' exists and will be overwritten with new aggregated embedding.")

    app = init_arcface()
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {vpath}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    print(f"[INFO] Scanning all frames: {total} (stride={SAMPLE_STRIDE})")

    ref_center = None
    collected, weights = [], []
    used_frames = 0
    accepted_frames = 0

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if SAMPLE_STRIDE > 1 and (idx % SAMPLE_STRIDE != 0):
            idx += 1
            continue

        faces = app.get(frame)
        face, q, area, sharp = pick_target_face(frame, faces, ref_center=ref_center)

        if face is not None:
            # thiết lập ref_center từ lần đầu gặp
            if ref_center is None:
                ref_center = face_center(face)

            # lọc chất lượng
            if area is not None and area < MIN_FACE_AREA:
                idx += 1; used_frames += 1; continue
            if sharp is not None and sharp < SHARPNESS_MIN:
                idx += 1; used_frames += 1; continue

            emb = l2norm(face.embedding.astype(np.float32))
            collected.append(emb)
            w = float(q) if (QUALITY_WEIGHTED and q is not None) else 1.0
            weights.append(w)
            accepted_frames += 1

        used_frames += 1
        idx += 1

    cap.release()

    if not collected:
        print("[WARN] No acceptable face found across the whole video.")
        exit(0)

    E = np.vstack(collected)     # [M,512]
    W = np.asarray(weights, dtype=np.float32).reshape(-1, 1)

    if QUALITY_WEIGHTED:
        # trung bình có trọng số
        final_emb = l2norm((E * W).sum(axis=0) / (W.sum() + 1e-9))
    else:
        # trung bình đều
        final_emb = l2norm(E.mean(axis=0))

    # lưu DB
    if name in names:
        idx = names.index(name)
        embs[idx] = final_emb
        print(f"[INFO] Updated embedding for '{name}'.")
    else:
        names.append(name)
        embs.append(final_emb)
        print(f"[INFO] Added new person '{name}'.")

    save_db(names, embs)
    print(f"[DONE] Aggregated {accepted_frames} embedding(s) from {used_frames} used frame(s) / {total} total.")
