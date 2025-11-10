import os
import cv2
import numpy as np
import h5py
import time
import onnxruntime as ort
from datetime import datetime
from insightface.app import FaceAnalysis

FACE_DB_DIR = "face_db"
DB_PATH = "db_embedding/embed_antelopev2.h5"
MODEL_NAME = "antelopev2"
ALLOWED_MODULES = ['detection', 'recognition']
DET_SIZE = (640, 640)
GPU_ID = 0

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

class EmbeddingExtractor:
    def __init__(self):
        print("[INFO] Loading face recognition model...")
        model_start = time.time()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
        self.face_app = FaceAnalysis(name=MODEL_NAME, allowed_modules=ALLOWED_MODULES, providers=providers)
        self.face_app.prepare(ctx_id=GPU_ID, det_size=DET_SIZE)
        model_load_time = time.time() - model_start
        print(f"[INFO] Model loaded in {model_load_time:.2f}s")
    
    def extract_from_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return None
        faces = self.face_app.get(image)
        if len(faces) == 0:
            return None
        largest_face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
        embedding = largest_face.embedding / (np.linalg.norm(largest_face.embedding) + 1e-9)
        return embedding
    
    def extract_from_folder(self, person_name, folder_path):
        embeddings = []
        valid_images = 0
        total_images = 0
        
        print(f"\n[INFO] Processing '{person_name}'...")
        
        for img_file in sorted(os.listdir(folder_path)):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                total_images += 1
                img_path = os.path.join(folder_path, img_file)
                embedding = self.extract_from_image(img_path)
                if embedding is not None:
                    embeddings.append(embedding)
                    valid_images += 1
                    print(f"  [{valid_images}/{total_images}] Extracted: {img_file}")
                else:
                    print(f"  [SKIP] No face found: {img_file}")
        
        if len(embeddings) == 0:
            print(f"[WARNING] No valid embeddings for '{person_name}'")
            return None
        
        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-9)
        
        print(f"[SUCCESS] '{person_name}': {valid_images}/{total_images} images processed")
        return np.array(embeddings), avg_embedding

def save_database(db_path, data):
    with h5py.File(db_path, 'w') as f:
        f.create_group("embeddings")
        f.create_group("avg_embeddings")
        
        metadata = f.create_group("metadata")
        metadata.attrs['created_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata.attrs['updated_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata.attrs['version'] = "1.0"
        metadata.attrs['model'] = MODEL_NAME
        metadata.attrs['num_people'] = len(data)
        
        names = list(data.keys())
        f.create_dataset("names", data=np.array(names, dtype=h5py.special_dtype(vlen=str)))
        
        for name, (embeddings, avg_embedding) in data.items():
            emb_path = f"embeddings/{name}"
            f.create_dataset(emb_path, data=embeddings)
            f[emb_path].attrs['num_samples'] = len(embeddings)
            
            avg_path = f"avg_embeddings/{name}"
            f.create_dataset(avg_path, data=avg_embedding)
            norm = float(np.linalg.norm(avg_embedding))
            f[avg_path].attrs['norm'] = norm
    
    print(f"\n[SAVED] Database saved to: {db_path}")
    print(f"[INFO] Total people: {len(data)}")

def main():
    print("=" * 60)
    print("Face Embedding Extraction System")
    print("=" * 60)
    
    if not os.path.exists(FACE_DB_DIR):
        print(f"[ERROR] Face database directory not found: {FACE_DB_DIR}")
        return
    
    extractor = EmbeddingExtractor()
    
    person_folders = [d for d in os.listdir(FACE_DB_DIR) 
                     if os.path.isdir(os.path.join(FACE_DB_DIR, d))]
    
    if len(person_folders) == 0:
        print(f"[ERROR] No person folders found in {FACE_DB_DIR}")
        return
    
    print(f"\n[INFO] Found {len(person_folders)} people in database")
    
    data = {}
    start_time = time.time()
    
    for person_name in person_folders:
        folder_path = os.path.join(FACE_DB_DIR, person_name)
        result = extractor.extract_from_folder(person_name, folder_path)
        if result is not None:
            embeddings, avg_embedding = result
            data[person_name] = (embeddings, avg_embedding)
    
    total_time = time.time() - start_time
    
    if len(data) > 0:
        save_database(DB_PATH, data)
        print(f"[INFO] Total processing time: {total_time:.2f}s")
    else:
        print("[ERROR] No valid data to save")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
