import os
import cv2
import numpy as np
import time
import h5py
from datetime import datetime
from insightface.app import FaceAnalysis

DB_PATH = "embeddings_db.h5"
NUM_SAMPLES = 20
CAPTURE_INTERVAL = 1  # Capture one frame per second
GPU_ID = 0
FACE_DB_DIR = "face_db"

# Ensure the face database directory exists
os.makedirs(FACE_DB_DIR, exist_ok=True)

class H5EmbeddingDB:
    """Class for managing embedding database using HDF5"""
    
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.names = []
        self.version = "1.0"
    
    def create_database(self):
        """Create a new database structure"""
        with h5py.File(self.db_path, 'w') as f:
            # Create main groups
            embeddings_group = f.create_group("embeddings")
            avg_embeddings_group = f.create_group("avg_embeddings")
            
            # Create metadata group
            metadata = f.create_group("metadata")
            metadata.attrs['created_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            metadata.attrs['updated_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            metadata.attrs['version'] = self.version
            metadata.attrs['description'] = "Face Recognition Embedding Database using HDF5"
            
            # Create dataset for names
            f.create_dataset("names", data=np.array([], dtype=h5py.special_dtype(vlen=str)))
            
            print(f"[INFO] Created new H5 database at {self.db_path}")
    
    def load(self):
        """Load database from file or create new if it doesn't exist"""
        if not os.path.exists(self.db_path):
            print(f"[INFO] Database {self.db_path} doesn't exist, creating new...")
            self.create_database()
            return [], []
        
        try:
            with h5py.File(self.db_path, 'r') as f:
                # Read name list
                if 'names' in f:
                    names = list(f['names'][()])
                else:
                    names = []
                
                # Read average embeddings
                avg_embeddings = []
                if 'avg_embeddings' in f:
                    for name in names:
                        if name in f['avg_embeddings']:
                            avg_embeddings.append(f['avg_embeddings'][name][()])
                
                print(f"[INFO] Loaded database from {self.db_path}")
                print(f"[INFO] People count: {len(names)}")
                return names, avg_embeddings
        except Exception as e:
            print(f"[ERROR] Error reading database: {e}")
            print("[INFO] Creating new database...")
            self.create_database()
            return [], []
    
    def save(self, names, embeddings_dict, avg_embeddings_dict):
        """Save or update database
        
        Args:
            names: List of person names
            embeddings_dict: Dict with person name as key, list of embeddings as value
            avg_embeddings_dict: Dict with person name as key, average embedding as value
        """
        # Create new file if it doesn't exist
        if not os.path.exists(self.db_path):
            self.create_database()
        
        # Check and handle duplicate names
        unique_names = []
        for name in names:
            if name not in unique_names:
                unique_names.append(name)
            else:
                print(f"[WARN] Duplicate name detected: '{name}'. Saving only once.")
        
        if len(unique_names) != len(names):
            print(f"[INFO] Removed {len(names) - len(unique_names)} duplicate names.")
            names = unique_names
        
        try:
            with h5py.File(self.db_path, 'a') as f:
                # Update name list
                if 'names' in f:
                    # Read current name list for checking
                    current_names = list(f['names'][()])
                    current_names = [name.decode('utf-8') if isinstance(name, bytes) else name for name in current_names]
                    
                    # Check if new name list has duplicates with current list
                    for name in names:
                        if current_names.count(name) > 1:
                            print(f"[WARN] Name '{name}' appears multiple times in database. Run face_database_fix_duplicates.py to fix.")
                    
                    # Delete old name list
                    del f['names']
                
                # Create new name dataset
                f.create_dataset("names", data=np.array(names, dtype=h5py.special_dtype(vlen=str)))
                
                # Update metadata
                if 'metadata' in f:
                    f['metadata'].attrs['updated_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f['metadata'].attrs['num_people'] = len(names)
                
                # Update embeddings and avg_embeddings
                for name in names:
                    # Save all sample embeddings
                    if name in embeddings_dict:
                        emb_path = f"embeddings/{name}"
                        if emb_path in f:
                            del f[emb_path]
                        
                        embeddings = embeddings_dict[name]
                        f.create_dataset(emb_path, data=np.array(embeddings))
                        f[emb_path].attrs['num_samples'] = len(embeddings)
                        f[emb_path].attrs['updated_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Save average embedding
                    if name in avg_embeddings_dict:
                        avg_path = f"avg_embeddings/{name}"
                        if avg_path in f:
                            del f[avg_path]
                        
                        avg_emb = avg_embeddings_dict[name]
                        f.create_dataset(avg_path, data=avg_emb)
                        norm = float(np.linalg.norm(avg_emb))
                        f[avg_path].attrs['norm'] = norm
                        f[avg_path].attrs['quality'] = "good" if 0.9 <= norm <= 1.1 else "suspicious"
                
                print(f"[INFO] Saved database with {len(names)} people")
        except Exception as e:
            print(f"[ERROR] Error saving database: {e}")
    
    def get_embeddings(self, name):
        """Get embeddings for a person"""
        try:
            with h5py.File(self.db_path, 'r') as f:
                if f"embeddings/{name}" in f:
                    return f[f"embeddings/{name}"][()]
                else:
                    return None
        except Exception as e:
            print(f"[ERROR] Error getting embeddings: {e}")
            return None
    
    def get_avg_embedding(self, name):
        """Get average embedding for a person"""
        try:
            with h5py.File(self.db_path, 'r') as f:
                if f"avg_embeddings/{name}" in f:
                    return f[f"avg_embeddings/{name}"][()]
                else:
                    return None
        except Exception as e:
            print(f"[ERROR] Error getting average embedding: {e}")
            return None
    
    def remove_person(self, name):
        """Remove a person from database"""
        try:
            with h5py.File(self.db_path, 'a') as f:
                # Remove embeddings
                if f"embeddings/{name}" in f:
                    del f[f"embeddings/{name}"]
                
                # Remove avg_embeddings
                if f"avg_embeddings/{name}" in f:
                    del f[f"avg_embeddings/{name}"]
                
                # Update name list
                if 'names' in f:
                    names = list(f['names'][()])
                    if name in names:
                        names.remove(name)
                        del f['names']
                        f.create_dataset("names", data=np.array(names, dtype=h5py.special_dtype(vlen=str)))
                        
                        # Update metadata
                        if 'metadata' in f:
                            f['metadata'].attrs['updated_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            f['metadata'].attrs['num_people'] = len(names)
                
                print(f"[INFO] Removed '{name}' from database")
                return True
        except Exception as e:
            print(f"[ERROR] Error removing person: {e}")
            return False

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

        # Display the number of captured samples
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

    # Return all collected embeddings
    return collected

if __name__ == "__main__":
    print("=== Enhanced ArcFace Enrollment with H5 ===")
    name = input("Enter person's name (label): ").strip()
    while not name:
        name = input("Name cannot be empty. Enter person's name: ").strip()

    # Initialize and load database
    db = H5EmbeddingDB()
    names, avg_embs = db.load()
    
    # Convert bytes to string if needed
    names = [n.decode('utf-8') if isinstance(n, bytes) else n for n in names]
    
    # Convert average embeddings list to dict
    avg_embeddings_dict = {}
    for i, n in enumerate(names):
        if i < len(avg_embs):
            avg_embeddings_dict[n] = avg_embs[i]
    
    # Dict to store all sample embeddings
    embeddings_dict = {}
    
    # Create embeddings dict from existing database
    for n in names:
        embs = db.get_embeddings(n)
        if embs is not None:
            embeddings_dict[n] = embs
    
    if name in names:
        print(f"[WARN] '{name}' already exists. Options:")
        print("1. Overwrite existing data")
        print("2. Add more samples to existing data")
        print("3. Cancel enrollment")
        choice = input("Enter your choice (1/2/3): ").strip()
        
        if choice == '3':
            print("[INFO] Enrollment cancelled.")
            exit(0)
        elif choice == '2':
            # Collect and add new samples
            new_embeddings = enroll(name)
            
            # Add to existing embeddings
            if name in embeddings_dict:
                current_embs = embeddings_dict[name]
                if isinstance(current_embs, np.ndarray) and current_embs.ndim == 2:
                    embeddings_dict[name] = np.vstack([current_embs, new_embeddings])
                else:
                    embeddings_dict[name] = np.array(new_embeddings)
            else:
                embeddings_dict[name] = np.array(new_embeddings)
                
            # Recalculate average embedding
            embs = embeddings_dict[name]
            avg_embeddings_dict[name] = np.mean([e / (np.linalg.norm(e) + 1e-9) for e in embs], axis=0)
            
            print(f"[INFO] Added {len(new_embeddings)} samples to '{name}'.")
        else:
            # Overwrite existing data
            embeddings = enroll(name)
            embeddings_dict[name] = np.array(embeddings)
            
            # Calculate new average embedding
            avg_embeddings_dict[name] = np.mean([e / (np.linalg.norm(e) + 1e-9) for e in embeddings], axis=0)
            
            print(f"[INFO] Overwritten data for '{name}' with {len(embeddings)} samples.")
    else:
        # Add new person
        embeddings = enroll(name)
        embeddings_dict[name] = np.array(embeddings)
        
        # Calculate average embedding
        avg_embeddings_dict[name] = np.mean([e / (np.linalg.norm(e) + 1e-9) for e in embeddings], axis=0)
        
        # Add to name list
        names.append(name)
        
        print(f"[INFO] Added new person '{name}' with {len(embeddings)} samples.")

    # Save database
    db.save(names, embeddings_dict, avg_embeddings_dict)
    print("[DONE] Enrollment complete.")
    print("[INFO] To check database, run 'python face_database_check.py'")