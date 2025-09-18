import numpy as np

data = np.load("embeddings_db.npz", allow_pickle=True)
print("Names:", data["names"])
print("Embeddings shape:", np.array(data["embeddings"]).shape)
