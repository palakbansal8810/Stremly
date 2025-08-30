from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
import joblib
import hashlib
from src.config import Config

class Embedder:
    def __init__(self, config: Config):
        self.config = config
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        config_str = f"{config.EMBEDDING_MODEL}_{config.CHUNK_SIZE}_{config.MAX_RETRIEVAL_RESULTS}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()
        self.index_path = f"embeddings/index_{config_hash}.faiss"
        self.data_path = f"embeddings/data_{config_hash}.pkl"

    def build_embeddings(self, data):
        all_chunks = data["policies"] + data["product_specs"] + data["code"] + data["screenshots"]
        self.chunks = all_chunks
        texts = [chunk["text"] for chunk in all_chunks]
        embeddings = self.model.encode(texts)

        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings))
        faiss.write_index(self.index, self.index_path)
        joblib.dump(self.chunks, self.data_path)
        print(f"✅ Embeddings stored at {self.index_path}")

    def load_embeddings(self):
        if os.path.exists(self.index_path) and os.path.exists(self.data_path):
            self.index = faiss.read_index(self.index_path)
            self.chunks = joblib.load(self.data_path)
            print("✅ Embeddings loaded successfully.")
        else:
            raise FileNotFoundError("Embeddings not found. Please run embedding build first.")