import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import joblib
import redis
import hashlib
from src.config import Config
import pickle
class Retriever:
    def __init__(self, config: Config):
        self.config = config
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        config_str = f"{config.EMBEDDING_MODEL}_{config.CHUNK_SIZE}_{config.MAX_RETRIEVAL_RESULTS}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()
        self.index = faiss.read_index(config.EMBEDDINGS_INDEX_PATH)
        self.chunks = joblib.load(config.EMBEDDINGS_DATA_PATH)
        self.redis = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT, db=config.REDIS_DB)

    import pickle
# ...existing code...

    def search(self, query, top_k=None):
        top_k = top_k or self.config.MAX_RETRIEVAL_RESULTS
        cache_key = f"retrieval:{hashlib.md5(query.encode()).hexdigest()}:{top_k}"
        cached = self.redis.get(cache_key)
        if cached:
            return pickle.loads(cached)  # <-- use pickle
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding), top_k)
        results = []
        for j, i in enumerate(indices[0]):
            if i >= 0:
                chunk = self.chunks[i]
                results.append((chunk["text"], float(distances[0][j]), chunk["doc_id"], chunk["chunk_id"]))
        self.redis.setex(cache_key, 3600, pickle.dumps(results))  # <-- use pickle
        return results