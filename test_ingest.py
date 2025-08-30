from src.ingest import DataIngestor
from src.embedder import Embedder
from src.retriever import Retriever
from src.config import Config  # <-- Add this

config = Config()  

data = DataIngestor(config).ingest_all()

embedder = Embedder(config)
embedder.build_embeddings(data)

retriever = Retriever(config)
results = retriever.search("Does the login support MFA?", top_k=2)

for text, score, *_ in results:
    print(f"Score: {score:.4f} | Snippet: {text[:100]}...")