import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self):
        # model_name = str("Qwen/Qwen3-Embedding-0.6B"),
        self.embedding_size=1024
        self.model = SentenceTransformer(str("Qwen/Qwen3-Embedding-0.6B"))

    async def embed(self, query: str) -> list[float] | np.ndarray:
        return self.model.encode(query).tolist()
