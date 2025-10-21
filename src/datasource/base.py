from abc import ABC, abstractmethod

from src.embedder.embedder import Embedder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
from qdrant_client import AsyncQdrantClient
from src.parser.parser import DocParser

class DataSource(ABC):

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.reranker_name = "zeroentropy/zerank-1-small"
        self.url = os.getenv("URL")
        self.api_key = os.getenv("API_KEY")
        self.model_key = "code"
        self.client = AsyncQdrantClient(url=self.url, api_key=self.api_key, timeout=60)

    @abstractmethod
    async def create_collection(self, collection_name: str): ...

    async def load_doc_to_db(self, collection_name: str, filename: str) -> None:
        if not await self.client.collection_exists(collection_name):
            await self.create_collection(collection_name)

        parser = DocParser()

        content = parser.parse_to_string(filename)
        await self.upsert_chunk(collection_name, content)

    @abstractmethod
    async def upsert_chunk(self, collection_name: str, code: str): ...

    @abstractmethod
    async def upsert_chunks(self, collection_name: str, chunks: list[str]): ...

    @abstractmethod
    async def search_functions(self, collection_name: str, query: str) -> list[str]: ...

    def rerank_with_zerank(self, query, points) -> list[str]:
        tokenizer = AutoTokenizer.from_pretrained(self.reranker_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForSequenceClassification.from_pretrained(self.reranker_name)
        model.eval()
        pairs = [(query, point) for point in points]
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            scores = model(**inputs).logits.squeeze(-1)
        reranked = sorted(zip(points, scores.tolist()), key=lambda x: x[1], reverse=True)
        return [points for points, score in reranked]