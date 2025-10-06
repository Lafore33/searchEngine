import os
import uuid
from src.parser.parser import DocParser
from src.embedder.embedder import Embedder
from qdrant_client import AsyncQdrantClient
from qdrant_client import models
from qdrant_client.models import PointStruct
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class DataSource:

    def __init__(self, embedder: Embedder):
        self.url = os.getenv("URL")
        self.api_key = os.getenv("API_KEY")
        self.model_key = "code"
        self.reranker_name = "zeroentropy/zerank-1-small"
        self.client = AsyncQdrantClient(url=self.url, api_key=self.api_key, timeout=60)
        self.embedder = embedder

    async def create_collection(self, collection_name: str):
        await self.client.create_collection(
            collection_name,
            vectors_config=models.VectorParams(
                size=self.embedder.embedding_size,
                distance=models.Distance.COSINE,
            )
        )

    async def load_doc_to_db(self, collection_name: str, filename: str) -> None:
        if not await self.client.collection_exists(collection_name):
            await self.create_collection(collection_name)

        parser = DocParser()

        content = parser.parse_to_string(filename)
        await self.upsert_chunk(collection_name, content)

    async def upsert_chunk(self, collection_name:str, code: str):
        await self.client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=await self.embedder.embed(code),
                    payload={self.model_key: code}
                )
            ]
        )

    async def upsert_chunks(self, collection_name:str, chunks: list[str]):
        await self.client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=await self.embedder.embed(chunk),
                    payload={self.model_key: chunk}
                ) for chunk in chunks
            ]
        )


    async def search_functions(self, collection_name: str, query: str) -> list[str]:
        vectors = await self.client.query_points(
            collection_name=collection_name,
            query=await self.embedder.embed(query)
        )

        return [point.payload[self.model_key] for point in vectors.points]

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