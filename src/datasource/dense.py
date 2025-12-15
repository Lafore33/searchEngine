import uuid
from typing import override
from qdrant_client import models
from src.embedder.dense import Embedder
from src.datasource.base import DataSource
from qdrant_client.models import PointStruct


class DenseDatasource(DataSource):

    def __init__(self, embedder: Embedder):
        super().__init__()
        self.embedder = embedder

    @override
    def create_collection(self, collection_name: str):
        self.client.create_collection(
            collection_name,
            vectors_config=models.VectorParams(
                size=self.embedder.embedding_size,
                distance=models.Distance.COSINE,
            )
        )

    @override
    def upsert_chunk(self, collection_name:str, code: str):
        self.client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=self.embedder.embed(code),
                    payload={self.model_key: code}
                )
            ]
        )

    @override
    def search_functions(self, collection_name: str, query: str) -> list[str]:
        vectors = self.client.query_points(
            collection_name=collection_name,
            query=self.embedder.embed(query)
        )

        return [point.payload[self.model_key] for point in vectors.points]