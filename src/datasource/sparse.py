import uuid
from typing import override

from src.datasource.base import DataSource
from src.embedder.embedder import Embedder
from qdrant_client import models
from qdrant_client.models import PointStruct


class SparseDatasource(DataSource):

    def __init__(self, embedder: Embedder):
        super().__init__(embedder)

    @override
    async def create_collection(self, collection_name: str):
        await self.client.create_collection(
            collection_name,
            sparse_vectors_config={
                "text": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                ),
            },
        )

    @override
    async def upsert_chunk(self, collection_name:str, code: str):
        await self.client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "text": models.SparseVector(await self.embedder.embed(code)),
                    },
                    payload={self.model_key: code}
                )
            ]
        )

    @override
    async def upsert_chunks(self, collection_name:str, chunks: list[str]):
        await self.client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "text": models.SparseVector(await self.embedder.embed(chunk)),
                    },
                    payload={self.model_key: chunk}
                ) for chunk in chunks
            ]
        )

    @override
    async def search_functions(self, collection_name: str, query: str) -> list[str]:
        vectors = await self.client.query_points(
            collection_name=collection_name,
            query=await self.embedder.embed(query),
            using="text",
        )

        return [point.payload[self.model_key] for point in vectors.points]