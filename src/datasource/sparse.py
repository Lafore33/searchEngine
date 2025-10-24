import uuid
from typing import override

from qdrant_client.http.models import SparseVector

from src.datasource.base import DataSource
from qdrant_client import models
from qdrant_client.models import PointStruct
import asyncio
from src.embedder.sparse import SparseEmbedder


class SparseDatasource(DataSource):

    def __init__(self, embedder: SparseEmbedder):
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
        embedding = (await self.embedder.embed(code))[0].as_object()
        await self.client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "text": models.SparseVector(
                            indices=embedding["indices"],
                            values=embedding["values"]
                        ),
                    },
                    payload={self.model_key: code}
                )
            ]
        )

    @override
    async def upsert_chunks(self, collection_name:str, chunks: list[str]):
        embeddings = await asyncio.gather(*[self.embedder.embed(chunk) for chunk in chunks])
        embeddings = [embedding.as_object() for embedding in embeddings]
        await self.client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "text": models.SparseVector(
                            indices=embedding["indices"],
                            values=embedding["values"]
                        ),
                    },
                    payload={self.model_key: embedding}
                ) for embedding in embeddings
            ]
        )

    @override
    async def search_functions(self, collection_name: str, query: str) -> list[str]:
        embedding = (await self.embedder.embed(query))[0].as_object()
        vectors = await self.client.query_points(
            collection_name=collection_name,
            query=SparseVector(
                indices=embedding["indices"],
                values=embedding["values"]
            ),
            using="text",
        )

        return [point.payload[self.model_key] for point in vectors.points]