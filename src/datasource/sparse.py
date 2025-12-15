import uuid
from typing import override

from qdrant_client.http.models import SparseVector

from src.datasource.base import DataSource
from qdrant_client import models
from qdrant_client.models import PointStruct
from src.embedder.sparse import SparseEmbedder


class SparseDatasource(DataSource):

    def __init__(self, sparse: SparseEmbedder):
        super().__init__()
        self.sparse = sparse

    @override
    def create_collection(self, collection_name: str):
        self.client.create_collection(
            collection_name,
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                ),
            },
        )

    @override
    def upsert_chunk(self, collection_name:str, code: str):
        embedding = (self.sparse.embed(code))[0].as_object()
        self.client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "sparse": models.SparseVector(**embedding),
                    },
                    payload={self.model_key: code}
                )
            ]
        )

    @override
    async def upsert_chunks(self, collection_name:str, chunks: list[str]):
        embeddings = [self.sparse.embed(chunk) for chunk in chunks]
        embeddings = [embedding.as_object() for embedding in embeddings]
        self.client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "sparse": models.SparseVector(**embedding),
                    },
                    payload={self.model_key: chunk}
                ) for embedding, chunk in zip(embeddings, chunks)
            ]
        )

    @override
    def search_functions(self, collection_name: str, query: str) -> list[str]:
        embedding = (self.sparse.embed(query))[0].as_object()
        vectors = self.client.query_points(
            collection_name=collection_name,
            query=SparseVector(**embedding),
            using="sparse",
        )

        return [point.payload[self.model_key] for point in vectors.points]