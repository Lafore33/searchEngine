import uuid
from typing import override
from qdrant_client import models
from src.embedder.dense import DenseEmbedder
from src.datasource.base import DataSource
from qdrant_client.models import PointStruct
from src.embedder.sparse import SparseEmbedder


class HybridDatasource(DataSource):

    def __init__(self, sparse_embedder: SparseEmbedder, dense_embedder: DenseEmbedder):
        super().__init__()
        self.sparse_embedder = sparse_embedder
        self.dense_embedder = dense_embedder

    @override
    def create_collection(self, collection_name: str):
        self.client.create_collection(
            collection_name,
            vectors_config={
                "dense" : models.VectorParams(
                            size=self.dense_embedder.embedding_size,
                            distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                ),
            },
        )

    @override
    def upsert_chunk(self, collection_name:str, code: str):
        sparse_embedding = self.sparse_embedder.embed(code)[0].as_object()
        dense_embedding = self.dense_embedder.embed(code)

        self.client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "sparse": models.SparseVector(**sparse_embedding),
                        "dense": dense_embedding,
                    },
                    payload={self.model_key: code}
                )
            ]
        )

    @override
    def search_functions(self, collection_name: str, query: str) -> list[str]:
        sparse_embedding = self.sparse_embedder.embed(query)[0].as_object()
        dense_embedding = self.dense_embedder.embed(query)

        vectors = self.client.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(
                    query=models.SparseVector(**sparse_embedding),
                    using="sparse",
                ),
                models.Prefetch(
                    query=dense_embedding,
                    using="dense",
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
        )

        return [point.payload[self.model_key] for point in vectors.points]