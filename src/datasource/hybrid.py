import uuid
from typing import override

from src.datasource.base import DataSource
from qdrant_client import models
from qdrant_client.models import PointStruct

from src.embedder.embedder import Embedder
from src.embedder.sparse import SparseEmbedder


class HybridDatasource(DataSource):

    def __init__(self, sparse: SparseEmbedder, dense: Embedder):
        super().__init__()
        self.sparse = sparse
        self.dense = dense

    @override
    def create_collection(self, collection_name: str):
        self.client.create_collection(
            collection_name,
            vectors_config={
                "dense" : models.VectorParams(
                            size=self.dense.embedding_size,
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
        sparse_embedding = (self.sparse.embed(code))[0].as_object()
        dense_embedding = self.dense.embed(code)

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
    def upsert_chunks(self, collection_name:str, chunks: list[str]):
        sparse_embeddings = [self.sparse.embed(chunk) for chunk in chunks]
        sparse_embeddings = [embedding.as_object() for embedding in sparse_embeddings]

        dense_embeddings = [self.dense.embed(chunk) for chunk in chunks]
        dense_embeddings = [embedding.as_object() for embedding in dense_embeddings]

        self.client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "sparse": models.SparseVector(**sparse),
                        "dense": dense,
                    },
                    payload={self.model_key: chunk}
                ) for sparse, dense, chunk in zip(sparse_embeddings, dense_embeddings, chunks)
            ]
        )

    @override
    def search_functions(self, collection_name: str, query: str) -> list[str]:
        sparse_embedding = (self.sparse.embed(query))[0].as_object()
        dense_embedding = self.dense.embed(query)

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