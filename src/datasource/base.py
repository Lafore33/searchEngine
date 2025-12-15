from abc import ABC, abstractmethod
from qdrant_client import QdrantClient


class DataSource(ABC):

    def __init__(self) -> None:
        self.model_key = "corpus"
        self.client = QdrantClient(":memory:")

    @abstractmethod
    def create_collection(self, collection_name: str): ...

    @abstractmethod
    def upsert_chunk(self, collection_name: str, code: str): ...

    @abstractmethod
    def search_functions(self, collection_name: str, query: str) -> list[str]: ...
