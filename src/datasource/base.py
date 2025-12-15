from abc import ABC, abstractmethod
import os
from qdrant_client import QdrantClient

class DataSource(ABC):

    def __init__(self) -> None:
        self.url = os.getenv("URL")
        self.api_key = os.getenv("API_KEY")
        self.model_key = "code"
        self.client = QdrantClient(url=self.url, api_key=self.api_key, timeout=60)

    @abstractmethod
    def create_collection(self, collection_name: str): ...

    @abstractmethod
    def upsert_chunk(self, collection_name: str, code: str): ...

    @abstractmethod
    def upsert_chunks(self, collection_name: str, chunks: list[str]): ...

    @abstractmethod
    def search_functions(self, collection_name: str, query: str) -> list[str]: ...
