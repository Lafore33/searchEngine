from fastembed import SparseTextEmbedding, SparseEmbedding


class SparseEmbedder:
    def __init__(self, model_name: str):
        self.model = SparseTextEmbedding(model_name=model_name)

    async def embed(self, query: str) -> list[SparseEmbedding]:
        return list(self.model.embed(query))