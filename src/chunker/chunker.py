from chonkie import SentenceChunker

# chunker is not needed here, as we do not want to chunk functions here, however I'll leave it here anyway
class DocChunker:

    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunker = SentenceChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split_to_chunks(self, content: str) -> list[str]:
        return [chunk.text for chunk in self.chunker(content)]