from src.datasource.datasource import DataSource
from src.embedder.embedder import Embedder
from src.parser.parser import DocParser
from src.chunker.chunker import DocChunker
from datasets import load_dataset, Dataset
from src.metrics import recall_at_k, mrr_at_k, ndcg_at_k
import asyncio
from dotenv import load_dotenv

chunk_size = 512
chunk_overlap = int(chunk_size * 0.2)

async def load_doc_to_db(db: DataSource, collection_name: str, filename: str) -> None:
    if not await db.client.collection_exists(collection_name):
        await db.create_collection(collection_name)

    parser = DocParser()
    chunker = DocChunker(chunk_size, chunk_overlap)

    content = parser.parse_to_string(filename)
    chunks = chunker.split_to_chunks(content)
    await db.upsert_chunks(collection_name, chunks)


async def load_code_to_db(db: DataSource, corpus: dict[str, list[str]]) -> None:
    if not await db.client.collection_exists("code"):
        await db.create_collection("code")

    for function in corpus["text"]:
        await db.upsert_chunk("code", function)

async def load_test_data(db: DataSource, corpus: list[str]) -> None:

    if not await db.client.collection_exists("code-test"):
        await db.create_collection("code-test")

    for function in corpus:
        await db.upsert_chunk("code-test", function)

async def test_search(db: DataSource, queries: list[str], corpus: list[str]):

    predictions = []
    gt = []
    for query, function in zip(queries, corpus):
        result = await db.search_functions("code-test", query)
        predictions.append(result)
        gt.append(function)

    return predictions, gt

async def main():
    load_dotenv()

    model = Embedder()
    db = DataSource(model)
    queries_dataset = load_dataset("CoIR-Retrieval/cosqa", "queries")["queries"]
    corpus_dataset = load_dataset("CoIR-Retrieval/cosqa", "corpus")["corpus"]
    # dataset = load_dataset("CoIR-Retrieval/cosqa", "default")
    test_corpus = [function for partition, function in zip(corpus_dataset['partition'], corpus_dataset['text']) if partition == "test"]
    test_queries = [query for partition, query in zip(queries_dataset['partition'], queries_dataset['text']) if partition == "test"]

    # await load_test_data(db, test_corpus)
    predictions, gt = await test_search(db, test_queries, test_corpus)
    print(recall_at_k(gt, predictions))
    print(mrr_at_k(gt, predictions))
    print(ndcg_at_k(gt, predictions))

if __name__ == "__main__":
    asyncio.run(main())



