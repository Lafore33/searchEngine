from src.datasource.datasource import DataSource
from src.embedder.embedder import Embedder
from src.parser.parser import DocParser
from datasets import Dataset
from src.metrics import recall_at_k, mrr_at_k, ndcg_at_k

# here I am assuming the file will contain the code as in the dataset, so I do not chunk it,
# however the chunk class is implemented and can be used in other cases
async def load_doc_to_db(db: DataSource, collection_name: str, filename: str) -> None:
    if not await db.client.collection_exists(collection_name):
        await db.create_collection(collection_name)

    parser = DocParser()

    content = parser.parse_to_string(filename)
    await db.upsert_chunk(collection_name, content)

async def load_code_to_db(db: DataSource, corpus: dict[str, list[str]]) -> None:
    if not await db.client.collection_exists("code"):
        await db.create_collection("code")

    for function in corpus["text"]:
        await db.upsert_chunk("code", function)

async def load_test_data(db: DataSource, corpus: list[str], force: bool) -> None:

    if (await db.client.collection_exists("code-test")) and force:
        await db.client.delete_collection("code-test")

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

def tune_model(model_name: str, default_dataset: Dataset, corpus_dataset: Dataset, queries_dataset: Dataset, epochs: int, show_plot: bool) -> None:
    model = Embedder(model_name)

    train_ids = [idx for idx, score in enumerate(default_dataset['train']["score"]) if score == 1]
    train_corpus = [str(corpus_dataset['text'][idx]) for idx in train_ids]
    train_queries = [str(queries_dataset['text'][idx]) for idx in train_ids]

    model.finetune(train_corpus, train_queries, epochs=epochs, show_plot=show_plot)

async def evaluate_model(model: Embedder, test_queries: list[str], test_corpus: list[str]) -> None:
    db = DataSource(model)
    predictions, gt = await test_search(db, test_queries, test_corpus)
    print(recall_at_k(gt, predictions))
    print(mrr_at_k(gt, predictions))
    print(ndcg_at_k(gt, predictions))



