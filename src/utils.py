from src.datasource.datasource import DataSource
from src.embedder.embedder import Embedder
from src.parser.parser import DocParser
from datasets import Dataset
from src.metrics import recall_at_k, mrr_at_k, ndcg_at_k
from torch import nn

# here I am assuming the file will contain the code as in the dataset, so I do not chunk it,
# however the chunk class is implemented and can be used in other cases

async def load_code_to_db(db: DataSource, collection_name: str, corpus: dict[str, list[str]]) -> None:
    if not await db.client.collection_exists(collection_name):
        await db.create_collection(collection_name)

    for function in corpus["text"]:
        await db.upsert_chunk(collection_name, function)

async def load_test_data(db: DataSource, collection_name: str,
                         corpus: list[str], force: bool) -> None:

    if (await db.client.collection_exists(collection_name)) and force:
        await db.client.delete_collection(collection_name)

    if not await db.client.collection_exists(collection_name):
        await db.create_collection(collection_name)

    for function in corpus:
        await db.upsert_chunk(collection_name, function)

async def test_search(db: DataSource, collection_name: str,
                      queries: list[str], corpus: list[str], rerank=False):

    predictions = []
    gt = []
    for query, function in zip(queries, corpus):
        result = await db.search_functions(collection_name, query)
        if rerank:
            result = db.rerank_with_zerank(query, result)
        predictions.append(result)
        gt.append(function)

    return predictions, gt

def tune_model(model: Embedder, loss: nn.Module,
               default_dataset: Dataset, corpus_dataset: Dataset,
               queries_dataset: Dataset, epochs: int, show_plot: bool) -> None:

    train_ids = [idx for idx, score in enumerate(default_dataset["train"]["score"]) if score == 1]
    train_corpus = [str(corpus_dataset["text"][idx]) for idx in train_ids]
    train_queries = [str(queries_dataset["text"][idx]) for idx in train_ids]

    model.finetune(loss, train_corpus, train_queries, epochs=epochs, show_plot=show_plot)

async def evaluate_model(model: Embedder, collection_name: str,
                         test_queries: list[str], test_corpus: list[str], rerank=False) -> None:
    db = DataSource(model)
    predictions, gt = await test_search(db, collection_name, test_queries, test_corpus, rerank)
    print(recall_at_k(gt, predictions))
    print(mrr_at_k(gt, predictions))
    print(ndcg_at_k(gt, predictions))



