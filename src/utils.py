from torch import nn
from datasets import Dataset
from src.datasource.base import DataSource
from src.embedder.embedder import Embedder
from src.metrics import recall_at_k, mrr_at_k, ndcg_at_k

def load_code_to_db(db: DataSource, collection_name: str, corpus: dict[str, list[str]]) -> None:
    if not db.client.collection_exists(collection_name):
        db.create_collection(collection_name)

    for function in corpus["text"]:
        db.upsert_chunk(collection_name, function)

def load_test_data(db: DataSource, collection_name: str,
                         corpus: list[str], force: bool) -> None:

    if (db.client.collection_exists(collection_name)) and force:
        db.client.delete_collection(collection_name)

    if not db.client.collection_exists(collection_name):
        db.create_collection(collection_name)

    for function in corpus:
        db.upsert_chunk(collection_name, function)

def test_search(db: DataSource, collection_name: str,
                      queries: list[str], corpus: list[str]):

    predictions = []
    gt = []
    for query, function in zip(queries, corpus):
        result = db.search_functions(collection_name, query)
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

def evaluate_model(db: DataSource, collection_name: str,
                         test_queries: list[str], test_corpus: list[str]) -> None:
    predictions, gt = test_search(db, collection_name, test_queries, test_corpus)
    print(recall_at_k(gt, predictions))
    print(mrr_at_k(gt, predictions))
    print(ndcg_at_k(gt, predictions))



