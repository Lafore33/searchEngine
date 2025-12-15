from torch import nn
from typing import Any
from numpy import floating
from datasets import Dataset
from transformers import Trainer
from src.embedder.dense import Embedder
from src.datasource.base import DataSource
from src.metrics import recall_at_k, mrr_at_k, ndcg_at_k

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

def train(model: Embedder, loss: nn.Module,
          default_dataset: Dataset, corpus_dataset: Dataset,
          queries_dataset: Dataset, epochs: int) -> Trainer:

    train_ids = [idx for idx, score in enumerate(default_dataset["train"]["score"]) if score == 1]
    train_corpus = [str(corpus_dataset["text"][idx]) for idx in train_ids]
    train_queries = [str(queries_dataset["text"][idx]) for idx in train_ids]

    return model.train(loss, train_corpus, train_queries, epochs=epochs)

def evaluate_model(db: DataSource, collection_name: str,
                         test_queries: list[str], test_corpus: list[str]) -> tuple[floating[Any], floating[Any], floating[Any]]:

    predictions, gt = test_search(db, collection_name, test_queries, test_corpus)
    recall, mrr, ndcg = recall_at_k(gt, predictions), mrr_at_k(gt, predictions), ndcg_at_k(gt, predictions)

    return recall, mrr, ndcg


