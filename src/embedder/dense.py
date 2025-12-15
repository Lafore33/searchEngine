import numpy as np
from torch import nn
from datasets import Dataset
from transformers import Trainer
from sentence_transformers import SentenceTransformer
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

TUNED_MODEL_PATH = "./tuned/finetuned"
BATCH_SIZE = 16


class DenseEmbedder:
    def __init__(self, model_name: str, embedding_size: int, load_tuned=False):
        self.embedding_size = embedding_size
        self.model_name = model_name
        self.is_tuned = load_tuned
        self.model = (
            SentenceTransformer(model_name)
            if not load_tuned
            else SentenceTransformer(TUNED_MODEL_PATH)
        )

    def embed(self, query: str) -> list[float] | np.ndarray:
        return self.model.encode(query).tolist()

    def train(self, loss: nn.Module, train_corpus: list[str], train_query: list[str],
              epochs: int) -> Trainer:

        train_examples = {"query": train_query, "code": train_corpus}
        train_dataset = Dataset.from_dict(train_examples)

        args = SentenceTransformerTrainingArguments(
            output_dir=TUNED_MODEL_PATH,
            num_train_epochs=epochs,
            per_device_train_batch_size=BATCH_SIZE,
        )

        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            loss=loss(self.model),
        )

        trainer.train()
        self.model.save(TUNED_MODEL_PATH)

        return trainer