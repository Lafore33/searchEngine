import os
import numpy as np
from torch import nn
from pathlib import Path
from datasets import Dataset
from dotenv import load_dotenv
from transformers import Trainer
from sentence_transformers import SentenceTransformer
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]
TUNED_MODEL_PATH = ROOT_DIR / os.getenv('TUNED_PATH')

BATCH_SIZE = 16


class DenseEmbedder:
    def __init__(self, model_name: str, embedding_size: int, load_tuned=False):
        self.embedding_size = embedding_size
        self.model_name = model_name
        self.is_tuned = load_tuned
        TUNED_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        self.model = (
            SentenceTransformer(str(TUNED_MODEL_PATH))
            if load_tuned and any(TUNED_MODEL_PATH.iterdir())
            else SentenceTransformer(model_name)
        )

    def embed(self, query: str) -> list[float] | np.ndarray:
        return self.model.encode(query).tolist()

    def train(self, loss: nn.Module, train_corpus: list[str], train_query: list[str],
              epochs: int) -> Trainer:

        train_examples = {"query": train_query, "code": train_corpus}
        train_dataset = Dataset.from_dict(train_examples)

        args = SentenceTransformerTrainingArguments(
            output_dir=str(TUNED_MODEL_PATH),
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
        self.model.save(str(TUNED_MODEL_PATH))

        return trainer