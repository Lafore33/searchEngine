import numpy as np
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from transformers import TrainerCallback
from torch import nn
import matplotlib.pyplot as plt
from typing import Callable

LossFunction = Callable[[SentenceTransformer], "SentenceTransformerLoss"]

TUNED_MODEL_PATH = "./tuned/finetuned"
BATCH_SIZE = 16

class Embedder:
    def __init__(self, model_name: str, embedding_size: int, load_tuned=False):
        self.embedding_size = embedding_size

        self.model = (
            SentenceTransformer(model_name)
            if not load_tuned
            else SentenceTransformer(TUNED_MODEL_PATH)
        )

    async def embed(self, query: str) -> list[float] | np.ndarray:
        return self.model.encode(query).tolist()

    def finetune(self, loss: nn.Module, train_corpus: list[str], train_query: list[str],
                 epochs: int, show_plot=False) -> None:

        class LossCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    losses = logs.get("loss", None)
                    if losses:
                        loss_history.append(losses)

        train_examples = {"query": train_query, "code": train_corpus}
        train_dataset = Dataset.from_dict(train_examples)

        args = SentenceTransformerTrainingArguments(
            output_dir=TUNED_MODEL_PATH,
            num_train_epochs=epochs,
            per_device_train_batch_size=BATCH_SIZE,
        )

        loss_history = []

        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            loss=loss(self.model),
            callbacks=[LossCallback()],
        )
        trainer.train()
        self.model.save(TUNED_MODEL_PATH)

        if show_plot:
            plt.figure(figsize=(8, 5))
            plt.plot(loss_history, label="Training loss")
            plt.xlabel("Logging step")
            plt.ylabel("Mean Loss")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.show()
