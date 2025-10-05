import numpy as np
from datasets import Dataset
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from transformers import TrainerCallback
import matplotlib.pyplot as plt

class Embedder:
    def __init__(self, model_name: str, embedding_size=1024, load_tuned=False):
        self.embedding_size = embedding_size
        self.model = (
            SentenceTransformer(model_name)
            if not load_tuned
            else SentenceTransformer("./tuned/finetuned")
        )

    async def embed(self, query: str) -> list[float] | np.ndarray:
        return self.model.encode(query).tolist()

    def finetune(self, train_corpus: list[str], train_query: list[str], epochs: int, show_plot=False) -> None:
        train_examples = {"query": train_query, "code": train_corpus}
        train_dataset = Dataset.from_dict(train_examples)

        train_loss = losses.MultipleNegativesRankingLoss(self.model)

        args = SentenceTransformerTrainingArguments(
            output_dir="./tuned/finetuned",
            num_train_epochs=epochs,
            per_device_train_batch_size=16,
        )

        loss_history = []

        class LossCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    losses = logs.get("loss", None)
                    if losses:
                        loss_history.append(losses)

        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            loss=train_loss,
            callbacks=[LossCallback()],
        )
        trainer.train()
        self.model.save("./tuned/finetuned")

        if show_plot:
            plt.figure(figsize=(8, 5))
            plt.plot(loss_history, label="Training loss")
            plt.xlabel("Logging step")
            plt.ylabel("Mean Loss")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.show()
