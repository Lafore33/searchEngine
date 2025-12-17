import uvicorn
import pydantic
from fastapi import FastAPI
from datasets import load_dataset
from src.utils import load_test_data
from src.embedder.dense import DenseEmbedder
from src.datasource.dense import DenseDatasource

collection_name = "code-test"
corpus_dataset = load_dataset("CoIR-Retrieval/cosqa", "corpus")["corpus"]
test_corpus = [function for partition, function in zip(corpus_dataset["partition"], corpus_dataset["text"]) if
               partition == "test"]



models = [
          DenseEmbedder("Qwen/Qwen3-Embedding-0.6B", 1024),
          DenseEmbedder("sentence-transformers/all-MiniLM-L6-v2", 384),
          # DenseEmbedder("sentence-transformers/all-MiniLM-L6-v2", 384, load_tuned=True)
          ]

model = models[0]

db = DenseDatasource(model)
load_test_data(db, collection_name, test_corpus, True)

app = FastAPI()

@app.get("/")
async def root():
    return {"status": "ok"}

@app.get("/search/")
async def search(query: str):
    results = db.search_functions(collection_name, query)
    return {"function": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)