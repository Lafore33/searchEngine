# Structure

## Project structure
```
├── src            
│   ├── datasource
│   │   ├── base.py  
│   │   ├── dense.py 
│   │   ├── hybrid.py
│   │   ├── sparse.py   
│   │
│   ├── embedder
│   │   ├── dense.py
│   │   ├── sparse.py
│   │
│   ├── notebooks
│   │   ├── dense_retrieval.ipynb
│   │   ├── hybrid_retrieval.ipynb
│   │   ├── sparse_retrieval.ipynb
│   │   ├── training.ipynb
│   │
│   ├── api.py
│   ├── metrics.py
│   ├── utils.py
```

### Descriptions

Datasource class represents the vector storage. For vector storage I decided to use Qdrant in memory. The path in the .env-example is needed to save trained models.
Embedder is used for transformers. Two transformers were used: MiniLM and Qwen. MiniLM was finetuned and Qwen was used as an additional comparison between MiniLM's

Metrics file contains implementations of the required metrics, utils contains helper functions I've used. 
The notebook contains tuning, displays the plot and metrics, as well as different methods of retrieval. I have tried dense, sparse, hybrid approaches and utilized mlflow for tracking these experiments.

I have also implemented simple inference with one endpoint: /search. There is also a dockerfile for a containerisation of the application, as well as the docker-image.yml workflow file for GitHub actions. The purpose of this workflow is to rebuild image each time the code is pushed to the main branch, it can be triggered manually as well.

To run the application you need to pull the container from the GitHub actions, and run it passing the environment variable TUNED_PATH:
```
docker run -p 8000:8000 -e TUNED_PATH=/models ghcr.io/lafore33/search-engine
```