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
│   │   ├── embedder.py
│   │   ├── sparse.py
│   │
│   ├── notebooks
│   │   ├── hybrid_retrieval.ipynb
│   │   ├── sparse_retrieval.ipynb
│   │   ├── training.ipynb
│   ├── metrics.py
│   ├── utils.py
```

### Descriptions

Datasource class represents the vector storage. For vector storage I decided to use Qdrant. I have created a cluster, which is hosted in the cloud. I use parameters provided in the .env-example to connect to the cluster.
Embedder is used for transformers. Two transformers were used: MiniLM and Qwen. MiniLM was finetuned and Qwen was used as an additional comparison between MiniLM's

Metrics file contains implementations of the required metrics, utils contains helper functions I've used. 
The notebook contains tuning, displays the plot and metrics.

### Loss function selection

I've chosen to use Multiple Negatives Ranking Loss. Firstly, it does not require to have a negative pair for the query, as it considers every other doc, except positive one, to be negative. And with the given dataset, that's exactly what I need.
As, for example, Triplet Loss requires to have positive as well as negative pairs. Or, for example Cross-Entropy Ranking Losses, they require some additional setup, and they are computationally havier.
So, as far as I understand, this loss is kind of to-go option here.

## Some additional thoughts and results

As the plot shows, metrics were improved after tuning. For retrieval, I've used dense approach here, however maybe worth trying in the future to use hybrid approach(sparse + dense).
Also, the rerank here could be a good idea.
Regarding the "How do the metrics change when you apply the model to function names instead of whole bodies?",
I have not tried this, however, in such a case, it'd be highly dependent on the quality of function naming, which may cause issues.

### Feedback
First of all — thank you for creating this internship task. It was interesting and gave me a good challenge. I appreciate the effort that went into putting it together. Working on it allowed me to learn a lot and explore different approaches.
I understand that there is still much work to be done to bring the project to its full potential, but I hope my submission reflects both my skills and my enthusiasm for the subject. I am eager to receive your feedback.
This opportunity means a great deal to me, and I am genuinely passionate about contributing to your work. I look forward to the possibility of growing and learning with your team.
Thank you again for your time, effort, and consideration.

