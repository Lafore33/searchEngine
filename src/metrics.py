import numpy as np
from typing import Any
from numpy import floating


def recall_at_k(ground_truth: list[str], search_results: list[list[str]], k=10) -> floating[Any]:

    relevant_answers = [1 for idx, gt in enumerate(ground_truth) if gt in search_results[idx][:k]]

    return np.mean(relevant_answers)

def mrr_at_k(ground_truth: list[str], search_results: list[list[str]], k=10) -> floating[Any]:

    relevant_rankings = []
    for idx, gt in enumerate(ground_truth):
        if gt in search_results[idx][:k]:
            relevant_rankings.append(1 / (search_results[idx].index(gt) + 1))
        else:
            relevant_rankings.append(0)

    return np.mean(relevant_rankings)

def ndcg_at_k(ground_truth: list[str], search_results: list[list[str]], k=10) -> floating[Any]:

    scores = []
    for idx, gt in enumerate(ground_truth):
        if gt in search_results[idx][:k]:
            dcg = 1 / np.log2((search_results[idx].index(gt) + 2))
            idcg = 1
            scores.append(dcg / idcg)
        else:
            scores.append(0)

    return np.mean(scores)