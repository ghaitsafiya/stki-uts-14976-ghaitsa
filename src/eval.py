"""
======================================================
File: eval.py
Deskripsi:
Modul evaluasi performa sistem IR.
Metrik yang digunakan:
- Precision@k, Recall@k
- MAP@k (Mean Average Precision)
- nDCG@k (Normalized Discounted Cumulative Gain)
======================================================
"""

import math
from typing import List


def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Hitung precision: proporsi hasil relevan di top-k"""
    retrieved_k = retrieved[:k]
    if not retrieved_k:
        return 0.0
    rel_count = sum(1 for d in retrieved_k if d in relevant)
    return rel_count / len(retrieved_k)


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Hitung recall: proporsi dokumen relevan yang berhasil ditemukan"""
    retrieved_k = retrieved[:k]
    if not relevant:
        return 0.0
    rel_count = sum(1 for d in retrieved_k if d in relevant)
    return rel_count / len(relevant)


def apk(actual: List[str], predicted: List[str], k: int) -> float:
    """Average Precision@k (AP) buat satu query"""
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    if not actual:
        return 0.0
    return score / min(len(actual), k)


def mapk(actual_list, predicted_list, k=5):
    """Mean Average Precision@k (MAP@k) buat beberapa query"""
    return sum(apk(a, p, k) for a, p in zip(actual_list, predicted_list)) / len(actual_list)


def dcg_at_k(retrieved: List[str], relevant_set: set, k: int):
    """Discounted Cumulative Gain (DCG)"""
    dcg = 0.0
    for i, doc in enumerate(retrieved[:k], start=1):
        rel = 1 if doc in relevant_set else 0
        dcg += (2 ** rel - 1) / math.log2(i + 1)
    return dcg


def ndcg_at_k(retrieved: List[str], relevant_set: set, k: int):
    """Normalized DCG (skor 0–1)"""
    dcg = dcg_at_k(retrieved, relevant_set, k)
    ideal_rels = min(len(relevant_set), k)
    ideal = sum((2 ** 1 - 1) / math.log2(i + 1) for i in range(1, ideal_rels + 1))
    return dcg / ideal if ideal > 0 else 0.0
