from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple
import math
import re


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


# ------------------------- Retrieval metrics -------------------------

def hit_at_k(retrieved: Sequence[str], gold_ids: Sequence[str]) -> float:
    if not retrieved:
        return 0.0
    gold = set(gold_ids)
    return 1.0 if any(doc_id in gold for doc_id in retrieved) else 0.0


def recall_at_k(retrieved: Sequence[str], gold_ids: Sequence[str]) -> float:
    if not gold_ids:
        return 0.0
    gold = set(gold_ids)
    got = sum(1 for doc_id in retrieved if doc_id in gold)
    return got / float(len(gold))


def mrr_at_k(retrieved: Sequence[str], gold_ids: Sequence[str]) -> float:
    gold = set(gold_ids)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in gold:
            return 1.0 / float(rank)
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], gold_ids: Sequence[str]) -> float:
    """Binary relevance nDCG@k."""
    gold = set(gold_ids)
    if not retrieved:
        return 0.0
    dcg = 0.0
    for i, doc_id in enumerate(retrieved, start=1):
        rel = 1.0 if doc_id in gold else 0.0
        if rel > 0:
            dcg += rel / math.log2(i + 1)
    # ideal DCG
    ideal_rel = [1.0] * min(len(gold), len(retrieved))
    idcg = 0.0
    for i, rel in enumerate(ideal_rel, start=1):
        idcg += rel / math.log2(i + 1)
    return dcg / idcg if idcg > 0 else 0.0


# ------------------------- Generation metrics -------------------------

def exact_match(pred: str, gold_answers: Iterable[str]) -> float:
    p = _normalize_text(pred)
    for g in gold_answers:
        if _normalize_text(g) == p:
            return 1.0
    return 0.0


def f1_score(pred: str, gold_answers: Iterable[str]) -> float:
    p_tokens = _normalize_text(pred).split()
    if not p_tokens:
        return 0.0
    best = 0.0
    for g in gold_answers:
        g_tokens = _normalize_text(g).split()
        if not g_tokens:
            continue
        common = {}
        for t in p_tokens:
            common[t] = min(p_tokens.count(t), g_tokens.count(t))
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / float(len(p_tokens))
        recall = num_same / float(len(g_tokens))
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best



