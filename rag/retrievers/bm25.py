from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from rank_bm25 import BM25Okapi


@dataclass
class BM25Retriever:
    corpus_tokens: List[List[str]]
    texts: List[str]

    @classmethod
    def from_texts(cls, texts: List[str]) -> "BM25Retriever":
        tokens = [t.lower().split() for t in texts]
        return cls(corpus_tokens=tokens, texts=texts)

    def search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        bm25 = BM25Okapi(self.corpus_tokens)
        scores = bm25.get_scores(query.lower().split())
        idxs = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [(i, float(scores[i])) for i in idxs]


