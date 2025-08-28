from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None  # type: ignore


@dataclass
class CrossEncoderConfig:
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "cpu"


class CEReranker:
    def __init__(self, config: CrossEncoderConfig) -> None:
        if CrossEncoder is None:
            raise ImportError("sentence-transformers is required for cross-encoder reranker.")
        self.model = CrossEncoder(config.model_name, device=config.device)

    def rerank(self, query: str, candidates: List[Tuple[int, str]], top_k: int) -> List[Tuple[int, float]]:
        pairs = [[query, text] for _, text in candidates]
        scores = self.model.predict(pairs).tolist()
        idxs = list(range(len(candidates)))
        idxs.sort(key=lambda i: -scores[i])
        ranked = [(candidates[i][0], float(scores[i])) for i in idxs[:top_k]]
        return ranked


