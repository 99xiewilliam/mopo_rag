from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class VectorIndex:
    embeddings: np.ndarray
    texts: List[str]

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        sims = cosine_similarity(query_embedding, self.embeddings)[0]
        idxs = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i])) for i in idxs]


