from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    faiss = None  # type: ignore


@dataclass
class FaissIndex:
    index: any
    texts: List[str]

    @classmethod
    def build(cls, embeddings: np.ndarray, texts: List[str], metric: str = "IP") -> "FaissIndex":
        if faiss is None:
            raise ImportError("faiss is not installed. Please `pip install faiss-cpu`.")
        dim = embeddings.shape[1]
        if metric == "IP":
            index = faiss.IndexFlatIP(dim)
        else:
            index = faiss.IndexFlatL2(dim)
        index.add(embeddings.astype(np.float32))
        return cls(index=index, texts=texts)

    def search(self, query_vectors: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        scores, idxs = self.index.search(query_vectors.astype(np.float32), top_k)
        # 只取单查询
        s = scores[0]
        i = idxs[0]
        pairs = [(int(ii), float(ss)) for ii, ss in zip(i, s)]
        return pairs


