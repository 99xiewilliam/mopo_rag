from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class HybridRetriever:
    alpha: float  # weight for dense score

    def merge(
        self,
        dense: List[Tuple[int, float]],
        sparse: List[Tuple[int, float]],
        top_k: int,
    ) -> List[Tuple[int, float]]:
        score_map = {}
        for i, s in dense:
            score_map[i] = score_map.get(i, 0.0) + self.alpha * s
        for i, s in sparse:
            score_map[i] = score_map.get(i, 0.0) + (1.0 - self.alpha) * s
        merged = sorted(score_map.items(), key=lambda x: -x[1])[:top_k]
        return [(int(i), float(s)) for i, s in merged]


