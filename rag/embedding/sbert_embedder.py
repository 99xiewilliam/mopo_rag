from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception as exc:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore


@dataclass
class SBertConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    normalize: bool = True


class SBertEmbedder:
    def __init__(self, config: SBertConfig) -> None:
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is not installed. Please `pip install sentence-transformers`." )
        self.model = SentenceTransformer(config.model_name, device=config.device)
        self.normalize = config.normalize

    def encode(self, texts: List[str]) -> np.ndarray:
        vectors = self.model.encode(texts, normalize_embeddings=self.normalize, batch_size=64, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)


