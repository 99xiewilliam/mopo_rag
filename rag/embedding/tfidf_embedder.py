from __future__ import annotations

from dataclasses import dataclass
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class TfidfConfig:
    max_features: int = 50000
    ngram_range: tuple = (1, 2)


class TfidfEmbedder:
    def __init__(self, config: TfidfConfig) -> None:
        self.config = config
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            ngram_range=self.config.ngram_range,
        )

    def fit_transform(self, texts: List[str]):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts: List[str]):
        return self.vectorizer.transform(texts)


