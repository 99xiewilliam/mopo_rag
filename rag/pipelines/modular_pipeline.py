from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from rag.chunking.simple_chunker import ChunkingConfig, SimpleChunker
from rag.embedding.tfidf_embedder import TfidfConfig, TfidfEmbedder
from rag.generators.template_generator import TemplateConfig, TemplateGenerator
from rag.index.vector_index import VectorIndex
from rag.retrievers.bm25 import BM25Retriever
from rag.retrievers.hybrid import HybridRetriever


@dataclass
class ModularConfig:
    input_dir: str
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 8
    alpha: float = 0.6


@dataclass
class ModularResult:
    answer: str
    contexts: List[str]


class ModularPipeline:
    def __init__(self, config: ModularConfig) -> None:
        self.cfg = config
        self.chunker = SimpleChunker(ChunkingConfig(config.chunk_size, config.chunk_overlap))
        self.embedder = TfidfEmbedder(TfidfConfig())
        self.generator = TemplateGenerator(
            TemplateConfig(template_str=(Path(__file__).resolve().parents[2] / "prompts" / "qa.jinja").read_text(encoding="utf-8"))
        )
        self.vector_index: Optional[VectorIndex] = None
        self.bm25: Optional[BM25Retriever] = None
        self.texts: List[str] = []

    def index(self) -> int:
        base = Path(self.cfg.input_dir)
        files = list(base.rglob("*.txt")) if base.exists() else []
        self.texts = self.chunker.chunk_files(files)
        if not self.texts:
            self.vector_index = None
            self.bm25 = None
            return 0
        X = self.embedder.fit_transform(self.texts)
        self.vector_index = VectorIndex(embeddings=X.toarray().astype(np.float32), texts=self.texts)
        self.bm25 = BM25Retriever.from_texts(self.texts)
        return len(self.texts)

    def ask(self, query: str, top_k: Optional[int] = None) -> ModularResult:
        k = top_k or self.cfg.top_k
        if not self.texts or self.vector_index is None or self.bm25 is None:
            return ModularResult(answer="No index built.", contexts=[])
        qv = self.embedder.transform([query]).toarray().astype(np.float32)
        dense = self.vector_index.search(qv, k)
        sparse = self.bm25.search(query, k)
        hybrid = HybridRetriever(self.cfg.alpha).merge(dense, sparse, k)
        contexts = [self.texts[i] for i, _ in hybrid]
        answer = self.generator.generate(question=query, contexts=contexts)
        return ModularResult(answer=answer, contexts=contexts)


