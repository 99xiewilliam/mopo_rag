from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from rag.chunking.simple_chunker import ChunkingConfig, SimpleChunker
from rag.embedding.sbert_embedder import SBertConfig, SBertEmbedder
from rag.index.faiss_index import FaissIndex
from rag.retrievers.bm25 import BM25Retriever
from rag.retrievers.hybrid import HybridRetriever
from rag.generators.template_generator import TemplateConfig, TemplateGenerator


@dataclass
class FaissConfig:
    input_dir: str
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    top_k: int = 8
    alpha: float = 0.6
    chunk_size: int = 800
    chunk_overlap: int = 120


@dataclass
class FaissResult:
    answer: str
    contexts: List[str]


class FaissPipeline:
    def __init__(self, cfg: FaissConfig) -> None:
        self.cfg = cfg
        self.chunker = SimpleChunker(ChunkingConfig(cfg.chunk_size, cfg.chunk_overlap))
        self.embedder = SBertEmbedder(SBertConfig(model_name=cfg.model_name, device=cfg.device))
        self.generator = TemplateGenerator(
            TemplateConfig(template_str=(Path(__file__).resolve().parents[2] / "prompts" / "qa.jinja").read_text(encoding="utf-8"))
        )
        self.texts: List[str] = []
        self.faiss_index: Optional[FaissIndex] = None
        self.bm25: Optional[BM25Retriever] = None

    def index(self) -> int:
        base = Path(self.cfg.input_dir)
        files = list(base.rglob("*.txt")) if base.exists() else []
        self.texts = self.chunker.chunk_files(files)
        if not self.texts:
            self.faiss_index = None
            self.bm25 = None
            return 0
        emb = self.embedder.encode(self.texts)
        self.faiss_index = FaissIndex.build(emb, self.texts, metric="IP")
        self.bm25 = BM25Retriever.from_texts(self.texts)
        return len(self.texts)

    def ask(self, query: str, top_k: Optional[int] = None) -> FaissResult:
        k = top_k or self.cfg.top_k
        if not self.texts or self.faiss_index is None or self.bm25 is None:
            return FaissResult(answer="No index built.", contexts=[])
        qv = self.embedder.encode([query])
        dense = self.faiss_index.search(qv, k)
        sparse = self.bm25.search(query, k)
        merged = HybridRetriever(self.cfg.alpha).merge(dense, sparse, k)
        contexts = [self.texts[i] for i, _ in merged]
        answer = self.generator.generate(question=query, contexts=contexts)
        return FaissResult(answer=answer, contexts=contexts)


