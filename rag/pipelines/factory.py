from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from rag.chunking.simple_chunker import ChunkingConfig, SimpleChunker
from rag.embedding.tfidf_embedder import TfidfConfig, TfidfEmbedder
from rag.embedding.sbert_embedder import SBertConfig, SBertEmbedder
from rag.index.vector_index import VectorIndex
from rag.index.faiss_index import FaissIndex
from rag.retrievers.bm25 import BM25Retriever
from rag.retrievers.hybrid import HybridRetriever
from rag.rerankers.cross_encoder import CEReranker, CrossEncoderConfig
from rag.generators.template_generator import TemplateConfig, TemplateGenerator
from rag.generators.openai_generator import OpenAIConfig, OpenAIGenerator


@dataclass
class ConfigDrivenResult:
    answer: str
    contexts: List[str]


class ConfigDrivenPipeline:
    def __init__(
        self,
        chunker: SimpleChunker,
        embedder: Any,
        dense_index_type: str,
        retriever_type: str,
        top_k: int,
        alpha: float,
        generator: Any,
        reranker: Optional[Any] = None,
    ) -> None:
        self.chunker = chunker
        self.embedder = embedder
        self.dense_index_type = dense_index_type  # "cosine" | "faiss"
        self.retriever_type = retriever_type      # "dense" | "bm25" | "hybrid"
        self.top_k_default = top_k
        self.alpha = alpha
        self.generator = generator
        self.reranker = reranker

        self.texts: List[str] = []
        self.vector_index: Optional[VectorIndex] = None
        self.faiss_index: Optional[FaissIndex] = None
        self.bm25: Optional[BM25Retriever] = None

    def index(self, input_dir: str) -> int:
        base = Path(input_dir)
        files = list(base.rglob("*.txt")) if base.exists() else []
        self.texts = self.chunker.chunk_files(files)
        if not self.texts:
            self.vector_index = None
            self.faiss_index = None
            self.bm25 = None
            return 0

        # Embeddings
        if hasattr(self.embedder, "fit_transform"):
            X = self.embedder.fit_transform(self.texts)
            vectors = X.toarray().astype(np.float32)
        else:
            vectors = self.embedder.encode(self.texts)

        # Dense index
        if self.dense_index_type == "faiss":
            self.faiss_index = FaissIndex.build(vectors, self.texts, metric="IP")
            self.vector_index = None
        else:
            self.vector_index = VectorIndex(embeddings=vectors, texts=self.texts)
            self.faiss_index = None

        # Sparse index (optional)
        self.bm25 = BM25Retriever.from_texts(self.texts)
        return len(self.texts)

    def _dense_search(self, qv: np.ndarray, k: int) -> List[Tuple[int, float]]:
        if self.faiss_index is not None:
            return self.faiss_index.search(qv, k)
        if self.vector_index is not None:
            return self.vector_index.search(qv, k)
        return []

    def ask(self, query: str, top_k: Optional[int] = None) -> ConfigDrivenResult:
        k = top_k or self.top_k_default
        if not self.texts:
            return ConfigDrivenResult(answer="No index built.", contexts=[])

        # Encode query
        if hasattr(self.embedder, "transform"):
            qv = self.embedder.transform([query]).toarray().astype(np.float32)
        else:
            qv = self.embedder.encode([query])

        dense = self._dense_search(qv, k)
        sparse = self.bm25.search(query, k) if self.bm25 is not None else []

        if self.retriever_type == "dense":
            merged = dense
        elif self.retriever_type == "bm25":
            merged = sparse
        else:
            merged = HybridRetriever(self.alpha).merge(dense, sparse, k)

        candidates = [(i, self.texts[i]) for i, _ in merged]
        if self.reranker is not None and candidates:
            ranked = self.reranker.rerank(query, candidates, top_k=k)
            ctx_ids = [i for i, _ in ranked]
        else:
            ctx_ids = [i for i, _ in merged]

        contexts = [self.texts[i] for i in ctx_ids[:k]]
        answer = self.generator.generate(question=query, contexts=contexts)
        return ConfigDrivenResult(answer=answer, contexts=contexts)


def build_pipeline_from_config(cfg: Dict[str, Any]) -> ConfigDrivenPipeline:
    data = cfg.get("data", {})
    components = cfg.get("components", {})
    retriever_cfg = cfg.get("retriever", {})

    # Chunker
    chunking = components.get("chunking", {})
    chunker = SimpleChunker(ChunkingConfig(
        chunk_size=int(chunking.get("chunk_size", 800)),
        chunk_overlap=int(chunking.get("chunk_overlap", 120)),
    ))

    # Embedder
    emb = components.get("embedding", {"type": "tfidf"})
    emb_type = emb.get("type", "tfidf")
    if emb_type == "sbert":
        embedder = SBertEmbedder(SBertConfig(
            model_name=emb.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            device=emb.get("device", "cpu"),
            normalize=bool(emb.get("normalize", True)),
        ))
    else:
        embedder = TfidfEmbedder(TfidfConfig(
            max_features=int(emb.get("max_features", 50000)),
            ngram_range=tuple(emb.get("ngram_range", (1, 2))),
        ))

    # Dense index type
    index_cfg = components.get("index", {"type": "cosine"})
    dense_index_type = index_cfg.get("type", "cosine")  # cosine | faiss

    # Generator
    gen = components.get("generator", {"type": "template"})
    gen_type = gen.get("type", "template")
    if gen_type == "openai":
        generator = OpenAIGenerator(OpenAIConfig(
            model=gen.get("model", "gpt-4o-mini"),
            api_key=gen.get("api_key"),
            api_base=gen.get("api_base"),
            temperature=float(gen.get("temperature", 0.2)),
        ))
    else:
        template_path = gen.get("template_path") or (Path(__file__).resolve().parents[2] / "prompts" / "qa.jinja")
        template_str = Path(str(template_path)).read_text(encoding="utf-8")
        generator = TemplateGenerator(TemplateConfig(template_str=template_str))

    # Reranker (optional)
    rer = components.get("reranker", {"type": "none"})
    rer_type = rer.get("type", "none")
    if rer_type == "cross_encoder":
        reranker = CEReranker(CrossEncoderConfig(model_name=rer.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"), device=rer.get("device", "cpu")))
    else:
        reranker = None

    # Retriever settings
    retriever_type = retriever_cfg.get("type", "hybrid")  # dense | bm25 | hybrid
    top_k = int(retriever_cfg.get("top_k", 8))
    alpha = float(retriever_cfg.get("alpha", 0.6))

    return ConfigDrivenPipeline(
        chunker=chunker,
        embedder=embedder,
        dense_index_type=dense_index_type,
        retriever_type=retriever_type,
        top_k=top_k,
        alpha=alpha,
        generator=generator,
        reranker=reranker,
    )


