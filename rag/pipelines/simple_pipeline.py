from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class PipelineConfig:
    default_top_k: int = 5


@dataclass
class AskResult:
    answer: str
    contexts: List[str]


class InMemoryCorpus:
    def __init__(self) -> None:
        self.documents: List[str] = []

    def load_from_dir(self, dir_path: Optional[str]) -> int:
        self.documents.clear()
        if dir_path is None:
            return 0
        base = Path(dir_path)
        if not base.exists():
            return 0
        for file in base.rglob("*.txt"):
            try:
                text = file.read_text(encoding="utf-8", errors="ignore")
                if text.strip():
                    self.documents.append(text)
            except Exception:
                continue
        return len(self.documents)


class SimpleRetriever:
    def __init__(self, corpus: InMemoryCorpus) -> None:
        self.corpus = corpus

    def retrieve(self, query: str, top_k: int) -> List[str]:
        if not self.corpus.documents:
            return []
        query_terms = {t.lower() for t in query.split() if t.strip()}
        if not query_terms:
            return self.corpus.documents[:top_k]
        scored = []
        for doc in self.corpus.documents:
            lower = doc.lower()
            score = sum(1 for t in query_terms if t in lower)
            if score > 0:
                scored.append((score, doc))
        if not scored:
            return self.corpus.documents[:top_k]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:top_k]]


class SimpleGenerator:
    def generate(self, query: str, contexts: List[str]) -> str:
        if not contexts:
            return f"No context available. Query: {query}"
        preview = contexts[0][:280].replace("\n", " ")
        return f"Answer (stub): based on retrieved context, here's a draft for: '{query}'. Context preview: {preview}"


class SimplePipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.corpus = InMemoryCorpus()
        self.retriever = SimpleRetriever(self.corpus)
        self.generator = SimpleGenerator()

    def index(self, source_dir: Optional[str]) -> int:
        return self.corpus.load_from_dir(source_dir)

    def ask(self, query: str, top_k: Optional[int] = None) -> AskResult:
        k = top_k or self.config.default_top_k
        contexts = self.retriever.retrieve(query=query, top_k=k)
        answer = self.generator.generate(query=query, contexts=contexts)
        return AskResult(answer=answer, contexts=contexts)


