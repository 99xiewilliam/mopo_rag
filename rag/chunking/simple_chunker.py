from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class ChunkingConfig:
    chunk_size: int = 800
    chunk_overlap: int = 120


class SimpleChunker:
    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config

    def chunk_text(self, text: str) -> List[str]:
        size = max(1, self.config.chunk_size)
        overlap = max(0, self.config.chunk_overlap)
        if not text:
            return []
        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + size)
            chunks.append(text[start:end])
            if end == len(text):
                break
            start = end - overlap
            if start < 0:
                start = 0
                if start >= len(text):
                    break
        return chunks

    def chunk_files(self, files: Iterable[Path]) -> List[str]:
        chunks: List[str] = []
        for fp in files:
            try:
                text = fp.read_text(encoding="utf-8", errors="ignore")
                chunks.extend(self.chunk_text(text))
            except Exception:
                continue
        return chunks


