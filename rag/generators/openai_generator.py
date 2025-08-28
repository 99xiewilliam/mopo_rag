from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import os

try:
    import openai  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    openai = None  # type: ignore


@dataclass
class OpenAIConfig:
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.2


class OpenAIGenerator:
    def __init__(self, config: OpenAIConfig) -> None:
        if openai is None:
            raise ImportError("openai package is not installed. `pip install openai`." )
        key = config.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY is required")
        openai.api_key = key
        if config.api_base:
            openai.base_url = config.api_base
        self.model = config.model
        self.temperature = config.temperature

    def generate(self, question: str, contexts: List[str]) -> str:
        prompt = "\n".join(["You are a helpful assistant.", f"Question: {question}", "Context:"] + contexts)
        resp = openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return resp.choices[0].message.content


