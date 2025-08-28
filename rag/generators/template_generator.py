from __future__ import annotations

from dataclasses import dataclass
from typing import List

from jinja2 import Template


@dataclass
class TemplateConfig:
    template_str: str


class TemplateGenerator:
    def __init__(self, config: TemplateConfig) -> None:
        self.template = Template(config.template_str)

    def generate(self, question: str, contexts: List[str]) -> str:
        return self.template.render(question=question, context_list=contexts)


