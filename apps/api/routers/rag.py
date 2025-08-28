from typing import List, Optional

from fastapi import APIRouter, Body
from pydantic import BaseModel, Field

from rag.pipelines.simple_pipeline import SimplePipeline, PipelineConfig
from rag.utils.config import get_data_input_dir


router = APIRouter(tags=["rag"])

_pipeline: Optional[SimplePipeline] = None


class IndexRequest(BaseModel):
    source_dir: Optional[str] = Field(
        default=None,
        description="目录路径：从该目录加载 .txt 文档以构建简单内存索引（占位实现）",
    )
    config_path: Optional[str] = Field(
        default=None,
        description="可选：YAML 配置路径；若未提供 source_dir，则从配置 data.input_dir 读取",
    )


class IndexResponse(BaseModel):
    status: str
    num_docs: int


class AskRequest(BaseModel):
    query: str = Field(..., description="用户问题")
    top_k: int = Field(5, ge=1, le=50, description="检索返回文档数量（占位实现）")


class AskResponse(BaseModel):
    answer: str
    contexts: List[str]


def _get_pipeline() -> SimplePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = SimplePipeline(PipelineConfig())
    return _pipeline


@router.post("/index", response_model=IndexResponse)
def build_index(payload: IndexRequest = Body(...)) -> IndexResponse:
    pipeline = _get_pipeline()
    source_dir = payload.source_dir or get_data_input_dir(payload.config_path)
    num_docs = pipeline.index(source_dir=source_dir)
    return IndexResponse(status="ok", num_docs=num_docs)


@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest = Body(...)) -> AskResponse:
    pipeline = _get_pipeline()
    result = pipeline.ask(query=payload.query, top_k=payload.top_k)
    return AskResponse(answer=result.answer, contexts=result.contexts)


