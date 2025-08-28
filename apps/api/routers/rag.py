from typing import List, Optional

from fastapi import APIRouter, Body
from pydantic import BaseModel, Field

from rag.pipelines.simple_pipeline import SimplePipeline, PipelineConfig
from rag.pipelines.factory import build_pipeline_from_config, ConfigDrivenPipeline
from rag.utils.config import load_config, get_data_input_dir, apply_env_from_config


router = APIRouter(tags=["rag"])

_pipeline_simple: Optional[SimplePipeline] = None
_pipeline_config: Optional[ConfigDrivenPipeline] = None


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


def _get_simple() -> SimplePipeline:
    global _pipeline_simple
    if _pipeline_simple is None:
        _pipeline_simple = SimplePipeline(PipelineConfig())
    return _pipeline_simple


def _get_config_pipeline(cfg_path: str) -> ConfigDrivenPipeline:
    global _pipeline_config
    if _pipeline_config is None:
        cfg = load_config(cfg_path)
        _pipeline_config = build_pipeline_from_config(cfg)
    return _pipeline_config


@router.post("/index", response_model=IndexResponse)
def build_index(payload: IndexRequest = Body(...)) -> IndexResponse:
    # 优先：如果提供了 config_path，则按配置驱动（允许 source_dir 覆盖 YAML 中的目录）
    if payload.config_path:
        cfg = load_config(payload.config_path)
        apply_env_from_config(cfg)
        cfg_source = get_data_input_dir(payload.config_path)
        source_dir = payload.source_dir or cfg_source
        pipe = _get_config_pipeline(payload.config_path)
        num_docs = pipe.index(source_dir or "")
        return IndexResponse(status="ok", num_docs=num_docs)

    # 否则：仅提供了 source_dir 时，使用简单流水线
    if payload.source_dir:
        simple = _get_simple()
        num_docs = simple.index(source_dir=payload.source_dir)
        return IndexResponse(status="ok", num_docs=num_docs)

    # 最后回退（未提供任何路径）
    simple = _get_simple()
    num_docs = simple.index(source_dir=None)
    return IndexResponse(status="ok", num_docs=num_docs)


@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest = Body(...)) -> AskResponse:
    if _pipeline_config is not None:
        result = _pipeline_config.ask(query=payload.query, top_k=payload.top_k)
        return AskResponse(answer=result.answer, contexts=result.contexts)
    simple = _get_simple()
    result = simple.ask(query=payload.query, top_k=payload.top_k)
    return AskResponse(answer=result.answer, contexts=result.contexts)


