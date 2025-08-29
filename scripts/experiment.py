from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import time
import yaml

from rag.metrics.eval import hit_at_k, recall_at_k, mrr_at_k, ndcg_at_k, exact_match, f1_score
from rag.pipelines.factory import build_pipeline_from_config
from rag.utils.config import load_config


def _hash_texts_dir(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return "na"
    sha = hashlib.sha256()
    for f in sorted(p.rglob("*.txt")):
        sha.update(f.name.encode())
        try:
            sha.update(f.read_bytes())
        except Exception:
            continue
    return sha.hexdigest()[:12]


def _cartesian_product(sweeps: Dict[str, Iterable[Any]]) -> List[Dict[str, Any]]:
    keys = list(sweeps.keys())
    vals = [list(v) for v in sweeps.values()]
    out: List[Dict[str, Any]] = []
    for combo in itertools.product(*vals):
        out.append({k: v for k, v in zip(keys, combo)})
    return out


def _deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_eval_dataset(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def evaluate_pipeline(pipeline, dataset: List[Dict[str, Any]], retriever_k: int) -> Dict[str, float]:
    ems: List[float] = []
    f1s: List[float] = []
    hits: List[float] = []
    recs: List[float] = []
    mrrs: List[float] = []
    ndcgs: List[float] = []

    for ex in dataset:
        q = ex.get("query") or ex.get("question") or ""
        gold_answers = ex.get("answers", [])
        gold_docs = ex.get("doc_ids", [])

        res = pipeline.ask(q, top_k=retriever_k)

        # 生成指标
        ems.append(exact_match(res.answer, gold_answers) if gold_answers else 0.0)
        f1s.append(f1_score(res.answer, gold_answers) if gold_answers else 0.0)

        # 检索指标（需要 pipeline 返回上下文原始 id，当前用文本内容哈希代替）
        retrieved_ids = [hashlib.sha1(ctx.encode()).hexdigest()[:10] for ctx in res.contexts]
        gold_ids_h = [hashlib.sha1(str(g).encode()).hexdigest()[:10] for g in gold_docs]
        hits.append(hit_at_k(retrieved_ids, gold_ids_h) if gold_docs else 0.0)
        recs.append(recall_at_k(retrieved_ids, gold_ids_h) if gold_docs else 0.0)
        mrrs.append(mrr_at_k(retrieved_ids, gold_ids_h) if gold_docs else 0.0)
        ndcgs.append(ndcg_at_k(retrieved_ids, gold_ids_h) if gold_docs else 0.0)

    def avg(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "EM": avg(ems),
        "F1": avg(f1s),
        "Hit@k": avg(hits),
        "Recall@k": avg(recs),
        "MRR@k": avg(mrrs),
        "nDCG@k": avg(ndcgs),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="experiment yaml")
    ap.add_argument("--default", default=str(Path(__file__).resolve().parents[1] / "rag" / "config" / "default.yaml"))
    ap.add_argument("--output", default="runs/experiment")
    args = ap.parse_args()

    exp_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base_cfg = load_config(args.default)

    sweeps: Dict[str, Any] = exp_cfg.get("sweeps", {})
    fixed: Dict[str, Any] = exp_cfg.get("fixed", {})
    data_cfg: Dict[str, Any] = exp_cfg.get("data", {})

    corpus_dir = data_cfg.get("corpus_path") or base_cfg.get("data", {}).get("input_dir")
    dataset_path = data_cfg.get("dataset_path")
    retr_k = int(exp_cfg.get("retriever_k", base_cfg.get("retriever", {}).get("top_k", 8)))

    combos = _cartesian_product(sweeps)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 载入评测集
    dataset = load_eval_dataset(dataset_path) if dataset_path else []

    results: List[Dict[str, Any]] = []

    for i, combo in enumerate(combos, start=1):
        # 组合为配置树
        override: Dict[str, Any] = {"components": {}, "retriever": {}}
        # 解析简单维度
        if "retriever" in combo:
            override["retriever"]["type"] = combo["retriever"]
        if "embedder" in combo:
            override["components"]["embedding"] = {"type": combo["embedder"]}
        if "use_faiss" in combo:
            override["components"]["index"] = {"type": "faiss" if combo["use_faiss"] else "cosine"}
        if "reranker" in combo:
            override["components"]["reranker"] = {"type": combo["reranker"]}
        if "top_k" in combo:
            override["retriever"]["top_k"] = int(combo["top_k"])
        if "generator" in combo:
            override["components"]["generator"] = {"type": combo["generator"]}

        # 合并 fixed 与 base
        merged = _deep_update(base_cfg, {"data": {"input_dir": corpus_dir}})
        merged = _deep_update(merged, fixed)
        merged = _deep_update(merged, override)

        # 构建与索引
        pipe = build_pipeline_from_config(merged)
        t0 = time.time()
        n_chunks = pipe.index(corpus_dir)
        build_time = time.time() - t0

        # 评测
        scores = evaluate_pipeline(pipe, dataset, retriever_k=merged.get("retriever", {}).get("top_k", retr_k))

        row = {
            "id": i,
            "config": combo,
            "fixed": fixed,
            "n_chunks": n_chunks,
            "build_time_sec": round(build_time, 3),
            **scores,
        }
        results.append(row)
        print(f"[{i}/{len(combos)}] {combo} -> {json.dumps(scores)}")

    # 排行（按 nDCG@k 与 EM 综合）
    results_sorted = sorted(results, key=lambda r: (r.get("nDCG@k", 0.0), r.get("EM", 0.0)), reverse=True)

    # 写出
    (out_dir / "leaderboard.json").write_text(json.dumps(results_sorted, indent=2), encoding="utf-8")
    print(f"\nSaved leaderboard to: {out_dir / 'leaderboard.json'}")


if __name__ == "__main__":
    main()


