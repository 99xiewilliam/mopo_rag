#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask question via API")
    parser.add_argument("--host", default="http://localhost:8000", help="API base url")
    parser.add_argument("--query", required=True, help="Question text")
    parser.add_argument("--top_k", type=int, default=5, help="Retriever top_k")
    args = parser.parse_args()

    url = f"{args.host.rstrip('/')}/v1/ask"
    payload = {"query": args.query, "top_k": args.top_k}
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    print(json.dumps(resp.json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


