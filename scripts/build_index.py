#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Build index via API")
    parser.add_argument("--host", default="http://localhost:8000", help="API base url")
    parser.add_argument("--source_dir", default=None, help="Directory with .txt files")
    parser.add_argument("--config_path", default=None, help="Optional YAML config path")
    args = parser.parse_args()

    url = f"{args.host.rstrip('/')}/v1/index"
    payload = {"source_dir": args.source_dir, "config_path": args.config_path}
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    print(json.dumps(resp.json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


