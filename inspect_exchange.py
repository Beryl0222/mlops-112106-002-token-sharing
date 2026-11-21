"""检查交换资料的形状和索引。"""

import argparse
import json
from pathlib import Path

import numpy as np


def inspect(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    validate(data)
    return {
        "资料": path.name,
        "字段": sorted(data),
        "字节": path.stat().st_size,
        "数值环境": np.__version__,
    }


def validate(data: dict) -> None:
    members = {m["id"]: m for m in data["members"]}
    for member in members.values():
        assert member["key"] in data["owners"]["k"]
        assert member["value"] in data["owners"]["v"]
    queries = {q["id"]: q for q in data["queries"]}
    assert all(set(q["visible"]) <= members.keys() for q in queries.values())
    assert all(s["query"] in queries for s in data["heads"])


def main() -> None:
    parser = argparse.ArgumentParser(description="检查本地交换资料")
    parser.add_argument("scene", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.scene), ensure_ascii=False))


if __name__ == "__main__":
    main()
