#!/usr/bin/env python3
"""CLI entry point.

Usage:
    python run.py                 # auto brain (LLM if a key is set, else heuristic)
    python run.py --brain llm     # force the LLM function-calling brain
    python run.py --brain heuristic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent import run_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="arXiv ERC/sentiment monitor agent")
    parser.add_argument("--brain", choices=["auto", "llm", "heuristic"], default="auto")
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.json"))
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = run_agent(config, brain=args.brain)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
