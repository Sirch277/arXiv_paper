"""Smoke tests: run with `python test_agent.py` (no API key, no network beyond arXiv)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from agent import AgentContext, HeuristicBrain
from tools import Memory, dispatch

CONFIG = json.loads((Path(__file__).parent / "config.json").read_text())


def test_memory_dedupe() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mem = Memory(Path(tmp))
        assert not mem.has("2501.00001")
        dispatch("remember_paper", {
            "arxiv_id": "2501.00001", "title": "t", "summary": "s",
            "relevance": 0.5, "flagged": True, "rationale": "r",
        }, config=CONFIG, memory=mem)
        # a fresh Memory over the same dir must see the paper (persistence)
        mem2 = Memory(Path(tmp))
        assert mem2.has("2501.00001")
    print("ok: memory persists and dedupes")


def test_heuristic_loop_end_to_end() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mem = Memory(Path(tmp))
        ctx = AgentContext(config=CONFIG, memory=mem)
        result = HeuristicBrain().run(ctx)
        digest = Path(result["digest_path"])
        assert digest.exists(), "digest was not written"
        kinds = [e["kind"] for e in ctx.trace]
        assert kinds.count("tool_call") >= 3, "expected search/remember/save tool calls"
        assert len(mem.runs) == 1, "run was not logged"
    print("ok: heuristic brain completes the full plan-act-observe loop")


def test_tool_dispatch_unknown() -> None:
    try:
        dispatch("nope", {}, config=CONFIG, memory=Memory(Path(tempfile.mkdtemp())))
    except ValueError:
        print("ok: unknown tools are rejected")
    else:
        raise AssertionError("dispatch should reject unknown tools")


if __name__ == "__main__":
    test_memory_dedupe()
    test_heuristic_loop_end_to_end()
    test_tool_dispatch_unknown()
    print("all tests passed")
