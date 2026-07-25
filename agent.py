"""Agent core: a tool-use loop with two interchangeable 'brains'.

- LLMBrain: real function calling against any OpenAI-compatible chat API
  (Kimi/Moonshot, OpenAI, a local vLLM...). Configured purely via env vars.
- HeuristicBrain: zero-key deterministic fallback that executes the same
  plan -> act -> observe loop with scripted decisions, so the whole pipeline
  (search -> dedupe -> summarize -> score -> digest) runs and is testable
  without an API key.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from tools import TOOL_SCHEMAS, Memory, dispatch

MAX_STEPS = 30

SYSTEM_PROMPT = """\
You are an autonomous literature-monitor agent for a researcher.

Your job each run:
1. Call get_research_profile to learn the researcher's interests.
2. Search arXiv with the configured queries (one search_arxiv call per query).
   Memory filters out already-processed papers automatically.
3. For each NEW paper: read the abstract, write a 2-3 sentence plain-language
   summary, score relevance 0-1 against the profile, and flag it if it meets
   the profile's flag_threshold. Record every judgement with remember_paper.
4. When done, call save_report exactly once with a markdown digest that leads
   with flagged papers (title, arXiv link, why it matters), then briefly lists
   the rest. Then stop.

Rules:
- Never skip remember_paper for a paper you evaluated.
- Score conservatively: only flag papers the researcher would plausibly read.
- Respect the profile's disinterests (e.g. financial sentiment is out of scope).
"""


# --------------------------------------------------------------------------- #
# Brain protocol + shared context
# --------------------------------------------------------------------------- #

@dataclass
class AgentContext:
    config: dict
    memory: Memory
    trace: list[dict[str, Any]] = field(default_factory=list)

    def record(self, kind: str, **data: Any) -> None:
        entry = {"kind": kind, **data}
        self.trace.append(entry)
        if kind == "tool_call":
            print(f"  [tool] {data['name']}({json.dumps(data['args'], ensure_ascii=False)[:140]})")
        elif kind == "note":
            print(f"  {data['text']}")


class Brain(Protocol):
    def run(self, ctx: AgentContext) -> dict[str, Any]:
        """Drive the loop; return the save_report result."""


# --------------------------------------------------------------------------- #
# LLM brain: OpenAI-compatible function calling, no SDK dependency
# --------------------------------------------------------------------------- #

class LLMBrain:
    def __init__(self) -> None:
        self.base_url = os.environ.get("LLM_BASE_URL", "https://api.moonshot.ai/v1").rstrip("/")
        self.api_key = os.environ.get("LLM_API_KEY") or os.environ.get("MOONSHOT_API_KEY") or ""
        self.model = os.environ.get("LLM_MODEL", "kimi-k2-0711-preview")
        if not self.api_key:
            raise RuntimeError(
                "LLM brain needs an API key: set LLM_API_KEY "
                "(+ optionally LLM_BASE_URL / LLM_MODEL).")

    def _chat(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "tools": TOOL_SCHEMAS,
            "temperature": 0.2,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read())
        return payload["choices"][0]["message"]

    def run(self, ctx: AgentContext) -> dict[str, Any]:
        queries_hint = "\n".join(f"- {q}" for q in ctx.config["arxiv"]["queries"])
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Run today's monitor. Use these queries:\n{queries_hint}"},
        ]
        report_result: dict[str, Any] | None = None

        for _ in range(MAX_STEPS):
            msg = self._chat(messages)
            messages.append(msg)
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                ctx.record("note", text=f"[llm] {msg.get('content', '')[:200]}")
                if report_result is not None:
                    break
                continue
            for call in tool_calls:
                name = call["function"]["name"]
                args = json.loads(call["function"]["arguments"] or "{}")
                ctx.record("tool_call", name=name, args=args)
                try:
                    result = dispatch(name, args, config=ctx.config, memory=ctx.memory)
                except Exception as exc:  # tool errors go back to the model, not the floor
                    result = {"error": str(exc)}
                if name == "save_report" and "error" not in result:
                    report_result = result
                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(result, ensure_ascii=False)[:12000],
                })

        if report_result is None:
            raise RuntimeError("LLM brain hit MAX_STEPS without saving a report")
        return report_result


# --------------------------------------------------------------------------- #
# Heuristic brain: deterministic zero-key fallback (demo + tests)
# --------------------------------------------------------------------------- #

_TOKEN_RE = re.compile(r"[a-z][a-z\-]+")


class HeuristicBrain:
    """Executes the same tool sequence with transparent, inspectable rules.

    Summarizer: first sentences of the abstract. Relevance: weighted keyword
    overlap with the profile, minus disinterest hits. Not a substitute for an
    LLM - but it proves out the loop, the memory semantics and the digest.
    """

    def run(self, ctx: AgentContext) -> dict[str, Any]:
        profile = dispatch("get_research_profile", {}, config=ctx.config, memory=ctx.memory)
        ctx.record("tool_call", name="get_research_profile", args={})

        # 1. gather new papers across all configured queries
        papers: dict[str, dict[str, Any]] = {}
        for query in ctx.config["arxiv"]["queries"]:
            ctx.record("tool_call", name="search_arxiv", args={"query": query})
            result = dispatch("search_arxiv", {"query": query},
                              config=ctx.config, memory=ctx.memory)
            ctx.record("note", text=f"       -> {len(result['new_papers'])} new "
                                    f"(of {result['total_hits']} hits)")
            for p in result["new_papers"]:
                papers[p["arxiv_id"]] = p

        # 2. evaluate each paper and remember the judgement
        flagged: list[str] = []
        evaluated: list[dict[str, Any]] = []
        threshold = profile["flag_threshold"]
        for p in papers.values():
            summary = self._summarize(p["abstract"])
            relevance, rationale = self._score(p, profile)
            is_flagged = relevance >= threshold
            ctx.record("tool_call", name="remember_paper",
                       args={"arxiv_id": p["arxiv_id"], "relevance": relevance,
                             "flagged": is_flagged})
            dispatch("remember_paper", {
                "arxiv_id": p["arxiv_id"], "title": p["title"], "summary": summary,
                "relevance": relevance, "flagged": is_flagged, "rationale": rationale,
            }, config=ctx.config, memory=ctx.memory)
            evaluated.append({**p, "summary": summary, "relevance": relevance,
                              "flagged": is_flagged, "rationale": rationale})
            if is_flagged:
                flagged.append(p["arxiv_id"])

        # 3. write the digest
        markdown = self._render_digest(evaluated, profile, ctx.config)
        ctx.record("tool_call", name="save_report", args={"flagged": len(flagged)})
        return dispatch("save_report", {
            "markdown": markdown, "flagged_ids": flagged,
            "papers_considered": len(evaluated),
        }, config=ctx.config, memory=ctx.memory)

    @staticmethod
    def _summarize(abstract: str) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", abstract.strip())
        return " ".join(sentences[:2])

    @staticmethod
    def _score(paper: dict[str, Any], profile: dict[str, Any]) -> tuple[float, str]:
        text = f"{paper['title']} {paper['abstract']}".lower()
        hits = [kw for kw in profile["keywords"] if kw.lower() in text]
        interest_hits = [i for i in profile["interests"]
                         if any(tok in text for tok in _TOKEN_RE.findall(i.lower())
                                if len(tok) > 4)]
        penalty = [d for d in profile.get("disinterests", []) if d.lower() in text]
        score = min(1.0, 0.18 * len(hits) + 0.12 * len(interest_hits))
        score = max(0.0, score - 0.25 * len(penalty))
        rationale = (f"matched keywords: {', '.join(hits[:4]) or 'none'}"
                     + (f"; excluded topic: {', '.join(penalty)}" if penalty else ""))
        return round(score, 2), rationale

    @staticmethod
    def _render_digest(evaluated: list[dict[str, Any]], profile: dict[str, Any],
                       config: dict) -> str:
        from datetime import datetime
        evaluated.sort(key=lambda p: p["relevance"], reverse=True)
        flagged = [p for p in evaluated if p["flagged"]]
        rest = [p for p in evaluated if not p["flagged"]]
        cap = config["output"]["max_papers_in_digest"]

        lines = [
            f"# arXiv monitor digest - {datetime.now():%Y-%m-%d}",
            "",
            f"**Profile:** {profile['name']} | **New papers evaluated:** "
            f"{len(evaluated)} | **Flagged for reading:** {len(flagged)}",
            "",
        ]
        if flagged:
            lines += ["## Flagged: read these", ""]
            for p in flagged[:cap]:
                lines += [
                    f"### {p['title']}",
                    f"- **Link:** {p['url']}  |  **Relevance:** {p['relevance']:.2f}  |  "
                    f"**Published:** {p['published'][:10]}",
                    f"- **Authors:** {', '.join(p['authors'][:5])}",
                    f"- **Summary:** {p['summary']}",
                    f"- **Why flagged:** {p['rationale']}",
                    "",
                ]
        else:
            lines += ["## No papers crossed the flag threshold today", ""]
        if rest:
            lines += ["## Also scanned (below threshold)", ""]
            for p in rest[: max(0, cap - len(flagged))]:
                lines.append(f"- [{p['title']}]({p['url']}) - relevance {p['relevance']:.2f}")
            lines.append("")
        lines += [
            "---",
            "*Generated by arxiv-sentiment-agent. Judgements are stored in "
            "`memory/seen_papers.json`; these papers will not be re-evaluated.*",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def run_agent(config: dict, brain: str = "auto") -> dict[str, Any]:
    memory = Memory()
    ctx = AgentContext(config=config, memory=memory)
    if brain == "auto":
        brain = "llm" if (os.environ.get("LLM_API_KEY") or os.environ.get("MOONSHOT_API_KEY")) else "heuristic"
    print(f"[agent] brain={brain} | seen papers in memory: {len(memory.seen)}")
    runner = LLMBrain() if brain == "llm" else HeuristicBrain()
    result = runner.run(ctx)
    print(f"[agent] done -> {result['digest_path']} "
          f"({result['flagged_count']} flagged)")
    return result
