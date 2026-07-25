"""Tool implementations exposed to the agent.

Each tool is a plain Python function. `TOOL_SCHEMAS` describes them in the
OpenAI function-calling format so the LLM brain can call them natively; the
heuristic brain calls the same functions through `dispatch`.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ARXIV_API = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

ROOT = Path(__file__).resolve().parent
MEMORY_DIR = ROOT / "memory"


# --------------------------------------------------------------------------- #
# Memory (persistent state shared across runs)
# --------------------------------------------------------------------------- #

class Memory:
    """JSON-file-backed memory: papers already processed + run history.

    This is what makes the agent stateful across runs: it never summarizes the
    same paper twice, and it accumulates a relevance track record you can
    inspect or feed back into the research profile.
    """

    def __init__(self, memory_dir: Path = MEMORY_DIR):
        self.dir = Path(memory_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.seen_path = self.dir / "seen_papers.json"
        self.runs_path = self.dir / "run_log.json"
        self.seen: dict[str, Any] = self._load(self.seen_path, {})
        self.runs: list[dict[str, Any]] = self._load(self.runs_path, [])

    @staticmethod
    def _load(path: Path, default: Any) -> Any:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return default

    def has(self, arxiv_id: str) -> bool:
        return arxiv_id in self.seen

    def remember_paper(self, arxiv_id: str, *, title: str, summary: str,
                       relevance: float, flagged: bool, rationale: str) -> None:
        self.seen[arxiv_id] = {
            "title": title,
            "summary": summary,
            "relevance": relevance,
            "flagged": flagged,
            "rationale": rationale,
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }
        self.seen_path.write_text(
            json.dumps(self.seen, indent=2, ensure_ascii=False), encoding="utf-8")

    def log_run(self, record: dict[str, Any]) -> None:
        self.runs.append(record)
        self.runs_path.write_text(
            json.dumps(self.runs, indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- #
# arXiv fetching
# --------------------------------------------------------------------------- #

def _fetch_arxiv(query: str, max_results: int) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode({
        "search_query": query,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": max_results,
    })
    req = urllib.request.Request(
        f"{ARXIV_API}?{params}", headers={"User-Agent": "arxiv-sentiment-agent/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        root = ET.fromstring(resp.read())

    papers = []
    for entry in root.findall(f"{ATOM}entry"):
        raw_id = entry.findtext(f"{ATOM}id", "")
        arxiv_id = raw_id.rsplit("/", 1)[-1].split("v")[0]
        published = entry.findtext(f"{ATOM}published", "")
        abstract = re.sub(r"\s+", " ", entry.findtext(f"{ATOM}summary", "")).strip()
        categories = [c.get("term", "") for c in entry.findall(f"{ATOM}category")]
        papers.append({
            "arxiv_id": arxiv_id,
            "title": re.sub(r"\s+", " ", entry.findtext(f"{ATOM}title", "")).strip(),
            "authors": [a.findtext(f"{ATOM}name", "") for a in entry.findall(f"{ATOM}author")],
            "abstract": abstract,
            "published": published,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "categories": categories,
        })
    return papers


# --------------------------------------------------------------------------- #
# Tool functions (what the agent actually calls)
# --------------------------------------------------------------------------- #

def search_arxiv(config: dict, memory: Memory, query: str,
                 max_results: int | None = None) -> dict[str, Any]:
    """Search arXiv and return only papers not already in memory."""
    max_results = max_results or config["arxiv"]["max_results_per_query"]
    lookback = config["arxiv"]["lookback_days"]
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback)

    papers = _fetch_arxiv(query, max_results)
    fresh = []
    for p in papers:
        try:
            published = datetime.fromisoformat(p["published"].replace("Z", "+00:00"))
        except ValueError:
            continue
        if published >= cutoff and not memory.has(p["arxiv_id"]):
            fresh.append(p)
    return {
        "query": query,
        "total_hits": len(papers),
        "new_papers": fresh,
        "note": "already-processed and out-of-window papers were filtered out by memory",
    }


def get_research_profile(config: dict) -> dict[str, Any]:
    """Return the user's research interests, keywords and flag threshold."""
    return config["profile"]


def remember_paper(memory: Memory, arxiv_id: str, title: str, summary: str,
                   relevance: float, flagged: bool, rationale: str) -> dict[str, Any]:
    """Persist the agent's judgement about a paper so it is never re-processed."""
    memory.remember_paper(arxiv_id, title=title, summary=summary,
                          relevance=relevance, flagged=flagged, rationale=rationale)
    return {"stored": arxiv_id, "flagged": flagged}


def save_report(config: dict, memory: Memory, markdown: str, flagged_ids: list[str],
                papers_considered: int) -> dict[str, Any]:
    """Write the digest to disk and log the run."""
    digest_dir = ROOT / config["output"]["digest_dir"]
    digest_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d")
    path = digest_dir / f"digest_{stamp}.md"
    path.write_text(markdown, encoding="utf-8")
    memory.log_run({
        "at": datetime.now(timezone.utc).isoformat(),
        "papers_considered": papers_considered,
        "flagged": flagged_ids,
        "digest": str(path.relative_to(ROOT)),
    })
    return {"digest_path": str(path), "flagged_count": len(flagged_ids)}


# --------------------------------------------------------------------------- #
# Function-calling glue
# --------------------------------------------------------------------------- #

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_arxiv",
            "description": (
                "Search arXiv for recent papers. Returns only papers published "
                "within the configured lookback window that the agent has NOT "
                "processed before (memory dedupe is automatic)."),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "arXiv API query, e.g. 'all:\"emotion recognition in conversation\"'"},
                    "max_results": {"type": "integer", "description": "Max results to fetch (default from config)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_research_profile",
            "description": "Get the user's research interests, keywords and relevance flag threshold.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember_paper",
            "description": (
                "Store your summary, relevance score (0-1), flag decision and "
                "rationale for a paper in long-term memory. Call once per paper "
                "you evaluate."),
            "parameters": {
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string"},
                    "title": {"type": "string"},
                    "summary": {"type": "string", "description": "2-3 sentence plain-language summary"},
                    "relevance": {"type": "number", "description": "0.0-1.0 relevance to the research profile"},
                    "flagged": {"type": "boolean", "description": "true if the user should read this paper"},
                    "rationale": {"type": "string", "description": "one sentence explaining the score"},
                },
                "required": ["arxiv_id", "title", "summary", "relevance", "flagged", "rationale"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_report",
            "description": (
                "Save the final markdown digest and finish the run. Call exactly "
                "once, after all papers have been remembered."),
            "parameters": {
                "type": "object",
                "properties": {
                    "markdown": {"type": "string", "description": "Full markdown digest"},
                    "flagged_ids": {"type": "array", "items": {"type": "string"}},
                    "papers_considered": {"type": "integer"},
                },
                "required": ["markdown", "flagged_ids", "papers_considered"],
            },
        },
    },
]


def dispatch(name: str, args: dict[str, Any], *, config: dict,
             memory: Memory) -> dict[str, Any]:
    """Route a tool call from either brain to the right function."""
    if name == "search_arxiv":
        return search_arxiv(config, memory, **args)
    if name == "get_research_profile":
        return get_research_profile(config)
    if name == "remember_paper":
        return remember_paper(memory, **args)
    if name == "save_report":
        return save_report(config, memory, **args)
    raise ValueError(f"unknown tool: {name}")
