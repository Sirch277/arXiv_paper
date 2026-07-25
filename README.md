# arXiv ERC/Sentiment Monitor Agent

A small **autonomous tool-use agent** that monitors arXiv for new papers on
Emotion Recognition in Conversation (ERC) and sentiment analysis, summarizes
them, scores their relevance to *your* research profile, flags the ones worth
reading, and writes a daily digest. It remembers every paper it has ever
processed, so nothing is summarized twice.

Built to demonstrate the three core agentic patterns on a real, useful task:

| Pattern | Where it lives |
|---|---|
| **Function calling** | `tools.py` — 4 tools with JSON schemas, dispatched by name |
| **Planning loop** | `agent.py` — plan → call tool → observe → repeat, until `save_report` |
| **Memory** | `tools.py:Memory` — persistent JSON state (`memory/seen_papers.json`, `memory/run_log.json`) |

**Zero dependencies** — standard library only (Python 3.10+).

## Architecture

```
run.py ──► agent.py ─────────────┐
           │  brain: LLMBrain     │  (OpenAI-compatible function calling,
           │        or            │   Kimi/Moonshot/OpenAI/local vLLM)
           │        HeuristicBrain│  (zero-key deterministic fallback)
           ▼                      │
           tools.py ◄─────────────┘
           │ search_arxiv        → arXiv API (Atom), lookback filter
           │ get_research_profile→ config.json
           │ remember_paper      → memory/seen_papers.json   (long-term memory)
           │ save_report         → digests/digest_YYYY-MM-DD.md + run log
           ▼
           memory/  (persisted across runs: dedupe + audit trail)
```

The **brains are interchangeable**: both drive the exact same tools through the
same `dispatch` function. `LLMBrain` lets the model decide what to call and
writes the summaries itself; `HeuristicBrain` is a scripted policy (keyword
scoring + first-sentences summarizer) so the full pipeline runs and is testable
with no API key. Swap in your key and the agent gets real LLM judgement with no
other code changes.

## Quick start

```bash
cd arxiv-sentiment-agent

# 1. Works out of the box (heuristic brain, no key needed)
python run.py

# 2. Real LLM function calling — any OpenAI-compatible endpoint
export LLM_API_KEY=sk-...                      # required
export LLM_BASE_URL=https://api.moonshot.ai/v1 # optional (this is the default)
export LLM_MODEL=kimi-k2-0711-preview          # optional
python run.py --brain llm

# 3. Tests (no key needed)
python test_agent.py
```

Each run prints its tool-call trace and writes:

- `digests/digest_YYYY-MM-DD.md` — flagged papers first, with summaries,
  relevance scores and rationale; then everything else scanned
- `memory/seen_papers.json` — every judgement ever made (dedupe source)
- `memory/run_log.json` — audit trail of all runs

**Make it yours:** edit `config.json` — `interests`, `keywords`,
`disinterests`, `flag_threshold`, the arXiv queries and the lookback window.

## How a run works (the loop)

1. `get_research_profile` — the agent grounds itself in your interests.
2. One `search_arxiv` call per configured query. The tool itself filters out
   papers older than `lookback_days` **and** anything already in memory —
   that's what makes runs idempotent.
3. For each new paper: summarize, score relevance 0–1, flag if
   `relevance ≥ flag_threshold`, then `remember_paper` (mandatory — the
   judgement is persisted before moving on).
4. `save_report` exactly once → digest + run log → loop ends.

Verified behavior: first live run evaluated **7 new papers** and flagged 6
(e.g. *EII-SCL: Emotional Inertia for MERC*, 0.90; *SpEmoC benchmark*, 0.96);
the immediate second run found **0 new papers** — memory dedupe working.

## Scheduling (cron)

Run it every morning at 08:00:

```cron
0 8 * * *  cd /path/to/arxiv-sentiment-agent && /usr/bin/python3 run.py >> monitor.log 2>&1
```

Weekly instead: `0 8 * * 1 ...`. Keep `lookback_days` ≥ the schedule gap so
nothing falls through the cracks.

## Files

```
config.json   research profile, queries, thresholds   ← edit this
tools.py      tool implementations + function schemas + Memory
agent.py      planning loop, LLMBrain, HeuristicBrain
run.py        CLI entry point
test_agent.py smoke tests (memory persistence, full loop, dispatch)
digests/      generated markdown digests
memory/       generated long-term memory (seen papers + run log)
```

## Extending it

- **Email/Slack delivery**: add a `send_digest` tool and one line to the
  system prompt — the loop picks it up automatically.
- **PDF deep-reads**: add a `fetch_pdf` tool (arXiv export endpoint) for
  flagged papers only.
- **Richer memory**: swap `Memory`'s JSON files for SQLite; the tool
  signatures don't change.
