# Contributing

Thanks for your interest. This project packages paper-backed quantitative
trading methods as MCP tools, plus a deterministic decision helper. Two hard
rules keep it trustworthy:

1. **No LLM, no subprocess, no network inside the decision logic.** The server
   supplies reproducible math; the calling agent supplies judgement. `make check`
   fails the build if it finds `subprocess` or `claude -p`.
2. **Read-only.** Tools place no orders and persist nothing. Open positions are
   passed in per call.

## Dev setup

```bash
git clone https://github.com/FlawlessByte/quant-research-mcp
cd quant-research-mcp
make dev          # uv sync + ruff
make check        # ruff + 40 offline tests + no-subprocess assertion
make smoke        # list the 13 tools (no network)
```

Tests must stay **fully offline** — network is monkeypatched in `tests/`. New
tests follow the same pattern (see `tests/conftest.py`).

## Adding a paper-backed method

The registry is built for this — one new module, one import line, nothing else
changes:

```python
# quant_research_mcp/methods/my_paper.py
from . import register
from .base import TradingMethod, TradeSetup, DataRequirements

def analyze(daily, session, context) -> TradeSetup:
    ...  # compute signal, entry, stop, target

register(TradingMethod(
    key="my_paper_method",
    name="My Method",
    paper="Author et al., Title (Year)",
    paper_url="https://arxiv.org/abs/...",
    regime_applicability="TRENDING",
    timeframe="swing",                 # intraday | swing | position
    data=DataRequirements(daily_period="2y"),
    description="One paragraph on the mechanics.",
    analyze=analyze,
))
```

Then add `from . import my_paper` to `quant_research_mcp/methods/__init__.py`.
It now appears in `quant_list_methods` and runs via `quant_analyze_setup`. Add a
trigger-logic test in `tests/test_methods.py` on synthetic data, and — if the
method makes entry decisions — a deterministic Q&A in `evals/evaluation.xml`.

## Before you open a PR

- `make check` is green (ruff + tests + no-subprocess).
- New behaviour has a test; new entry logic has an eval.
- Commits use [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
- Cite the paper for any new method.

## Not financial advice

This is research tooling. Do not add anything that places orders, gives
investment advice, or implies a guaranteed outcome.
