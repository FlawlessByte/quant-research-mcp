# quant_research_mcp

<!-- mcp-name: io.github.FlawlessByte/quant-research-mcp -->

[![CI](https://github.com/FlawlessByte/quant-research-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/FlawlessByte/quant-research-mcp/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![MCP](https://img.shields.io/badge/MCP-server-purple.svg)](https://modelcontextprotocol.io/)

An MCP server exposing a **registry of paper-backed quantitative trading methods**
plus a **deterministic, no-LLM decision helper**. The server supplies reproducible
math (screening, indicators, regime detection, method signals, scored entry
decisions); the calling agent supplies judgement (e.g. reading headlines into a
sentiment signal). No `claude -p`, no subprocess, no LLM inside the server.

> ⚠️ **Not financial advice.** This is read-only **research and educational
> tooling**. It performs quantitative computation only and **places no orders**.
> Nothing it outputs is investment advice or a recommendation. Market data may be
> delayed or wrong, backtests are not forecasts, and trading carries substantial
> risk of loss. **You alone are responsible for any capital you risk.** Provided
> "as is" without warranty — see [`LICENSE`](LICENSE).

## Why this exists

Repackages a working day-trading pipeline as composable MCP tools, with two goals:

1. **Extensible to future papers.** Each strategy is a `TradingMethod` registered in
   a registry. Adding a new paper = drop one module, call `register(...)`, import it.
   Nothing else changes.
2. **Decision-making is scientific, not generative.** The old pipeline gated entries
   with an LLM call. Here that is replaced by `quant_score_decision` — a pure
   function combining reward:risk, regime strength, volume, RSI positioning and
   ATR-normalised stop quality into an auditable score. Same inputs → same output.

## Tools (13)

| Tool | Network | Purpose |
|---|---|---|
| `quant_list_methods` | no | List registered methods (key, paper, timeframe, regimes). |
| `quant_describe_method` | no | Full detail + citation for one method. |
| `quant_screen_universe` | yes | Rank tickers by gap / rel-volume / ATR%. |
| `quant_compute_indicators` | yes | EMA9/20, RSI, ATR, VWAP, Hurst for a ticker. |
| `quant_detect_regime` | yes | Hurst → TRENDING / MEAN_REVERTING / RANDOM_WALK. |
| `quant_analyze_setup` | yes | Run a per-ticker method → signal + entry/stop/target (+ HTF context). |
| `quant_analyze_universe` | yes | Run a universe method: momentum ranking, pairs spread. |
| `quant_backtest_method` | yes | **Validate a method**: replay its own analyze() over history → win rate, expectancy, drawdown, IS/OOS halves. |
| `quant_check_events` | yes | Next earnings (+days), ex-dividend — binary-event risk. |
| `quant_portfolio_risk` | yes | Stateless heat / correlation / concentration check + candidate verdict. |
| `quant_market_context` | yes | SPY/QQQ/IWM, VIX, 11 sectors ranked, risk-on/off breadth. |
| `quant_score_decision` | no | **Decision helper**: setup (+ sentiment, earnings, heat) → score, verdict, size, timeframe-aware execution plan. Deterministic. |
| `quant_fetch_news` | yes | Recent headlines (data only; agent forms sentiment). |

All tools are read-only and support `response_format: markdown | json`.

## Workflows

**Day trade**
```
quant_market_context                  # tape read: risk-on/off, sectors
  → quant_screen_universe             # find movers
  → quant_analyze_setup               # hurst_regime_orb on the top name
  → quant_check_events + quant_fetch_news   # binary risk + agent sentiment
  → quant_portfolio_risk              # heat/correlation vs your open book
  → quant_score_decision(setup, news_sentiment, days_to_earnings,
                         portfolio_heat_pct)
```

**Swing (days)** — `rsi2_reversion`, `pairs_cointegration` via
`quant_analyze_setup` / `quant_analyze_universe`; earnings veto matters most
here (`quant_check_events` → `days_to_earnings`).

**Position (weeks+)** — `donchian_trend` per ticker, `xs_momentum` over the
universe; re-rank monthly, weekly stop review per the execution plan.

**Before trusting any method**: `quant_backtest_method` on your tickers and
period — it replays the method's own signal logic with costs, and reports
in-sample vs out-of-sample halves so you can see decay.

The agent writes any narrative; the server guarantees the numbers.

## The decision helper (`quant_score_decision`)

Pure function. Composite score (weights in `config.DECISION_WEIGHTS`):

- **reward_risk** — realised R:R vs the target multiple.
- **regime_strength** — `|Hurst − 0.5|` scaled (distance from random walk).
- **volume_confirmation** — relative volume vs the floor.
- **momentum_position** — RSI in a healthy band for the direction (not exhausted).
- **stop_quality** — stop distance normalised by daily ATR (noise-tight stops penalised).

Optional `news_sentiment` (the **agent** derives this) boosts an aligned score or
**vetoes** a contradicted one at confidence ≥ 0.5. Output includes the per-factor
breakdown, fixed-fractional `position_size` (with a haircut for noise-tight stops),
and a mechanically derived `execution_plan` (entry trigger, order type, stop ladder,
profit taking, time stop, abort conditions).

## Bundled methods (5)

| Key | Timeframe | Paper |
|---|---|---|
| `hurst_regime_orb` | intraday | [arXiv:2205.11122](https://arxiv.org/pdf/2205.11122) — Hurst regime → ORB / VWAP fade |
| `rsi2_reversion` | swing | Connors & Alvarez 2009 — RSI(2) pullback above the 200d SMA |
| `pairs_cointegration` | swing | [Gatev et al. 2006](https://doi.org/10.1093/rfs/hhj020) — Engle-Granger spread z-score |
| `donchian_trend` | position | Faith, Turtle Rules; [Moskowitz et al. 2012](https://www.sciencedirect.com/science/article/pii/S0304405X11002613) — 55d breakout, 20d/ATR trail |
| `xs_momentum` | position | [Jegadeesh & Titman 1993](https://doi.org/10.1111/j.1540-6261.1993.tb04702.x) — 12-1 cross-sectional momentum, top-N book |

## Data providers

Default is **yfinance** (free, ~15-min delayed). The data layer sits behind a
provider interface (`quant_research_mcp/providers/`): implement the
`DataProvider` protocol for Alpaca/Polygon/IBKR, register the factory, then run
with `QUANT_DATA_PROVIDER=<name>`. All calls are TTL-cached in-process (daily
15 min, intraday 60 s, news 5 min, events/sector 1 h).

## Adding a future paper

```python
# quant_research_mcp/methods/my_paper.py
from . import register
from .base import TradingMethod, TradeSetup

def analyze(daily, session, context) -> TradeSetup:
    ...  # compute signal, entry, stop, target

register(TradingMethod(
    key="my_paper_method",
    name="My Method",
    paper="Author et al., Title",
    paper_url="https://arxiv.org/abs/...",
    regime_applicability="TRENDING",
    description="One paragraph on the mechanics.",
    analyze=analyze,
))
```

Then add `from . import my_paper` to `methods/__init__.py`. It now appears in
`quant_list_methods` and is runnable via `quant_analyze_setup(method_key=...)`.

## Quick start (Makefile)

```bash
make install      # create venv + install deps (uv sync)
make smoke        # import server, list the 13 tools (no network)
make run          # run the server over stdio
make help         # list every target
```

### Install as a shell binary

```bash
make install-cli              # installs `quant-research-mcp` onto your PATH (uv tool)
quant-research-mcp            # runs the server over stdio from anywhere
```

If the command isn't found after install, run `uv tool update-shell` and restart
your shell (uv's tool bin dir, usually `~/.local/bin`, must be on PATH).

| Target | What it does |
|---|---|
| `make install` | `uv sync` — venv + dependencies |
| `make install-cli` | install the `quant-research-mcp` binary onto your PATH |
| `make uninstall-cli` | remove the binary |
| `make dev` | install + ruff for linting |
| `make run` | run the MCP server (stdio) |
| `make smoke` | import + list tools, no network |
| `make test` | run the pytest suite (no network) |
| `make inspect` | launch the MCP Inspector |
| `make check` | tests + lint + assert no LLM/subprocess calls |
| `make register` | print the stdio client-config JSON |
| `make claude-add` | register the local clone with Claude Code (user scope) |
| `make claude-remove` | remove from Claude Code |
| `make build` | build sdist + wheel into `dist/` |
| `make docker-build` | build the Docker image locally |
| `make mcpb` | pack a Claude Desktop `.mcpb` bundle |
| `make clean` | drop caches, venv and build artifacts |

## Install

Every method runs the same stdio server — pick whichever fits your setup.

| Method | Command | Needs |
|---|---|---|
| **uvx from GitHub** (no clone) | `uvx --from git+https://github.com/FlawlessByte/quant-research-mcp quant-research-mcp` | [uv](https://docs.astral.sh/uv/) |
| **PyPI** | `uvx quant-research-mcp` · `pipx install quant-research-mcp` | uv / pipx |
| **Docker** | `docker run -i --rm ghcr.io/flawlessbyte/quant-research-mcp` | Docker |
| **From source** | `git clone … && cd quant-research-mcp && make install` | git + uv |
| **Claude Desktop** | install the `.mcpb` from the [latest release](https://github.com/FlawlessByte/quant-research-mcp/releases) | Claude Desktop |

> PyPI and Docker images are published on tagged releases; until the first
> release, use the **uvx from GitHub** or **from source** rows.

## Use with Claude Code

Fastest — runs straight from GitHub, no clone:

```bash
claude mcp add quant_research -- \
  uvx --from git+https://github.com/FlawlessByte/quant-research-mcp quant-research-mcp
```

After a PyPI release this shortens to:

```bash
claude mcp add quant_research -- uvx quant-research-mcp
```

Containerised:

```bash
claude mcp add quant_research -- docker run -i --rm ghcr.io/flawlessbyte/quant-research-mcp
```

From a local clone (development): `make claude-add` registers this checkout at
user scope; `make register` prints a paste-ready `.mcp.json` block.

**Verify** inside Claude Code:

```
/mcp                       # should list 'quant_research' as connected
```

Then ask e.g. *"screen the universe and analyze the top name with the hurst
method, then score the entry decision."* It will call `quant_screen_universe`
→ `quant_analyze_setup` → `quant_score_decision`.

## From source (development)

```bash
git clone https://github.com/FlawlessByte/quant-research-mcp
cd quant-research-mcp
make install      # uv sync — venv + dependencies
make smoke        # list the 13 tools (no network)
make test         # 40 offline tests
```

Optional: `make install-cli` puts a `quant-research-mcp` binary on your PATH
(via `uv tool`). If it isn't found afterwards, run `uv tool update-shell` and
restart your shell. Inspect tool schemas with `make inspect`.

## Releasing (maintainers)

CI runs ruff + the 40 offline tests on every push/PR (Python 3.12 & 3.13).
Cutting a release is tag-driven:

```bash
# bump version in pyproject.toml + server.json + manifest.json, commit, then:
git tag v0.1.0 && git push origin v0.1.0
```

That fires two workflows:

- **`release.yml`** → `uv build`, publish to **PyPI** via Trusted Publishing
  (OIDC, no stored token), and attach the wheel/sdist + a `.mcpb` bundle to the
  GitHub Release.
- **`docker.yml`** → build and push `ghcr.io/flawlessbyte/quant-research-mcp`.

One-time setup:

1. **PyPI:** create the project and add a Trusted Publisher (owner `FlawlessByte`,
   repo `quant-research-mcp`, workflow `release.yml`, environment `pypi`).
2. **GHCR:** after the first push, set the package visibility to public.
3. **MCP registry:** after the first PyPI release, list it with the
   [`mcp-publisher`](https://github.com/modelcontextprotocol/registry) CLI —
   `mcp-publisher login github` then `mcp-publisher publish` (uses `server.json`;
   GitHub login proves ownership of the `io.github.FlawlessByte/…` namespace).

> The Claude Desktop `.mcpb` invokes `uvx` under the hood, so a one-click install
> still requires [uv](https://docs.astral.sh/uv/) on the machine — bundling
> pandas/scipy/statsmodels wheels directly would be large and platform-specific.

## Limitations (what a serious trader still needs elsewhere)

- **Delayed data** until you wire a real-time provider key (interface is ready;
  yfinance is ~15-min delayed and its news feed is thin).
- **No macro calendar** (FOMC/CPI/NFP) — no reliable free feed; pass your own
  judgement through `news_sentiment` / `news_confidence`.
- **No options data** (IV, term structure, gamma levels), no short
  interest/float, no Level 2 — next frontier.
- **Backtests are parameter validation, not forecasts**: yfinance history has
  survivorship bias, costs are estimates, intraday replay is capped at ~60 days
  of 5m bars by the provider.
- **No persistence** by design — the server stores nothing; supply open
  positions per call (`quant_portfolio_risk`). A trade journal is a planned
  opt-in module.
