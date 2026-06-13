# Security Policy

## Reporting a vulnerability

Please report security issues privately via GitHub's **Report a vulnerability**
button (Security tab → Advisories) on
<https://github.com/FlawlessByte/quant-research-mcp>, rather than opening a public
issue. We aim to acknowledge reports within a few days.

## Scope

This server is **read-only**: it fetches public market data, computes
indicators/decisions, and returns text. It places no orders, holds no
credentials, and persists nothing. There is no authentication layer because
there is nothing to authenticate to.

Relevant considerations:

- **Network egress.** The default `yfinance` provider makes outbound HTTPS
  requests to public market-data endpoints. No API keys are sent.
- **No code execution from inputs.** Tool inputs are validated by Pydantic
  models; nothing is `eval`'d or shelled out. `make check` asserts the codebase
  contains no `subprocess` or LLM/`claude -p` calls.
- **Supply chain.** Dependencies are pinned via `uv.lock`. Report any concern
  with a transitive dependency through the channel above.

## Not financial advice

This software is research and educational tooling only. Its output is not
investment advice and carries no warranty. Trading involves substantial risk of
loss; you are solely responsible for any capital you risk. See `LICENSE`.
