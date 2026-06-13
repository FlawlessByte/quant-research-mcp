## Summary

<!-- What does this PR do? One paragraph. -->

## Type

- [ ] Bug fix
- [ ] New trading method / paper
- [ ] Enhancement to existing method or tool
- [ ] Infrastructure / CI / release
- [ ] Docs only

## Checklist

- [ ] `make check` passes (40+ tests green, ruff clean, no subprocess calls)
- [ ] `make smoke` lists expected tool count
- [ ] New method: registered in `methods/__init__.py`, paper cited in `TradingMethod`
- [ ] New tool: added to tools table in README
- [ ] Deterministic logic change: evals updated in `evals/evaluation.xml`
- [ ] No hardcoded paths, no secrets, no LLM calls inside the server

## Test plan

<!-- How did you test this? What edge cases did you check? -->

## Related issues

<!-- Closes #... -->

---

> **Not financial advice.** This project is educational and research tooling only.
