#!/usr/bin/env python3
"""Dev entry shim. The implementation lives in quant_research_mcp.server.

Kept so `uv run python server.py` and `import server` keep working; the
installed `quant-research-mcp` binary uses quant_research_mcp.server:main.
"""

from quant_research_mcp.server import main, mcp  # noqa: F401

if __name__ == "__main__":
    main()
