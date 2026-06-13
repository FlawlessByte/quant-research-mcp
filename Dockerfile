# Containerised stdio MCP server. Run with -i so the server can read stdin:
#   docker run -i --rm ghcr.io/flawlessbyte/quant-research-mcp
FROM python:3.12-slim

# uv for fast, locked installs.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Install dependencies first (cached layer), then the project.
COPY pyproject.toml uv.lock README.md ./
COPY quant_research_mcp ./quant_research_mcp
COPY server.py ./
RUN uv sync --frozen --no-dev

# stdio transport: the entry point talks MCP over stdin/stdout.
ENTRYPOINT ["uv", "run", "--no-dev", "quant-research-mcp"]
