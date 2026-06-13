"""Shared response formatting: JSON vs Markdown selection."""

import json
from enum import Enum


class ResponseFormat(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"


def to_json(payload) -> str:
    return json.dumps(payload, indent=2, default=str)


def kv_lines(d: dict, keys=None) -> list[str]:
    keys = keys or list(d.keys())
    return [f"- **{k}**: {d[k]}" for k in keys if k in d and d[k] is not None]
