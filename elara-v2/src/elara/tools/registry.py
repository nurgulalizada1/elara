"""Tool registry: name -> Tool instance."""

from __future__ import annotations

from elara.providers.base import ToolSpec
from elara.tools.base import Tool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool '{tool.name}' already registered")
        if not tool.name.isidentifier():
            raise ValueError(f"invalid tool name '{tool.name}'")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return sorted(self._tools.values(), key=lambda t: t.name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def llm_specs(self, exclude: set[str] | None = None) -> list[ToolSpec]:
        exclude = exclude or set()
        return [t.spec() for t in self.all() if t.exposed_to_llm and t.name not in exclude]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
