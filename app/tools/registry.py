from app.tools.base import BaseTool


class ToolRegistry:
    """Stores and provides access to MAI tools."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool by its unique name."""

        if tool.name in self._tools:
            raise ValueError(
                f"Tool '{tool.name}' is already registered."
            )

        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        """Return a registered tool."""

        try:
            return self._tools[name]
        except KeyError as exc:
            raise ValueError(
                f"Tool '{name}' is not registered."
            ) from exc

    def has(self, name: str) -> bool:
        """Check whether a tool exists."""

        return name in self._tools

    def list_tools(self) -> list[BaseTool]:
        """Return all registered tools."""

        return list(self._tools.values())