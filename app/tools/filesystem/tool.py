import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.tools.base import BaseTool
from app.tools.result import ToolResult


class FileSearchInput(BaseModel):
    query: str = Field(min_length=1, description="The question or keywords to search for in project files.")


class FileSearchTool(BaseTool):
    """Lightweight document retrieval for project files and markdown notes."""

    def __init__(self, root_dir: str | None = None) -> None:
        self.root_dir = Path(root_dir or ".").resolve()

    @property
    def name(self) -> str:
        return "file_search"

    @property
    def description(self) -> str:
        return "Search local project files and documents for relevant text."

    @property
    def input_schema(self) -> type[BaseModel]:
        return FileSearchInput

    def run(self, **kwargs: Any) -> ToolResult:
        try:
            arguments = self.input_schema.model_validate(kwargs)
            query_tokens = self._tokens(arguments.query)
            matches: list[dict[str, str]] = []

            for file_path in self.root_dir.rglob("*"):
                if not file_path.is_file() or not self._is_text_file(file_path):
                    continue

                try:
                    text = file_path.read_text(encoding="utf-8")
                except Exception:
                    continue

                score = self._score_text(text, query_tokens)
                if score <= 0:
                    continue

                snippet = self._extract_snippet(text, query_tokens)
                title = self._extract_title(file_path, text)
                matches.append(
                    {
                        "title": title,
                        "path": str(file_path.relative_to(self.root_dir)),
                        "score": str(score),
                        "snippet": snippet,
                    }
                )

            matches.sort(key=lambda item: float(item["score"]), reverse=True)
            if not matches:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    data={"query": arguments.query, "matches": []},
                    error="No relevant document matches were found.",
                )

            return ToolResult(
                tool_name=self.name,
                success=True,
                data={
                    "query": arguments.query,
                    "matches": matches[:5],
                },
            )
        except Exception as exc:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=str(exc),
            )

    def _is_text_file(self, path: Path) -> bool:
        return path.suffix.lower() in {".txt", ".md", ".rst", ".py", ".json", ".yaml", ".yml"}

    def _score_text(self, text: str, query_tokens: set[str]) -> float:
        tokens = self._tokens(text)
        overlap = len(query_tokens & tokens)
        if overlap == 0:
            return 0.0
        return float(overlap + len(query_tokens.intersection(tokens)) * 0.25)

    def _extract_snippet(self, text: str, query_tokens: set[str]) -> str:
        haystack = text.lower()
        for token in sorted(query_tokens, key=len, reverse=True):
            index = haystack.find(token.lower())
            if index != -1:
                start = max(0, index - 80)
                end = min(len(text), index + 160)
                snippet = text[start:end].replace("\n", " ")
                return snippet.strip()
        return text[:200].replace("\n", " ").strip()

    @staticmethod
    def _extract_title(file_path: Path, text: str) -> str:
        heading_match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
        if heading_match:
            title = heading_match.group(1).strip()
            if title:
                return title
        return file_path.name

    @staticmethod
    def _tokens(value: str) -> set[str]:
        return {token.lower() for token in re.findall(r"[a-z0-9]+", value.lower())}
