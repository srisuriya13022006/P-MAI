import ast
import operator
from typing import Any

from app.tools.base import BaseTool
from app.tools.result import ToolResult
from app.tools.schemas import CalculatorInput
from app.tools.calculator.parser import normalize_expression

class CalculatorTool(BaseTool):
    """Safely evaluates basic arithmetic expressions."""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return (
            "Performs basic arithmetic calculations using "
            "numbers and arithmetic operators."
        )

    @property
    def input_schema(self):
        return CalculatorInput

    def run(self, **kwargs: Any) -> ToolResult:
        try:
            arguments = self.input_schema.model_validate(kwargs)

            expression = normalize_expression(
                arguments.expression
            )

            if expression is None:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error="Could not understand the arithmetic expression.",
                )

            result = self._evaluate_expression(
                expression
            )

            return ToolResult(
                tool_name=self.name,
                success=True,
                data={
                    "result": result,
                    "expression": expression,
                },
            )

        except Exception as exc:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=str(exc),
            )

    def _evaluate_expression(
        self,
        expression: str,
    ) -> int | float:

        tree = ast.parse(
            expression,
            mode="eval",
        )

        return self._evaluate(tree.body)

    def _evaluate(self, node: ast.AST) -> int | float:

        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                raise ValueError(
                    "Boolean values are not allowed."
                )

            if not isinstance(
                node.value,
                (int, float),
            ):
                raise ValueError(
                    "Only numbers are allowed."
                )

            return node.value

        if isinstance(node, ast.UnaryOp):
            operand = self._evaluate(node.operand)

            if isinstance(node.op, ast.UAdd):
                return +operand

            if isinstance(node.op, ast.USub):
                return -operand

            raise ValueError(
                "Unsupported unary operator."
            )

        if isinstance(node, ast.BinOp):
            left = self._evaluate(node.left)
            right = self._evaluate(node.right)

            operations: dict[
                type[ast.operator],
                Any,
            ] = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.FloorDiv: operator.floordiv,
                ast.Mod: operator.mod,
                ast.Pow: operator.pow,
            }

            operation = operations.get(
                type(node.op)
            )

            if operation is None:
                raise ValueError(
                    "Unsupported arithmetic operator."
                )

            return operation(left, right)

        raise ValueError(
            f"Unsupported expression: "
            f"{type(node).__name__}"
        )