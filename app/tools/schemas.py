from pydantic import BaseModel, Field


class CalculatorInput(BaseModel):
    """Input required by the calculator tool."""

    expression: str = Field(
        min_length=1,
        description="Arithmetic expression to evaluate.",
    )