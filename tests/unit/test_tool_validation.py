from app.tools.calculator.tool import CalculatorTool


def main() -> None:
    calculator = CalculatorTool()

    # Missing expression
    result = calculator.run()

    assert result.success is False

    # Empty expression
    result = calculator.run(
        expression=""
    )

    assert result.success is False

    # Unsupported code
    result = calculator.run(
        expression="__import__('os').system('dir')"
    )

    assert result.success is False

    print("Tool validation tests passed.")


if __name__ == "__main__":
    main()