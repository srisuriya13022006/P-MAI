from app.tools.calculator.tool import CalculatorTool


def main() -> None:
    calculator = CalculatorTool()

    test_cases = [
        ("45 * 87", 3915),
        ("100 / 4", 25.0),
        ("10 + 5 * 2", 20),
        ("2 ** 10", 1024),
        ("20 % 3", 2),
         ("45 * 87", 3915),
    ("45 multiplied by 87", 3915),
    ("45 times 87", 3915),
    ("45 x 87", 3915),
    ("100 divided by 4", 25.0),
    ("10 plus 5", 15),
    ("20 minus 7", 13),
    ]

    for expression, expected in test_cases:

        result = calculator.run(
            expression=expression
        )

        print(
            f"{expression} -> "
            f"{result.data} "
            f"(expected: {expected})"
        )

        assert result.success is True
        assert result.data["result"] == expected

    print("\nCalculator tests passed.")


if __name__ == "__main__":
    main()