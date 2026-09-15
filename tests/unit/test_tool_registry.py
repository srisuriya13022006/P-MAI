from app.tools.calculator.tool import CalculatorTool
from app.tools.datetime.tool import DateTimeTool
from app.tools.registry import ToolRegistry


def main():
    registry = ToolRegistry()

    registry.register(CalculatorTool())
    registry.register(DateTimeTool())

    print(
        "Registered tools:",
        [tool.name for tool in registry.list_tools()]
    )

    assert registry.has("calculator")
    assert registry.has("datetime")

    calculator = registry.get("calculator")
    result = calculator.run(expression="45 * 87")

    print("Tool result:", result)

    assert result.success is True
    assert result.data["result"] == 3915

    datetime_tool = registry.get("datetime")
    datetime_result = datetime_tool.run()

    print("DateTime result:", datetime_result)

    assert datetime_result.success is True
    assert "date" in datetime_result.data
    assert "time" in datetime_result.data
    assert "day" in datetime_result.data

    print("Tool registry tests passed.")


if __name__ == "__main__":
    main()