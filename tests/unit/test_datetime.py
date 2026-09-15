from app.tools.datetime.tool import DateTimeTool
from datetime import datetime
from zoneinfo import ZoneInfo


def main():
    tool = DateTimeTool()

    result = tool.run()

    print("Tool result:", result)

    assert result.success is True
    assert result.tool_name == "datetime"

    assert "date" in result.data
    assert "time" in result.data
    assert "day" in result.data

    fixed_now = datetime(2026, 9, 15, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
    location_result = DateTimeTool(
        clock=lambda timezone: fixed_now.astimezone(timezone),
    ).run(location="Kolkata")

    assert location_result.success is True
    assert location_result.data["time"] == "17:30:00"
    assert location_result.data["timezone"] == "Asia/Kolkata"

    print("DateTime tool tests passed.")


if __name__ == "__main__":
    main()