from datetime import datetime
from zoneinfo import ZoneInfo
import pytest

from app.agent.orchestrator import MAIOrchestrator
from app.agent.policy import apply_policy
from app.schemas.agent import AgentDecision
from app.tools.datetime.timezone import TimezoneResolver
from app.tools.datetime.tool import DateTimeTool


@pytest.fixture
def fixed_clock():
    # Fixed reference: Tuesday, 2026-10-06 12:00:00 UTC
    fixed_utc = datetime(2026, 10, 6, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
    return lambda tz: fixed_utc.astimezone(tz)


def _base_decision() -> AgentDecision:
    return AgentDecision(
        intent="local",
        route="local",
        needs_clarification=False,
        tools=[],
        tool_arguments={},
        reason="test",
    )


def test_current_datetime_basic():
    """Tool returns date, time, day, and timezone."""
    tool = DateTimeTool()
    result = tool.run()

    assert result.success is True
    assert result.tool_name == "datetime"
    assert "date" in result.data
    assert "time" in result.data
    assert "day" in result.data
    assert "timezone" in result.data


def test_current_date(fixed_clock):
    """Verifies date retrieval with fixed clock."""
    tool = DateTimeTool(clock=fixed_clock)
    result = tool.run(timezone="UTC", query_type="date")

    assert result.success is True
    assert result.data["date"] == "2026-10-06"
    assert result.data["day"] == "Tuesday"


def test_current_time(fixed_clock):
    """Verifies time retrieval with fixed clock."""
    tool = DateTimeTool(clock=fixed_clock)
    result = tool.run(timezone="UTC", query_type="time")

    assert result.success is True
    assert result.data["time"] == "12:00:00"


def test_current_day(fixed_clock):
    """Verifies weekday retrieval with fixed clock."""
    tool = DateTimeTool(clock=fixed_clock)
    result = tool.run(timezone="UTC", query_type="day")

    assert result.success is True
    assert result.data["day"] == "Tuesday"


def test_tomorrow_calculation(fixed_clock):
    """Tomorrow is exactly one day after fixed date."""
    tool = DateTimeTool(clock=fixed_clock)
    result = tool.run(timezone="UTC", relative_day="tomorrow")

    assert result.success is True
    assert result.data["date"] == "2026-10-07"
    assert result.data["day"] == "Wednesday"
    assert result.data["relative_day"] == "tomorrow"


def test_yesterday_calculation(fixed_clock):
    """Yesterday is exactly one day before fixed date."""
    tool = DateTimeTool(clock=fixed_clock)
    result = tool.run(timezone="UTC", relative_day="yesterday")

    assert result.success is True
    assert result.data["date"] == "2026-10-05"
    assert result.data["day"] == "Monday"
    assert result.data["relative_day"] == "yesterday"


def test_timezone_conversions(fixed_clock):
    """Tests deterministic timezone conversions for Asia/Kolkata, America/New_York, and Asia/Tokyo."""
    tool = DateTimeTool(clock=fixed_clock)

    # 1. Asia/Kolkata (UTC + 5:30) -> 17:30:00, 2026-10-06
    kolkata = tool.run(location="Kolkata")
    assert kolkata.success is True
    assert kolkata.data["time"] == "17:30:00"
    assert kolkata.data["date"] == "2026-10-06"
    assert kolkata.data["timezone"] == "Asia/Kolkata"

    # 2. America/New_York (EDT, UTC - 4:00) -> 08:00:00, 2026-10-06
    new_york = tool.run(location="New York")
    assert new_york.success is True
    assert new_york.data["time"] == "08:00:00"
    assert new_york.data["date"] == "2026-10-06"
    assert new_york.data["timezone"] == "America/New_York"

    # 3. Asia/Tokyo (JST, UTC + 9:00) -> 21:00:00, 2026-10-06
    tokyo = tool.run(location="Tokyo")
    assert tokyo.success is True
    assert tokyo.data["time"] == "21:00:00"
    assert tokyo.data["date"] == "2026-10-06"
    assert tokyo.data["timezone"] == "Asia/Tokyo"


def test_invalid_timezone_fails_gracefully():
    """Invalid timezone returns clean error without crashing."""
    tool = DateTimeTool()
    result = tool.run(timezone="Atlantis/NonExistent")

    assert result.success is False
    assert result.error is not None
    assert "Unknown location or timezone" in result.error


def test_location_and_timezone_mutual_exclusion():
    """Supplying both location and timezone returns clean validation error."""
    tool = DateTimeTool()
    result = tool.run(location="Tokyo", timezone="Asia/Tokyo")

    assert result.success is False
    assert "Provide either location or timezone, not both" in result.error


@pytest.mark.parametrize(
    "user_message, expected_query_type, expected_rel_day",
    [
        ("What time is it?", "time", "today"),
        ("What's today's date?", "date", "today"),
        ("What day is today?", "day", "today"),
        ("What date is tomorrow?", "date", "tomorrow"),
        ("What date was yesterday?", "date", "yesterday"),
        ("What day will tomorrow be?", "day", "tomorrow"),
        ("What is the current date and time?", "datetime", "today"),
    ],
)
def test_policy_routes_datetime_queries(user_message, expected_query_type, expected_rel_day):
    """Verifies deterministic policy routing for all basic datetime query variations."""
    decision = _base_decision()
    routed = apply_policy(user_message, decision)

    assert routed.route == "tool"
    assert routed.tools == ["datetime"]
    assert routed.tool_arguments.get("query_type") == expected_query_type
    assert routed.tool_arguments.get("relative_day") == expected_rel_day


def test_policy_extracts_location_for_datetime():
    """Verifies location extraction in policy for 'What time is it in Tokyo?'."""
    decision = _base_decision()
    routed = apply_policy("What time is it in Tokyo?", decision)

    assert routed.route == "tool"
    assert routed.tools == ["datetime"]
    assert routed.tool_arguments.get("location") == "Tokyo"
    assert routed.tool_arguments.get("query_type") == "time"


def test_orchestrator_formats_concise_datetime_responses(fixed_clock):
    """Verifies concise formatting for date, time, day, and tomorrow in orchestrator."""
    orchestrator = MAIOrchestrator()
    orchestrator.tool_registry._tools["datetime"] = DateTimeTool(clock=fixed_clock)
    orchestrator.router.analyze = lambda msg: _base_decision()

    # 1. date -> date-only response (no time)
    r_date = orchestrator.handle("What's today's date?", [])
    assert "2026-10-06" in r_date
    assert "12:00:00" not in r_date
    assert "Tuesday" not in r_date

    # 2. time -> time-only response (no full date)
    r_time = orchestrator.handle("What time is it?", [])
    assert "17:30:00" in r_time or "12:00:00" in r_time
    assert "2026-10-06" not in r_time

    # 3. day -> weekday-only response (no time or numeric date)
    r_day = orchestrator.handle("What day is today?", [])
    assert "Tuesday" in r_day
    assert "2026-10-06" not in r_day
    assert "12:00:00" not in r_day

    # 4. datetime -> date + time response
    r_dt = orchestrator.handle("What is the current date and time?", [])
    assert "2026-10-06" in r_dt
    assert "Tuesday" in r_dt
    assert ("17:30:00" in r_dt or "12:00:00" in r_dt)

    # 5. location + time -> location-specific time
    r_loc_time = orchestrator.handle("What time is it in Tokyo?", [])
    assert "21:00:00" in r_loc_time
    assert "Tokyo" in r_loc_time
    assert "2026-10-06" not in r_loc_time

    # 6. location + date -> location-specific date
    r_loc_date = orchestrator.handle("What's the date in Tokyo?", [])
    assert "2026-10-06" in r_loc_date
    assert "Tokyo" in r_loc_date
    assert "21:00:00" not in r_loc_date

    # 6b. location + day -> location-specific day
    r_loc_day = orchestrator.handle("What's the day in Tokyo?", [])
    assert "Tuesday" in r_loc_day
    assert "Tokyo" in r_loc_day
    assert "2026-10-06" not in r_loc_day
    assert "21:00:00" not in r_loc_day

    # 7. tomorrow -> tomorrow-specific response
    r_tomorrow_date = orchestrator.handle("What date is tomorrow?", [])
    assert "2026-10-07" in r_tomorrow_date
    assert "Tomorrow" in r_tomorrow_date
    assert "Today is" not in r_tomorrow_date

    r_tomorrow_day = orchestrator.handle("What day will tomorrow be?", [])
    assert "Wednesday" in r_tomorrow_day
    assert "Tomorrow" in r_tomorrow_day
    assert "Today is" not in r_tomorrow_day

    # 8. yesterday -> yesterday-specific response
    r_yesterday_date = orchestrator.handle("What date was yesterday?", [])
    assert "2026-10-05" in r_yesterday_date
    assert "Yesterday" in r_yesterday_date
    assert "Today is" not in r_yesterday_date