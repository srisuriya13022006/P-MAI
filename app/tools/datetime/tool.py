from datetime import datetime, timedelta
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, model_validator

from app.tools.base import BaseTool
from app.tools.result import ToolResult
from app.tools.datetime.timezone import TimezoneResolver


class DateTimeInput(BaseModel):
    location: str | None = Field(
        default=None,
        description="City or region, for example 'New York', 'Tokyo', or 'Kolkata'.",
    )
    timezone: str | None = Field(
        default=None,
        description="IANA timezone, for example 'America/New_York', 'Asia/Tokyo', or 'UTC'.",
    )
    relative_day: Literal["today", "tomorrow", "yesterday", "current", "now"] | None = Field(
        default="today",
        description="Relative day reference: 'today', 'tomorrow', or 'yesterday'.",
    )
    query_type: Literal["date", "time", "day", "datetime"] | None = Field(
        default=None,
        description="Specific datetime field requested: 'date', 'time', 'day', or 'datetime'.",
    )

    @model_validator(mode="after")
    def validate_location_options(self):
        if self.location and self.timezone:
            raise ValueError("Provide either location or timezone, not both.")
        return self


class DateTimeTool(BaseTool):

    def __init__(
        self,
        timezone_resolver: TimezoneResolver | None = None,
        clock: Callable[..., datetime] | None = None,
    ) -> None:
        self._timezone_resolver = timezone_resolver or TimezoneResolver()
        self._clock = clock or datetime.now

    @property
    def name(self) -> str:
        return "datetime"

    @property
    def description(self) -> str:
        return (
            "Get the current or relative date and time, optionally for a city or IANA timezone."
        )

    @property
    def input_schema(self) -> type[BaseModel]:
        return DateTimeInput

    def run(self, **kwargs: Any) -> ToolResult:
        try:
            arguments = self.input_schema.model_validate(kwargs)
            tz, timezone_name = self._timezone_resolver.resolve(
                location=arguments.location,
                timezone=arguments.timezone,
            )
            now = self._clock(tz)
            if now.tzinfo is None:
                now = now.replace(tzinfo=tz)
            else:
                now = now.astimezone(tz)

            rel = (arguments.relative_day or "today").strip().lower()
            if rel == "tomorrow":
                target_dt = now + timedelta(days=1)
            elif rel == "yesterday":
                target_dt = now - timedelta(days=1)
            elif rel in ("today", "current", "now"):
                target_dt = now
            else:
                raise ValueError(
                    f"Unsupported relative day '{arguments.relative_day}'. "
                    "Use 'today', 'tomorrow', or 'yesterday'."
                )

            data = {
                "date": target_dt.strftime("%Y-%m-%d"),
                "time": target_dt.strftime("%H:%M:%S"),
                "day": target_dt.strftime("%A"),
                "timezone": timezone_name,
                "relative_day": "tomorrow" if rel == "tomorrow" else ("yesterday" if rel == "yesterday" else "today"),
            }
            if arguments.query_type:
                data["query_type"] = arguments.query_type
            if arguments.location:
                data["location"] = arguments.location

            return ToolResult(
                tool_name=self.name,
                success=True,
                data=data,
            )

        except Exception as exc:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=str(exc),
            )