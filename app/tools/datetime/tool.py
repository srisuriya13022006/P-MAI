from datetime import datetime
from typing import Any, Callable

from pydantic import BaseModel, Field, model_validator

from app.tools.base import BaseTool
from app.tools.result import ToolResult
from app.tools.datetime.timezone import TimezoneResolver


class DateTimeInput(BaseModel):
    location: str | None = Field(
        default=None,
        description="City or region, for example 'New York' or 'Kolkata'.",
    )
    timezone: str | None = Field(
        default=None,
        description="IANA timezone, for example 'America/New_York'.",
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
            "Get the current date and time, optionally for a city or IANA timezone."
        )

    @property
    def input_schema(self) -> type[BaseModel]:
        return DateTimeInput

    def run(self, **kwargs: Any) -> ToolResult:
        try:
            arguments = self.input_schema.model_validate(kwargs)
            timezone, timezone_name = self._timezone_resolver.resolve(
                location=arguments.location,
                timezone=arguments.timezone,
            )
            now = self._clock(timezone)

            return ToolResult(
                tool_name=self.name,
                success=True,
                data={
                    "date": now.strftime("%Y-%m-%d"),
                    "time": now.strftime("%H:%M:%S"),
                    "day": now.strftime("%A"),
                    "timezone": timezone_name,
                },
            )

        except Exception as exc:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=str(exc),
            )