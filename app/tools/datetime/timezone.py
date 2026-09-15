from datetime import datetime, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class TimezoneResolver:
    """Resolves common location names and IANA timezone identifiers."""

    _ALIASES = {
        "london": "Europe/London",
        "new york": "America/New_York",
        "los angeles": "America/Los_Angeles",
        "chicago": "America/Chicago",
        "toronto": "America/Toronto",
        "sao paulo": "America/Sao_Paulo",
        "mexico city": "America/Mexico_City",
        "honolulu": "Pacific/Honolulu",
        "dubai": "Asia/Dubai",
        "kolkata": "Asia/Kolkata",
        "mumbai": "Asia/Kolkata",
        "delhi": "Asia/Kolkata",
        "bangalore": "Asia/Kolkata",
        "bengaluru": "Asia/Kolkata",
        "chennai": "Asia/Kolkata",
        "hyderabad": "Asia/Kolkata",
        "singapore": "Asia/Singapore",
        "tokyo": "Asia/Tokyo",
        "seoul": "Asia/Seoul",
        "beijing": "Asia/Shanghai",
        "sydney": "Australia/Sydney",
        "melbourne": "Australia/Melbourne",
        "paris": "Europe/Paris",
        "berlin": "Europe/Berlin",
        "moscow": "Europe/Moscow",
    }

    def resolve(
        self,
        location: str | None = None,
        timezone: str | None = None,
    ) -> tuple[tzinfo, str]:
        requested = timezone or location
        if not requested:
            local_timezone = datetime.now().astimezone().tzinfo
            timezone_name = getattr(local_timezone, "key", None)
            return local_timezone, timezone_name or datetime.now().astimezone().tzname()

        normalized = requested.strip().lower()
        timezone_name = self._ALIASES.get(normalized, requested.strip())

        try:
            return ZoneInfo(timezone_name), timezone_name
        except ZoneInfoNotFoundError as exc:
            raise ValueError(
                f"Unknown location or timezone '{requested}'. "
                "Use a city name or an IANA timezone such as Asia/Kolkata."
            ) from exc