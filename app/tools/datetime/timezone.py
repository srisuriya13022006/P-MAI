from datetime import datetime, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class TimezoneResolver:
    """Resolves common location names and IANA timezone identifiers."""

    _ALIASES = {
        "london": "Europe/London",
        "new york": "America/New_York",
        "los angeles": "America/Los_Angeles",
        "san francisco": "America/Los_Angeles",
        "seattle": "America/Los_Angeles",
        "chicago": "America/Chicago",
        "dallas": "America/Chicago",
        "houston": "America/Chicago",
        "austin": "America/Chicago",
        "boston": "America/New_York",
        "washington": "America/New_York",
        "toronto": "America/Toronto",
        "vancouver": "America/Vancouver",
        "sao paulo": "America/Sao_Paulo",
        "mexico city": "America/Mexico_City",
        "honolulu": "Pacific/Honolulu",
        "dubai": "Asia/Dubai",
        "kolkata": "Asia/Kolkata",
        "mumbai": "Asia/Kolkata",
        "delhi": "Asia/Kolkata",
        "new delhi": "Asia/Kolkata",
        "bangalore": "Asia/Kolkata",
        "bengaluru": "Asia/Kolkata",
        "chennai": "Asia/Kolkata",
        "hyderabad": "Asia/Kolkata",
        "pune": "Asia/Kolkata",
        "singapore": "Asia/Singapore",
        "tokyo": "Asia/Tokyo",
        "seoul": "Asia/Seoul",
        "beijing": "Asia/Shanghai",
        "shanghai": "Asia/Shanghai",
        "hong kong": "Asia/Hong_Kong",
        "bangkok": "Asia/Bangkok",
        "sydney": "Australia/Sydney",
        "melbourne": "Australia/Melbourne",
        "auckland": "Pacific/Auckland",
        "paris": "Europe/Paris",
        "berlin": "Europe/Berlin",
        "amsterdam": "Europe/Amsterdam",
        "rome": "Europe/Rome",
        "madrid": "Europe/Madrid",
        "moscow": "Europe/Moscow",
        "utc": "UTC",
        "gmt": "GMT",
    }

    def resolve(
        self,
        location: str | None = None,
        timezone: str | None = None,
    ) -> tuple[tzinfo, str]:
        requested = timezone or location
        if not requested:
            try:
                from app.core.config import settings
                default_tz = settings.default_timezone or "Asia/Kolkata"
                return ZoneInfo(default_tz), default_tz
            except Exception:
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