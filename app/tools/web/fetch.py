import ipaddress
import re
import socket
import time
from html import unescape
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from pydantic import BaseModel, Field

from app.core.config import settings
from app.tools.base import BaseTool
from app.tools.result import ToolResult

UNSAFE_SCHEMES = ("javascript:", "data:", "file:", "ftp:", "mailto:", "tel:")
LOOPBACK_HOSTNAMES = {"localhost", "ip6-localhost", "ip6-loopback"}


def is_safe_web_url(url: str) -> tuple[bool, str]:
    """Validate that a URL uses http/https and does not target localhost, private,

    link-local, loopback, or reserved IP addresses (SSRF protection).
    Returns (is_safe, error_or_reason).
    """
    if not url or not isinstance(url, str):
        return False, "URL must be a non-empty string."

    raw = url.strip()
    lower_raw = raw.lower()
    if lower_raw.startswith(UNSAFE_SCHEMES):
        return False, "Unsupported or unsafe URL scheme."

    try:
        parsed = urlparse(raw)
    except Exception as exc:
        return False, f"Malformed URL: {exc}"

    if parsed.scheme.lower() not in ("http", "https"):
        return False, f"Unsupported URL scheme '{parsed.scheme}'. Only http and https are permitted."

    hostname = parsed.hostname
    if not hostname:
        return False, "URL is missing a valid hostname."

    lower_host = hostname.lower().strip("[]")

    # Fast-check loopback hostnames
    if lower_host in LOOPBACK_HOSTNAMES or lower_host.endswith(".localhost"):
        return False, "Requests to localhost or loopback addresses are blocked for security."

    # Validate IP address literal (standard, integer, hex, or mapped)
    ip_obj = None
    try:
        ip_obj = ipaddress.ip_address(lower_host)
    except ValueError:
        # Check integer or hex obfuscated IP addresses (e.g., 2130706433 or 0x7f000001)
        if lower_host.isdigit():
            try:
                ip_obj = ipaddress.ip_address(int(lower_host))
            except ValueError:
                pass
        elif lower_host.startswith("0x"):
            try:
                ip_obj = ipaddress.ip_address(int(lower_host, 16))
            except ValueError:
                pass

    if ip_obj is not None:
        if (
            ip_obj.is_loopback
            or ip_obj.is_private
            or ip_obj.is_link_local
            or ip_obj.is_reserved
            or ip_obj.is_multicast
            or ip_obj.is_unspecified
        ):
            return False, f"Access to private/local IP address '{lower_host}' is blocked for security."
        if ip_obj.version == 6 and ip_obj.ipv4_mapped:
            mapped = ip_obj.ipv4_mapped
            if (
                mapped.is_loopback
                or mapped.is_private
                or mapped.is_link_local
                or mapped.is_reserved
                or mapped.is_multicast
                or mapped.is_unspecified
            ):
                return False, f"Access to private/local mapped IP address '{mapped}' is blocked for security."
        return True, ""

    # Domain name: resolve DNS and validate all resolved IP addresses
    try:
        target_port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
        addr_info = socket.getaddrinfo(lower_host, target_port, proto=socket.IPPROTO_TCP)
        if not addr_info:
            return False, f"No IP addresses resolved for '{lower_host}'."
        for family, socktype, proto, canonname, sockaddr in addr_info:
            ip_str = sockaddr[0]
            resolved_ip = ipaddress.ip_address(ip_str)
            if (
                resolved_ip.is_loopback
                or resolved_ip.is_private
                or resolved_ip.is_link_local
                or resolved_ip.is_reserved
                or resolved_ip.is_multicast
                or resolved_ip.is_unspecified
            ):
                return False, f"Resolved host address '{ip_str}' is a private or loopback IP."
            if resolved_ip.version == 6 and resolved_ip.ipv4_mapped:
                mapped = resolved_ip.ipv4_mapped
                if (
                    mapped.is_loopback
                    or mapped.is_private
                    or mapped.is_link_local
                    or mapped.is_reserved
                    or mapped.is_multicast
                    or mapped.is_unspecified
                ):
                    return False, f"Resolved mapped IP '{mapped}' is a private or loopback IP."
    except socket.gaierror as e:
        return False, f"DNS resolution failed for '{lower_host}': {e}"
    except Exception as e:
        return False, f"Host validation failed for '{lower_host}': {e}"

    return True, ""


class CleanHTMLTextExtractor(HTMLParser):
    """HTML parser that strips scripts, styles, and chrome, extracting clean readable text."""

    def __init__(self) -> None:
        super().__init__()
        self.title: str = ""
        self._in_title: bool = False
        self._skip_depth: int = 0
        self._skip_tags = {"script", "style", "noscript", "svg", "canvas", "header", "footer", "nav"}
        self._block_tags = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "div", "article", "section", "blockquote", "pre"}
        self._text_chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lower_tag = tag.lower()
        if lower_tag == "title":
            self._in_title = True
        elif lower_tag in self._skip_tags:
            self._skip_depth += 1
        elif lower_tag in self._block_tags:
            self._text_chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lower_tag = tag.lower()
        if lower_tag == "title":
            self._in_title = False
        elif lower_tag in self._skip_tags:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif lower_tag in self._block_tags:
            self._text_chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title and not self.title:
            self.title = data.strip()
        elif self._skip_depth == 0:
            self._text_chunks.append(data)

    def get_text(self) -> str:
        raw_text = "".join(self._text_chunks)
        # Unescape HTML entities
        raw_text = unescape(raw_text)
        # Normalize whitespace while preserving paragraphs
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in raw_text.splitlines()]
        cleaned = "\n".join(line for line in lines if line)
        # Collapse excessive linebreaks
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()


def extract_html_content(html_text: str, max_chars: int = 8000) -> tuple[str, str, bool]:
    """Parse HTML and return (title, extracted_text, is_truncated)."""
    parser = CleanHTMLTextExtractor()
    try:
        parser.feed(html_text)
    except Exception:
        # Parser failure on severe malformed input fails safe with basic regex fallback
        pass

    title = parser.title.strip()
    if not title:
        # Fallback regex for title
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html_text, re.IGNORECASE | re.DOTALL)
        if title_match:
            title = re.sub(r"<.*?>", "", title_match.group(1)).strip()

    text = parser.get_text()
    if not text:
        # Fallback stripping if parser produced empty text
        fallback = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", "", html_text, flags=re.DOTALL | re.IGNORECASE)
        fallback = re.sub(r"<.*?>", " ", fallback)
        text = re.sub(r"\s+", " ", unescape(fallback)).strip()

    is_truncated = False
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "..."
        is_truncated = True

    return title or "Web Page", text, is_truncated


class WebFetchInput(BaseModel):
    url: str = Field(min_length=1, description="The HTTP or HTTPS URL of the web page to fetch.")


class WebFetchTool(BaseTool):
    """Safely fetch and extract readable text from a public web page."""

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return "Fetch and extract readable content from a public web page URL safely."

    @property
    def input_schema(self) -> type[BaseModel]:
        return WebFetchInput

    def run(self, **kwargs: Any) -> ToolResult:
        original_url = ""
        current_url = ""
        try:
            arguments = self.input_schema.model_validate(kwargs)
            original_url = arguments.url.strip()
            current_url = original_url

            start_time = time.monotonic()
            total_timeout = float(settings.web_fetch_timeout)
            deadline = start_time + total_timeout

            def get_remaining_timeout() -> float:
                return deadline - time.monotonic()

            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36 (P-MAI Assistant/1.0)"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.7",
                "Accept-Language": "en-US,en;q=0.9",
            }

            max_bytes = settings.web_fetch_max_bytes
            max_chars = settings.web_fetch_max_chars

            session = requests.Session()
            max_redirects = 5
            response = None

            # Redirect loop with monotonic deadline budget and per-hop SSRF validation
            for hop in range(max_redirects + 1):
                remaining = get_remaining_timeout()
                if remaining <= 0:
                    timeout_msg = (
                        "The request timed out across redirect hops."
                        if hop > 0
                        else "The request timed out while connecting to the target server."
                    )
                    return ToolResult(
                        tool_name=self.name,
                        success=False,
                        data={
                            "url": original_url,
                            "final_url": current_url,
                            "status_code": getattr(response, "status_code", None),
                            "content_type": "",
                            "title": "",
                            "text": "",
                            "truncated": False,
                        },
                        error=timeout_msg,
                    )

                # Validate current URL before each request
                is_safe, hop_error = is_safe_web_url(current_url)
                if not is_safe:
                    error_prefix = "Security check failed on redirect" if hop > 0 else "Security check failed"
                    return ToolResult(
                        tool_name=self.name,
                        success=False,
                        data={
                            "url": original_url,
                            "final_url": current_url,
                            "status_code": getattr(response, "status_code", None),
                            "content_type": "",
                            "title": "",
                            "text": "",
                            "truncated": False,
                        },
                        error=f"{error_prefix}: {hop_error}",
                    )

                remaining = get_remaining_timeout()
                if remaining <= 0:
                    return ToolResult(
                        tool_name=self.name,
                        success=False,
                        data={
                            "url": original_url,
                            "final_url": current_url,
                            "status_code": getattr(response, "status_code", None),
                            "content_type": "",
                            "title": "",
                            "text": "",
                            "truncated": False,
                        },
                        error="The request timed out while connecting to the target server.",
                    )

                response = session.get(
                    current_url,
                    headers=headers,
                    timeout=remaining,
                    allow_redirects=False,
                    stream=True,
                )

                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get("Location")
                    if not location:
                        break
                    next_url = urljoin(current_url, location)
                    current_url = next_url
                    continue

                break

            if response is None:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    data={"url": original_url, "final_url": current_url},
                    error="No response received from target URL.",
                )

            status_code = response.status_code
            content_type = response.headers.get("Content-Type", "")

            # Check HTTP error status
            if status_code >= 400:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    data={
                        "url": original_url,
                        "final_url": current_url,
                        "status_code": status_code,
                        "content_type": content_type,
                        "title": "",
                        "text": "",
                        "truncated": False,
                    },
                    error=f"HTTP request returned status {status_code}.",
                )

            # 3. Content-Type check
            lower_ct = content_type.lower()
            is_html_or_text = any(
                allowed in lower_ct
                for allowed in ("text/html", "text/plain", "application/xhtml+xml", "text/markdown")
            )
            if not is_html_or_text:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    data={
                        "url": original_url,
                        "final_url": current_url,
                        "status_code": status_code,
                        "content_type": content_type,
                        "title": "",
                        "text": "",
                        "truncated": False,
                    },
                    error=f"Unsupported page content type '{content_type}'.",
                )

            # 4. Stream response body up to max_bytes under global deadline
            content_bytes = bytearray()
            for chunk in response.iter_content(chunk_size=4096):
                if time.monotonic() >= deadline:
                    return ToolResult(
                        tool_name=self.name,
                        success=False,
                        data={
                            "url": original_url,
                            "final_url": current_url,
                            "status_code": status_code,
                            "content_type": content_type,
                            "title": "",
                            "text": "",
                            "truncated": False,
                        },
                        error="The request timed out while reading the response from the target server.",
                    )
                content_bytes.extend(chunk)
                if len(content_bytes) >= max_bytes:
                    break

            if time.monotonic() >= deadline:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    data={
                        "url": original_url,
                        "final_url": current_url,
                        "status_code": status_code,
                        "content_type": content_type,
                        "title": "",
                        "text": "",
                        "truncated": False,
                    },
                    error="The request timed out before content extraction could complete.",
                )

            encoding = response.encoding or "utf-8"
            try:
                decoded_text = content_bytes.decode(encoding, errors="replace")
            except Exception:
                decoded_text = content_bytes.decode("utf-8", errors="replace")

            # 5. Extract clean text and title
            title, extracted_text, is_truncated = extract_html_content(
                decoded_text,
                max_chars=max_chars,
            )

            if not extracted_text:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    data={
                        "url": original_url,
                        "final_url": current_url,
                        "status_code": status_code,
                        "content_type": content_type,
                        "title": title,
                        "text": "",
                        "truncated": False,
                    },
                    error="Page contained no readable text content.",
                )

            return ToolResult(
                tool_name=self.name,
                success=True,
                data={
                    "url": original_url,
                    "final_url": current_url,
                    "status_code": status_code,
                    "content_type": content_type,
                    "title": title,
                    "text": extracted_text,
                    "truncated": is_truncated,
                },
            )

        except requests.Timeout:
            return ToolResult(
                tool_name=self.name,
                success=False,
                data={"url": original_url, "final_url": current_url},
                error="The request timed out while connecting to the target server.",
            )
        except requests.ConnectionError as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                data={"url": original_url, "final_url": current_url},
                error=f"Connection failed: {e}",
            )
        except Exception as exc:
            return ToolResult(
                tool_name=self.name,
                success=False,
                data={"url": original_url, "final_url": current_url},
                error=str(exc),
            )
