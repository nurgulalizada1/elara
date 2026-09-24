"""web_fetch: download a public web page as text. Output is untrusted."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import ClassVar
from urllib.parse import urljoin

import httpx
from pydantic import BaseModel, Field

from elara.core.errors import ToolError
from elara.security.network import validate_public_url
from elara.security.untrusted import Trust
from elara.tools.base import Permission, Tool, ToolContext

MAX_BYTES = 2_000_000
MAX_REDIRECTS = 5


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "template", "svg", "head"}
    BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section",
             "article", "pre", "blockquote"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        if tag == "title":
            self._in_title = True
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> tuple[str, str]:
    p = _TextExtractor()
    p.feed(html)
    text = re.sub(r"[ \t\r\f\v]+", " ", "".join(p.parts))
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return p.title.strip(), text


class FetchInput(BaseModel):
    url: str
    max_chars: int = Field(default=15_000, ge=500, le=100_000)


class FetchOutput(BaseModel):
    url: str
    final_url: str
    status: int
    title: str
    content: str
    truncated: bool


class WebFetchTool(Tool):
    name: ClassVar[str] = "web_fetch"
    description: ClassVar[str] = ("Fetch a public web page (http/https) and return its readable "
                                  "text. Page content is untrusted data.")
    Input = FetchInput
    Output = FetchOutput
    permission = Permission.READ_ONLY
    output_trust = Trust.UNTRUSTED
    category = "web"
    timeout_s = 25.0
    safety_notes = "Blocks private/loopback addresses (SSRF) and re-checks every redirect."

    def __init__(self, client: httpx.AsyncClient, allow_private: bool = False):
        self.client = client
        self.allow_private = allow_private

    def source_label(self, args: FetchInput) -> str:
        return f"web:{args.url}"

    async def run(self, args: FetchInput, ctx: ToolContext) -> FetchOutput:
        url = args.url
        for _ in range(MAX_REDIRECTS + 1):
            await validate_public_url(url, allow_private=self.allow_private)
            try:
                async with self.client.stream("GET", url, follow_redirects=False, timeout=20,
                                              headers={"user-agent": "ELARA/0.1"}) as resp:
                    if resp.is_redirect and resp.headers.get("location"):
                        url = urljoin(url, resp.headers["location"])
                        continue
                    ctype = resp.headers.get("content-type", "")
                    if not any(t in ctype for t in ("text/", "html", "json", "xml")):
                        raise ToolError(f"unsupported content type: {ctype or 'unknown'}")
                    body = bytearray()
                    async for chunk in resp.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > MAX_BYTES:
                            break
                    status = resp.status_code
                    encoding = resp.encoding or "utf-8"
            except httpx.HTTPError as e:
                raise ToolError(f"fetch failed: {type(e).__name__}") from e
            if status >= 400:
                raise ToolError(f"HTTP {status} from {url}")
            raw = body.decode(encoding, errors="replace")
            title, text = html_to_text(raw) if "html" in ctype else ("", raw)
            return FetchOutput(url=args.url, final_url=url, status=status, title=title,
                               content=text[: args.max_chars],
                               truncated=len(text) > args.max_chars)
        raise ToolError("too many redirects")

    def render(self, out: FetchOutput) -> str:
        return f"{out.title}\n{out.final_url}\n\n{out.content}" + (
            "\n[... truncated ...]" if out.truncated else "")
