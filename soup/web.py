"""Keyless search, web reading, Wikipedia lookup, and local transcription."""

from __future__ import annotations

import html
import json
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Dict, List, Optional, Tuple

from .expr import Call, Expr, Lit, Seq, call

_AGENT = "Soup/0.1 (concept-oriented assistant)"
_MAX_PAGE_BYTES = 3 * 1024 * 1024
_MAX_AUDIO_BYTES = 100 * 1024 * 1024
_MAX_OUTPUT = 40000


def _ddgs_class():
    """Load the current package name, with its Python 3.9 predecessor."""
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError as exc:
            raise RuntimeError(
                "web search needs the ddgs package; install the project requirements"
            ) from exc
    return DDGS


def web_search(query: str, max_results: int = 5) -> Expr:
    """Return DuckDuckGo results as Soup values with their fields intact."""
    count = max(1, min(int(max_results), 10))
    results = _ddgs_class()().text(query, max_results=count)
    return call(
        "SearchResults",
        query=Lit(query),
        results=Seq(
            tuple(
                call(
                    "SearchResult",
                    title=Lit(str(result.get("title") or "Untitled result").strip()),
                    url=Lit(str(result.get("href") or result.get("link") or "").strip()),
                    snippet=Lit(
                        str(result.get("body") or result.get("description") or "").strip()
                    ),
                )
                for result in results[:count]
            )
        ),
    )


def _get(url: str, max_bytes: int = _MAX_PAGE_BYTES) -> Tuple[bytes, str]:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("only http and https URLs can be visited")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": _AGENT, "Accept": "text/html,text/plain,application/json,*/*"},
    )
    with urllib.request.urlopen(request, timeout=20.0) as response:
        raw = response.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raw = raw[:max_bytes]
        headers = getattr(response, "headers", None)
        charset = headers.get_content_charset() if headers is not None else None
        content_type = headers.get("Content-Type", "") if headers is not None else ""
    encoding = charset or "utf-8"
    return raw, "%s\n%s" % (encoding, content_type)


def _decode(raw: bytes, metadata: str) -> str:
    encoding = metadata.split("\n", 1)[0]
    try:
        return raw.decode(encoding, "replace")
    except LookupError:
        return raw.decode("utf-8", "replace")


class _PageParser(HTMLParser):
    """Extract a page as headings, paragraphs, and links."""

    _hidden_tags = {"script", "style", "noscript", "svg", "iframe", "template", "nav", "footer"}
    _headings = {"h1", "h2", "h3", "h4", "h5", "h6"}
    _paragraphs = {"address", "blockquote", "dd", "figcaption", "li", "p", "pre", "td", "th"}

    def __init__(self, base_url: str = "") -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.hidden: List[str] = []
        self.title = ""
        self.sections: List[Dict[str, object]] = [{"heading": "", "paragraphs": []}]
        self.links: List[Tuple[str, str]] = []
        self.capture_tag: Optional[str] = None
        self.capture_kind: Optional[str] = None
        self.capture: List[str] = []
        self.link_url: Optional[str] = None
        self.link_text: List[str] = []
        self.truncated = False
        self.char_count = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if self.hidden:
            if tag in self._hidden_tags:
                self.hidden.append(tag)
            return
        if tag in self._hidden_tags:
            self.hidden.append(tag)
            return
        if tag == "title":
            self._start_capture(tag, "title")
        elif tag in self._headings:
            self._start_capture(tag, "heading")
        elif tag in self._paragraphs:
            self._start_capture(tag, "paragraph")
        elif tag == "a":
            href = dict(attrs).get("href", "")
            self.link_url = urllib.parse.urljoin(self.base_url, href) if href else ""
            self.link_text = []
        elif tag == "br" and self.capture_tag is not None:
            self.capture.append(" ")

    def _start_capture(self, tag: str, kind: str) -> None:
        self._flush_capture()
        self.capture_tag = tag
        self.capture_kind = kind
        self.capture = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self.hidden:
            if tag == self.hidden[-1]:
                self.hidden.pop()
            return
        if tag == "a" and self.link_url is not None:
            text = self._plain("".join(self.link_text))
            if self.link_url and text and len(self.links) < 200:
                self.links.append((text, self.link_url))
            self.link_url = None
            self.link_text = []
        if tag == self.capture_tag:
            self._flush_capture()

    def handle_data(self, data: str) -> None:
        if self.hidden:
            return
        if self.capture_tag is not None:
            remaining = _MAX_OUTPUT - self.char_count
            if remaining <= 0:
                self.truncated = True
                return
            kept = data[:remaining]
            self.capture.append(kept)
            self.char_count += len(kept)
            self.truncated = self.truncated or len(kept) < len(data)
        if self.link_url is not None:
            self.link_text.append(data)

    @staticmethod
    def _plain(text: str) -> str:
        return re.sub(r"\s+", " ", html.unescape(text)).strip()

    def _flush_capture(self) -> None:
        if self.capture_tag is None:
            return
        text = self._plain("".join(self.capture))
        kind = self.capture_kind
        tag = self.capture_tag
        self.capture_tag = None
        self.capture_kind = None
        self.capture = []
        if not text:
            return
        if kind == "title":
            if not self.title:
                self.title = text
        elif kind == "heading":
            current = self.sections[-1]
            if current["heading"] or current["paragraphs"]:
                current = {"heading": "", "paragraphs": []}
                self.sections.append(current)
            current["heading"] = text
            if tag == "h1" and not self.title:
                self.title = text
        elif kind == "paragraph":
            self.sections[-1]["paragraphs"].append(text)

    def expression(self, url: str) -> Call:
        self._flush_capture()
        sections = []
        for section in self.sections:
            heading = str(section["heading"])
            paragraphs = section["paragraphs"]
            if heading or paragraphs:
                sections.append(
                    call(
                        "WebSection",
                        heading=Lit(heading),
                        paragraphs=Seq(tuple(Lit(text) for text in paragraphs)),
                    )
                )
        links = tuple(call("WebLink", text=Lit(text), url=Lit(href)) for text, href in self.links)
        return call(
            "WebPage",
            url=Lit(url),
            title=Lit(self.title),
            sections=Seq(tuple(sections)),
            links=Seq(links),
            truncated=Lit(self.truncated),
        )


def visit_webpage(url: str) -> Expr:
    """Fetch a page into structured Soup sections and links."""
    raw, metadata = _get(url)
    source = _decode(raw, metadata)
    content_type = metadata.split("\n", 1)[1].lower()
    if "html" in content_type or re.search(r"<\s*(html|body|main|article)\b", source[:4096], re.I):
        parser = _PageParser(url)
        parser.feed(source)
        parser.close()
        return parser.expression(url)
    paragraph = source[:_MAX_OUTPUT]
    return call(
        "WebPage",
        url=Lit(url),
        title=Lit(""),
        sections=Seq(
            (call("WebSection", heading=Lit(""), paragraphs=Seq((Lit(paragraph),))),)
        ),
        links=Seq(),
        truncated=Lit(len(source) > _MAX_OUTPUT),
    )


def wikipedia_search(query: str, max_results: int = 5) -> Expr:
    """Return Wikipedia matches as SearchResults, with an intro on the top hit."""
    count = max(1, min(int(max_results), 10))
    search_url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": count,
            "format": "json",
            "utf8": 1,
        }
    )
    raw, metadata = _get(search_url)
    data = json.loads(_decode(raw, metadata))
    hits = data.get("query", {}).get("search", [])[:count]
    if not hits:
        return call("SearchResults", query=Lit(query), results=Seq())

    top = hits[0]
    summary = ""
    if top.get("pageid"):
        detail_url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
            {
                "action": "query",
                "pageids": top.get("pageid", ""),
                "prop": "extracts",
                "exintro": 1,
                "explaintext": 1,
                "redirects": 1,
                "format": "json",
            }
        )
        raw, metadata = _get(detail_url)
        details = json.loads(_decode(raw, metadata))
        pages = details.get("query", {}).get("pages", {})
        if isinstance(pages, list):
            page = pages[0] if pages else {}
        else:
            page = next(iter(pages.values()), {})
        summary = _plain_fragment(page.get("extract", ""))

    results = []
    for index, hit in enumerate(hits):
        title = str(hit.get("title") or query).strip()
        page_url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(
            title.replace(" ", "_"), safe="()_,-"
        )
        snippet = summary if index == 0 and summary else _plain_fragment(hit.get("snippet", ""))
        results.append(
            call(
                "SearchResult",
                title=Lit(title),
                url=Lit(page_url),
                snippet=Lit(snippet),
            )
        )
    return call("SearchResults", query=Lit(query), results=Seq(tuple(results)))


def _plain_fragment(value) -> str:
    text = re.sub(r"<[^>]*>", " ", str(value or ""))
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def transcribe_audio(audio: str, model: Optional[str] = None) -> str:
    """Transcribe a local audio file or HTTP(S) audio URL with a local model."""
    temporary = None
    path = audio
    if audio.startswith(("http://", "https://")):
        suffix = os.path.splitext(urllib.parse.urlsplit(audio).path)[1][:12] or ".audio"
        handle = tempfile.NamedTemporaryFile(prefix="soup-audio-", suffix=suffix, delete=False)
        temporary = handle.name
        try:
            parsed = urllib.parse.urlsplit(audio)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                raise ValueError("only http and https audio URLs can be transcribed")
            request = urllib.request.Request(audio, headers={"User-Agent": _AGENT})
            with urllib.request.urlopen(request, timeout=30.0) as response:
                total = 0
                while True:
                    chunk = response.read(min(1024 * 1024, _MAX_AUDIO_BYTES + 1 - total))
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > _MAX_AUDIO_BYTES:
                        raise ValueError("audio URL exceeds the 100 MB limit")
                    handle.write(chunk)
            handle.close()
            path = temporary
        except Exception:
            handle.close()
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

    try:
        try:
            import mlx_whisper
        except ImportError:
            mlx_whisper = None
        if mlx_whisper is not None:
            result = mlx_whisper.transcribe(
                path,
                path_or_hf_repo=model or "mlx-community/whisper-large-v3-turbo",
                verbose=False,
            )
            return str(result.get("text", "")).strip()

        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "speech transcription needs mlx-whisper on Apple Silicon or transformers with PyTorch"
            ) from exc
        result = pipeline(
            "automatic-speech-recognition",
            model=model or "openai/whisper-large-v3-turbo",
        )(path)
        return str(result.get("text", "")).strip()
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass
