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


def web_search(query: str, max_results: int = 5) -> str:
    """Return keyless DuckDuckGo search results formatted as Markdown."""
    count = max(1, min(int(max_results), 10))
    results = _ddgs_class()().text(query, max_results=count)
    if not results:
        return "No web results found for %r." % query
    blocks = []
    for result in results[:count]:
        title = str(result.get("title") or "Untitled result").strip()
        url = str(result.get("href") or result.get("link") or "").strip()
        body = str(result.get("body") or result.get("description") or "").strip()
        heading = "[%s](%s)" % (title, url) if url else title
        blocks.append(heading + ("\n" + body if body else ""))
    return "## Search Results\n\n" + "\n\n".join(blocks)


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


class _MarkdownParser(HTMLParser):
    """A small standard-library HTML-to-Markdown converter for page text."""

    _hidden_tags = {"head", "script", "style", "noscript", "svg", "iframe", "template", "nav", "footer"}
    _blocks = {
        "address", "article", "aside", "blockquote", "dd", "div", "dl", "dt",
        "figcaption", "figure", "h1", "h2", "h3", "h4", "h5", "h6", "hr",
        "li", "main", "ol", "p", "pre", "section", "table", "tr", "ul",
    }

    def __init__(self, base_url: str = "") -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.parts: List[str] = []
        self.hidden: List[str] = []
        self.links: List[str] = []
        self.pre = False

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if self.hidden:
            if tag in self._hidden_tags:
                self.hidden.append(tag)
            return
        if tag in self._hidden_tags:
            self.hidden.append(tag)
            return
        if tag in self._blocks:
            self.parts.append("\n\n")
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.parts.append("#" * int(tag[1]) + " ")
        elif tag == "br":
            self.parts.append("\n")
        elif tag == "li":
            self.parts.append("- ")
        elif tag in ("strong", "b"):
            self.parts.append("**")
        elif tag in ("em", "i"):
            self.parts.append("*")
        elif tag == "code" and not self.pre:
            self.parts.append("`")
        elif tag == "pre":
            self.pre = True
            self.parts.append("\n```\n")
        elif tag == "a":
            href = dict(attrs).get("href", "")
            if href:
                href = urllib.parse.urljoin(self.base_url, href)
            self.links.append(href)
            if href:
                self.parts.append("[")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self.hidden:
            if tag == self.hidden[-1]:
                self.hidden.pop()
            return
        if tag == "a" and self.links:
            href = self.links.pop()
            if href:
                self.parts.append("](%s)" % href)
        elif tag in ("strong", "b"):
            self.parts.append("**")
        elif tag in ("em", "i"):
            self.parts.append("*")
        elif tag == "code" and not self.pre:
            self.parts.append("`")
        elif tag == "pre":
            self.pre = False
            self.parts.append("\n```\n")
        elif tag in self._blocks:
            self.parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        if self.hidden:
            return
        if self.pre:
            self.parts.append(data)
        else:
            self.parts.append(re.sub(r"\s+", " ", data))

    def markdown(self) -> str:
        text = "".join(self.parts)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n[ \t]+", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def visit_webpage(url: str) -> str:
    """Fetch a page and return readable Markdown, capped to a safe size."""
    raw, metadata = _get(url)
    source = _decode(raw, metadata)
    content_type = metadata.split("\n", 1)[1].lower()
    if "html" in content_type or re.search(r"<\s*(html|body|main|article)\b", source[:4096], re.I):
        parser = _MarkdownParser(url)
        parser.feed(source)
        parser.close()
        text = parser.markdown()
    else:
        text = source.strip()
    if len(text) > _MAX_OUTPUT:
        text = text[:_MAX_OUTPUT] + "\n\n... page text truncated ..."
    return text


def wikipedia_search(query: str, max_results: int = 5) -> str:
    """Find a Wikipedia article and return its introduction as Markdown."""
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
    hits = data.get("query", {}).get("search", [])
    if not hits:
        return "No Wikipedia pages found for %r." % query

    top = hits[0]
    page_url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(
        str(top.get("title", "")).replace(" ", "_"), safe="()_,-"
    )
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
    title = str(page.get("title") or top.get("title") or query)
    summary = str(page.get("extract") or "").strip()
    if not summary:
        summary = re.sub(r"<[^>]*>", " ", str(top.get("snippet") or ""))
        summary = html.unescape(re.sub(r"\s+", " ", summary)).strip()
    related = [
        "[%s](https://en.wikipedia.org/wiki/%s)"
        % (
            str(hit.get("title", "")),
            urllib.parse.quote(str(hit.get("title", "")).replace(" ", "_"), safe="()_,-"),
        )
        for hit in hits[1:]
    ]
    output = "## [Wikipedia: %s](%s)\n\n%s" % (title, page_url, summary)
    if related:
        output += "\n\nOther matches: " + ", ".join(related)
    if len(output) > _MAX_OUTPUT:
        output = output[:_MAX_OUTPUT] + "\n\n... article text truncated ..."
    return output


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
