"""Turn HTML pages, PDFs, Word files and text files into clean plain text."""
from __future__ import annotations

import io
import json
import re
from html.parser import HTMLParser
from pathlib import Path

SKIP_TAGS = {"script", "style", "noscript", "svg", "nav", "footer", "header", "aside", "form", "iframe", "template", "button", "select"}
BLOCK_TAGS = {"p", "div", "section", "article", "main", "li", "ul", "ol", "br", "tr", "table", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "figure", "figcaption", "dd", "dt"}
HEADING_TAGS = {"h1", "h2", "h3", "h4"}
SUPPORTED_FILE_TYPES = (".pdf", ".docx", ".txt", ".md", ".csv")
MAX_PDF_PAGES = 200


class ExtractError(ValueError):
    pass


class _Page(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.meta: list[str] = []
        self.jsonld: list[str] = []
        self.lines: list[tuple[bool, str]] = []  # (is_heading, text)
        self._buf: list[str] = []
        self._skip = 0
        self._in_title = False
        self._in_jsonld = False
        self._heading = False
        self._jsonld_buf: list[str] = []

    # -- helpers ---------------------------------------------------------
    def _flush(self) -> None:
        text = re.sub(r"\s+", " ", " ".join(self._buf)).strip()
        self._buf = []
        if text:
            self.lines.append((self._heading, text))

    # -- parser callbacks --------------------------------------------------
    def handle_starttag(self, tag, attrs):
        a = {k: (v or "") for k, v in attrs}
        if tag == "title":
            self._in_title = True
        elif tag == "meta":
            key = (a.get("name") or a.get("property") or "").lower()
            if key in ("description", "og:description", "twitter:description", "og:title", "og:site_name") and a.get("content"):
                self.meta.append(a["content"].strip())
        elif tag == "script" and "ld+json" in a.get("type", "").lower():
            self._in_jsonld = True
            self._jsonld_buf = []
            return
        if tag in SKIP_TAGS:
            self._skip += 1
            return
        if tag in BLOCK_TAGS:
            self._flush()
            self._heading = tag in HEADING_TAGS

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag == "script" and self._in_jsonld:
            self._in_jsonld = False
            self.jsonld.append("".join(self._jsonld_buf))
            return
        if tag in SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
            return
        if tag in BLOCK_TAGS:
            self._flush()
            self._heading = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif self._in_jsonld:
            self._jsonld_buf.append(data)
        elif not self._skip:
            self._buf.append(data)

    def close(self):
        super().close()
        self._flush()


def _jsonld_descriptions(blobs: list[str]) -> list[str]:
    out: list[str] = []

    def walk(node):
        if isinstance(node, dict):
            for key in ("name", "description", "slogan", "foundingDate"):
                val = node.get(key)
                if isinstance(val, str) and len(val) > 15:
                    out.append(val.strip())
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    for blob in blobs:
        try:
            walk(json.loads(blob))
        except Exception:
            continue
    return out


def decode_bytes(data: bytes, hint: str | None = None) -> str:
    for enc in filter(None, [hint, "utf-8", "cp1252"]):
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("latin-1", errors="replace")


def html_to_text(html: str) -> tuple[str, str]:
    """Return (title, text). Keeps headings and sentence-like lines; falls back to
    meta tags / JSON-LD so script-rendered sites still yield something useful."""
    page = _Page()
    try:
        page.feed(html)
        page.close()
    except Exception:  # malformed HTML: keep whatever was parsed
        pass
    title = re.sub(r"\s+", " ", page.title).strip()
    seen: set[str] = set()
    parts: list[str] = []

    def add(text: str) -> None:
        key = text.lower()
        if key not in seen:
            seen.add(key)
            parts.append(text)

    if title:
        add(title)
    for m in page.meta:
        add(m)
    for d in _jsonld_descriptions(page.jsonld):
        add(d)
    for is_heading, line in page.lines:
        if is_heading or len(line.split()) >= 5:
            add(line)
    return title, "\n".join(parts).strip()


def pdf_bytes_to_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise ExtractError("PDF is password-protected.")
        pages = reader.pages[:MAX_PDF_PAGES]
        return "\n\n".join((p.extract_text() or "") for p in pages).strip()
    except ExtractError:
        raise
    except Exception as exc:
        raise ExtractError(f"Could not read PDF ({exc.__class__.__name__}).")


def docx_bytes_to_text(data: bytes) -> str:
    try:
        from docx import Document

        doc = Document(io.BytesIO(data))
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))
        return "\n".join(parts).strip()
    except Exception as exc:
        raise ExtractError(f"Could not read Word file ({exc.__class__.__name__}).")


def bytes_to_text(data: bytes, content_type: str = "", filename: str = "") -> tuple[str, str]:
    """Dispatch on content type / extension. Returns (title, text)."""
    name = filename.lower()
    if content_type == "application/pdf" or name.endswith(".pdf"):
        return "", pdf_bytes_to_text(data)
    if name.endswith(".docx"):
        return "", docx_bytes_to_text(data)
    text = decode_bytes(data)
    if content_type.startswith("text/html") or content_type == "application/xhtml+xml" or "<html" in text[:2000].lower():
        return html_to_text(text)
    return "", text.strip()


def file_to_text(path: str, max_mb: int) -> tuple[str, str]:
    p = Path(path)
    ext = p.suffix.lower()
    if ext not in SUPPORTED_FILE_TYPES:
        raise ExtractError(f"Unsupported file type '{ext}'. Use: {', '.join(SUPPORTED_FILE_TYPES)}")
    size = p.stat().st_size
    if size > max_mb * 1024 * 1024:
        raise ExtractError(f"File is larger than {max_mb} MB.")
    return bytes_to_text(p.read_bytes(), filename=p.name)
