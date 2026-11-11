import pytest

from agent import security
from agent.chunking import chunk_text
from agent.extract import ExtractError, bytes_to_text, file_to_text, html_to_text


# ------------------------------------------------------------------ login
def test_login(settings):
    assert security.check_login("Priya Sharma", "class-secret-123", settings)
    assert not security.check_login("priya", "wrong", settings)
    assert not security.check_login("../etc", "class-secret-123", settings)  # bad username
    assert not security.check_login("", "class-secret-123", settings)
    assert security.normalize_username("Priya  Sharma") == "priya-sharma"


def test_login_refuses_when_no_code_configured(settings, monkeypatch):
    monkeypatch.setenv("APP_ACCESS_CODE", "")
    from agent.config import get_settings

    s = get_settings()
    assert not security.check_login("priya", "anything", s)
    monkeypatch.setenv("ALLOW_NO_AUTH", "1")
    assert security.check_login("priya", "anything", get_settings())


# --------------------------------------------------------------- SSRF guard
@pytest.mark.parametrize("url", [
    "http://localhost/", "http://127.0.0.1/", "http://10.0.0.1/", "http://192.168.1.1/",
    "http://169.254.169.254/latest/meta-data/", "http://evil.com/", "http://meta.evil.com/",
    "http://loop.evil.com/", "http://v6.evil.com/", "http://v6loop.evil.com/",
    "file:///etc/passwd", "ftp://good.com/x", "http://good.com:22/", "http://user:pw@good.com/",
    "http://nonexistent.invalid/", "not a url", "",
])
def test_blocked_urls(url):
    with pytest.raises(security.UrlBlocked):
        security.validate_url(url)


def test_public_url_allowed():
    assert security.validate_url("https://good.com/about") == "https://good.com/about"
    assert security.validate_url("http://good.com:80/x")


class FakeResp:
    def __init__(self, status=200, headers=None, body=b"", chunk=1000):
        self.status_code, self.headers, self._body, self._chunk = status, headers or {}, body, chunk

    def iter_content(self, n):
        for i in range(0, len(self._body), self._chunk):
            yield self._body[i : i + self._chunk]

    def close(self):
        pass


class FakeSession:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, **kw):
        self.calls.append(url)
        assert kw["allow_redirects"] is False
        return self.responses.pop(0)


def test_safe_get_ok(settings):
    s = FakeSession(FakeResp(200, {"Content-Type": "text/html; charset=utf-8"}, b"<html>hi</html>"))
    r = security.safe_get("https://good.com/", settings, s)
    assert r.data == b"<html>hi</html>" and r.content_type == "text/html"


def test_redirect_to_private_blocked(settings):
    s = FakeSession(FakeResp(302, {"Location": "http://meta.evil.com/latest"}))
    with pytest.raises(security.UrlBlocked):
        security.safe_get("https://good.com/", settings, s)
    assert s.calls == ["https://good.com/"]  # the private hop was never requested


def test_redirect_chain_followed_and_capped(settings):
    ok = FakeSession(FakeResp(301, {"Location": "/new"}), FakeResp(200, {"Content-Type": "text/plain"}, b"hello"))
    assert security.safe_get("https://good.com/old", settings, ok).url == "https://good.com/new"
    loop = FakeSession(*[FakeResp(302, {"Location": "/again"}) for _ in range(10)])
    with pytest.raises(security.FetchError):
        security.safe_get("https://good.com/", settings, loop)


def test_size_type_and_status_limits(settings):
    big = FakeSession(FakeResp(200, {"Content-Type": "text/html"}, b"x" * (settings.max_fetch_bytes + 10), chunk=65536))
    with pytest.raises(security.FetchError, match="too large"):
        security.safe_get("https://good.com/", settings, big)
    with pytest.raises(security.FetchError, match="content type"):
        security.safe_get("https://good.com/", settings, FakeSession(FakeResp(200, {"Content-Type": "application/zip"}, b"PK")))
    with pytest.raises(security.FetchError, match="HTTP 404"):
        security.safe_get("https://good.com/", settings, FakeSession(FakeResp(404)))


# ------------------------------------------------------------------ extract
HTML = """<html><head><title>Acme Foods | About</title>
<meta name="description" content="Acme Foods makes honest snacks from local farms since 1998.">
<script type="application/ld+json">{"@type":"Organization","name":"Acme Foods","description":"Family-owned snack maker focused on sustainable sourcing."}</script>
<style>.x{color:red}</style></head>
<body><nav>Home Products Contact us today for more</nav>
<header>Site header banner navigation links here okay</header>
<main><h1>Our promise</h1><p>We source every ingredient from farms within 200 kilometres of our kitchens.</p>
<p>Short.</p><p>Every pack is fully recyclable and our factories run on solar power.</p></main>
<script>var secret = "do not index this script content at all";</script>
<footer>Copyright 2026 footer legal text goes right here</footer></body></html>"""


def test_html_extraction():
    title, text = html_to_text(HTML)
    assert title == "Acme Foods | About"
    assert "honest snacks from local farms" in text          # meta description
    assert "Family-owned snack maker" in text                # JSON-LD
    assert "Our promise" in text                             # heading kept
    assert "within 200 kilometres" in text and "solar power" in text
    for junk in ("Home Products Contact", "Copyright 2026", "do not index", "color:red", "Short."):
        assert junk not in text


def test_js_only_page_still_yields_meta():
    html = '<html><head><title>Wipro</title><meta property="og:description" content="Wipro is a leading technology services and consulting company."></head><body><div id="root"></div><script src="app.js"></script></body></html>'
    _, text = html_to_text(html)
    assert "leading technology services" in text


def test_bytes_dispatch_and_file_types(tmp_path):
    assert bytes_to_text(b"plain text here", "text/plain")[1] == "plain text here"
    p = tmp_path / "notes.txt"
    p.write_text("Brand voice is warm and direct.", encoding="utf-8")
    assert "warm and direct" in file_to_text(str(p), 10)[1]
    bad = tmp_path / "x.exe"
    bad.write_bytes(b"MZ")
    with pytest.raises(ExtractError):
        file_to_text(str(bad), 10)
    big = tmp_path / "big.txt"
    big.write_bytes(b"a" * (2 * 1024 * 1024))
    with pytest.raises(ExtractError, match="larger"):
        file_to_text(str(big), 1)


def test_docx_extraction(tmp_path):
    from docx import Document

    d = Document()
    d.add_paragraph("Acme Foods tone of voice: friendly, plain-spoken and never salesy.")
    t = d.add_table(rows=1, cols=2)
    t.rows[0].cells[0].text, t.rows[0].cells[1].text = "Tagline", "Honest snacks"
    p = tmp_path / "guide.docx"
    d.save(p)
    text = file_to_text(str(p), 10)[1]
    assert "friendly, plain-spoken" in text and "Tagline | Honest snacks" in text


def _tiny_pdf(text: str) -> bytes:
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offsets = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % (len(objs) + 1, xref)
    return out


def test_pdf_extraction(tmp_path):
    p = tmp_path / "brand.pdf"
    p.write_bytes(_tiny_pdf("Acme sustainability report 2025"))
    assert "sustainability report" in file_to_text(str(p), 10)[1]
    with pytest.raises(ExtractError):
        bytes_to_text(b"not a pdf", "application/pdf")


# ----------------------------------------------------------------- chunking
def test_chunking_overlap_and_size():
    text = " ".join(f"Sentence number {i} talks about brand topic {i % 7}." for i in range(120))
    chunks = chunk_text(text, target_words=60, overlap_words=15)
    assert len(chunks) > 3
    assert all(len(c.split()) <= 75 for c in chunks)
    assert set(chunks[0].split()[-3:]) & set(chunks[1].split()[:20])  # overlap carried forward
    assert chunk_text("") == []
    assert chunk_text("Tiny fact here.") == ["Tiny fact here."]
    assert len(chunk_text("word " * 1000, target_words=100)) >= 10  # run-on text is hard-split


def test_individual_accounts(settings, monkeypatch):
    from agent.config import get_settings
    monkeypatch.setenv("APP_USERS", "Priya:pass-one-1,rahul:pass-two-2")
    s = get_settings()
    assert security.check_login("priya", "pass-one-1", s)
    assert security.check_login("Rahul", "pass-two-2", s)
    assert not security.check_login("priya", "pass-two-2", s)      # someone else's password
    assert not security.check_login("stranger", "pass-one-1", s)   # not in the list
    assert not security.check_login("priya", "class-secret-123", s)  # shared code no longer works
    monkeypatch.setenv("APP_USERS", "bob:short")
    import pytest
    with pytest.raises(ValueError):
        get_settings()
