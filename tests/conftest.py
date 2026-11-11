import socket

import pytest

from agent.config import get_settings


@pytest.fixture
def settings(tmp_path, monkeypatch):
    for k in ("OPENAI_API_KEY", "APP_ACCESS_CODE", "APP_USERS", "ALLOW_NO_AUTH", "EMBEDDER", "SEARCH_PROVIDER"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("APP_ACCESS_CODE", "class-secret-123")
    monkeypatch.setenv("EMBEDDER", "hash")
    return get_settings()


FAKE_DNS = {
    "example.com": "93.184.216.34",
    "brand.test": "203.0.113.10",  # documentation range is not "global" -> use a real-looking one below
    "good.com": "8.8.8.8",
    "evil.com": "10.0.0.5",
    "meta.evil.com": "169.254.169.254",
    "loop.evil.com": "127.0.0.1",
    "v6.evil.com": "::ffff:10.1.2.3",
    "v6loop.evil.com": "::1",
}


@pytest.fixture(autouse=True)
def fake_dns(monkeypatch):
    def fake_getaddrinfo(host, port, *a, **k):
        if host in FAKE_DNS:
            ip = FAKE_DNS[host]
            fam = socket.AF_INET6 if ":" in ip else socket.AF_INET
            return [(fam, socket.SOCK_STREAM, 6, "", (ip, port or 0))]
        if host.replace(".", "").isdigit():
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (host, port or 0))]
        raise socket.gaierror("no such host")

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
