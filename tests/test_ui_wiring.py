"""Checks ui.py's handler wiring against a stub `gradio` (the real package is only
needed to *run* the app). Verifies every handler returns exactly as many values as
it has output components, and that the full user journey works through the handlers."""
import importlib
import json
import sys
import types

import pytest

from agent.embeddings import HashEmbedder
from agent.security import FetchResult
from agent.service import Service

WIRES: list = []


class Comp:
    def __init__(self, *a, **k):
        # Deliberately NO .value/.visible/.choices attributes: real Gradio 6 components created
        # inside a handler don't expose them, so handlers must never read them back.
        self.a, self.kw = a, k

    def __enter__(self):
        return self

    def __exit__(self, *e):
        return False

    def _wire(self, kind, fn, inputs=None, outputs=None, **kw):
        WIRES.append((kind, fn, inputs if isinstance(inputs, list) else ([inputs] if inputs else []), outputs if isinstance(outputs, list) else ([outputs] if outputs else [])))

    def click(self, *a, **k):
        self._wire("click", *a, **k)

    def change(self, *a, **k):
        self._wire("change", *a, **k)

    def load(self, *a, **k):
        self._wire("load", *a, **k)

    def queue(self, *a, **k):
        return self


class Request:
    def __init__(self, username):
        self.username = username


class GrError(Exception):
    pass


def make_stub():
    m = types.ModuleType("gradio")
    for name in ("Blocks", "Tabs", "Tab", "Row", "Accordion", "Markdown", "Textbox", "Dropdown", "Slider", "Button", "File", "CheckboxGroup"):
        setattr(m, name, type(name, (Comp,), {}))
    m.Request, m.Error = Request, GrError
    return m


@pytest.fixture
def ui(settings, monkeypatch, tmp_path):
    WIRES.clear()
    monkeypatch.setitem(sys.modules, "gradio", make_stub())
    sys.modules.pop("ui", None)
    ui_mod = importlib.import_module("ui")

    html = b"<html><head><title>Acme About</title></head><body><main><p>Acme Foods is a family owned snack maker founded in 1998 in Pune by two sisters.</p><p>Our factories run on solar power and every ingredient comes from nearby farms.</p></main></body></html>"

    def fetch(url, s, session=None):
        return FetchResult(url, "text/html", html)

    def llm(messages, model, s, max_output_tokens=0, json_mode=False):
        return json.dumps({"variations": [{"angle": "Benefit-led", "content": "Line one\n\nLine two #Acme", "sources": [1], "brand_note": "solar"},
                                          {"angle": "Story-led", "content": "Story post", "sources": [], "brand_note": ""}]})

    svc = Service(settings, llm_fn=llm, fetch_fn=fetch, embedder=HashEmbedder(), search_fn=lambda q, n: [], wiki_fn=lambda n: [])
    ui_mod.build_ui(svc)
    handlers = {fn.__name__: (fn, ins, outs) for _, fn, ins, outs in WIRES}
    return handlers, Request("priya")


def call(handlers, name, *args, req):
    fn, ins, outs = handlers[name]
    assert len(args) == len(ins), f"{name}: {len(ins)} inputs wired, {len(args)} given"
    result = fn(*args, req)
    if hasattr(result, "__next__"):  # generator handler: take the last yield
        result = list(result)[-1]
    result = result if isinstance(result, (tuple, list)) else (result,)
    assert len(result) == len(outs), f"{name}: returns {len(result)} values for {len(outs)} outputs"
    return result


def test_request_annotation_is_a_real_class(ui):
    """Gradio injects the login info only when the parameter's annotation IS gr.Request
    (a string annotation, e.g. from `from __future__ import annotations`, is ignored)."""
    import inspect

    handlers, _ = ui
    for name, (fn, _ins, _outs) in handlers.items():
        ann = inspect.signature(fn).parameters["request"].annotation
        assert ann is Request, f"{name}: 'request' annotation is {ann!r}, Gradio will not inject it"


def test_every_handler_is_wired(ui):
    handlers, _ = ui
    assert set(handlers) == {"on_load", "on_brand_change", "on_create", "on_delete_brand", "on_upload", "on_paste_urls", "on_discover",
                             "on_ingest_selected", "on_delete_doc", "on_generate", "on_industry_change", "on_eval_setup", "on_eval_run"}


def test_full_journey_through_handlers(ui, tmp_path):
    handlers, req = ui
    hdr, dd, docs, doc_dd = call(handlers, "on_load", req=req)
    assert "priya" in hdr and dd.kw["choices"] == []

    status, dd, docs, doc_dd, hdr = call(handlers, "on_create", "Acme Foods", "FMCG", "Voice: friendly and plain-spoken, never salesy at all.", req=req)
    bid = dd.kw["value"]
    assert "Created" in status and bid and "(1 docs)" in dd.kw["choices"][0][0]

    # bad input surfaces as a message, not a crash
    assert call(handlers, "on_create", "x", "", "", req=req)[0].startswith("⚠️")
    assert call(handlers, "on_paste_urls", None, "https://good.com/a", req=req)[0].startswith("⚠️")

    f = tmp_path / "facts.txt"
    f.write_text("Acme packaging is fully recyclable. Delivery vans are electric and routes are optimised daily.", encoding="utf-8")
    status, docs, doc_dd, dd = call(handlers, "on_upload", bid, [str(f)], req=req)
    assert "✅" in status and "facts.txt" in docs

    status, docs, doc_dd, dd = call(handlers, "on_paste_urls", bid, "https://good.com/about\nhttps://good.com/about2", req=req)
    assert status.count("✅") == 2

    status, checks = call(handlers, "on_discover", bid, "good.com", req=req)
    assert "candidate sources" in status and checks.kw["value"]
    status, docs, doc_dd, dd = call(handlers, "on_ingest_selected", bid, [], req=req)
    assert status.startswith("⚠️")
    status, docs, doc_dd, dd = call(handlers, "on_ingest_selected", bid, [checks.kw["choices"][0][1]], req=req)
    assert "✅" in status or "❌" in status

    out = call(handlers, "on_generate", bid, "linkedin", "Promote our sustainability work", "solar factories", 2, "gpt-5-nano", 6, 0.5, req=req)
    stats, *slots, sources, hdr = out
    assert "gpt-5-nano" in stats and len(slots) == 10
    assert slots[0].kw["visible"] and slots[1].kw["value"] == "Line one\n\nLine two #Acme"
    assert slots[2].kw["visible"] and not slots[4].kw["visible"]  # 2 variations shown, 3 hidden
    assert "S1" in sources and "generations left today" in hdr

    # failure paths return the right shape too
    assert call(handlers, "on_generate", None, "linkedin", "goal goal", "", 1, "gpt-5-nano", 6, 0.5, req=req)[0].startswith("⚠️")
    assert call(handlers, "on_generate", bid, "linkedin", "goal goal", "", 1, "gpt-evil", 6, 0.5, req=req)[0].startswith("⚠️")

    status, docs, doc_dd, dd = call(handlers, "on_delete_doc", bid, doc_dd.kw["choices"][0][1], req=req)
    assert "Removed" in status
    status, dd, docs, doc_dd = call(handlers, "on_delete_brand", bid, req=req)
    assert "deleted" in status and dd.kw["choices"] == []


def test_industry_hint_and_required(ui):
    handlers, req = ui
    assert "Covers:" in call(handlers, "on_industry_change", "BFSI", req=req)[0]
    assert "Choose the closest" in call(handlers, "on_industry_change", None, req=req)[0]
    assert "Choose an industry" in call(handlers, "on_create", "Acme Foods", "", "", req=req)[0]


def test_other_user_sees_nothing(ui):
    handlers, req = ui
    _, dd, *_ = call(handlers, "on_create", "Acme Foods", "FMCG", "", req=req)
    bid = dd.kw["value"]
    rahul = Request("rahul")
    _, dd2, docs, _ = call(handlers, "on_load", req=rahul)
    assert dd2.kw["choices"] == []
    out = call(handlers, "on_generate", bid, "linkedin", "Promote our sustainability work", "", 1, "gpt-5-nano", 6, 0.5, req=rahul)
    assert out[0].startswith("⚠️") and "Unknown brand" in out[0]
    assert "Unknown brand" in call(handlers, "on_delete_brand", bid, req=rahul)[0]


def test_unauthenticated_request_rejected(ui):
    handlers, _ = ui
    with pytest.raises(GrError):
        handlers["on_load"][0](Request(None))


def test_eval_tab_handlers(ui):
    handlers, req = ui
    setup = call(handlers, "on_eval_setup", req=req)
    assert len(setup) == 2
    run = call(handlers, "on_eval_run", "gpt-5-nano", "gpt-5-nano", 4, 6, 0.5, req=req)
    assert len(run) == 3 and "different" in (run[0] + run[1]).lower() or "model" in (run[0] + run[1]).lower()
