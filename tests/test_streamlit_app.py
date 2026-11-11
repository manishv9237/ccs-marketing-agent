"""Runs streamlit_app.py against a tiny stand-in for Streamlit (the real package may be absent in CI).
It checks the script's own logic and its calls into the service, not Streamlit's rendering."""
import json
import runpy
import sys
import types
from pathlib import Path

import pytest

from agent.embeddings import HashEmbedder
from agent.security import FetchResult
from agent.service import Service

ROOT = Path(__file__).resolve().parent.parent
HTML = (b"<html><head><title>About</title></head><body><main><p>Acme is a family owned maker founded in 1998 in Pune by two sisters.</p>"
        b"<p>Our factories run on solar power and every ingredient comes from nearby farms across the state.</p></main></body></html>")


class Stop(Exception):
    pass


class Rerun(Exception):
    pass


class Ctx:
    def __init__(self, st):
        self._st = st

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):
        return getattr(self._st, name)


class FakeSt:
    def __init__(self):
        self.session_state = {}
        self.secrets = {}
        self.values, self.clicks, self.log = {}, set(), []

    # -- control flow
    def stop(self):
        raise Stop

    def rerun(self):
        raise Rerun

    def cache_resource(self, **kw):
        def deco(fn):
            box = []
            return lambda: box[0] if box else (box.append(fn()) or box[0])
        return deco

    # -- containers
    def columns(self, spec, **kw):
        return [Ctx(self) for _ in range(spec if isinstance(spec, int) else len(spec))]

    def tabs(self, labels):
        return [Ctx(self) for _ in labels]

    def form(self, *a, **kw):
        return Ctx(self)

    expander = spinner = empty = lambda self, *a, **kw: Ctx(self)

    def progress(self, *a, **kw):
        return Ctx(self)

    # -- inputs
    def text_input(self, label, **kw):
        return self.values.get(label, "")

    def text_area(self, label, value="", **kw):
        return self.values.get(label, value)

    def selectbox(self, label, options, index=0, **kw):
        if label in self.values:
            return self.values[label]
        opts = list(options)
        return opts[index] if opts and index is not None else None

    def multiselect(self, label, options, default=None, **kw):
        return self.values.get(label, default or [])

    def slider(self, label, lo, hi, value=None, step=None, **kw):
        return self.values.get(label, value)

    def file_uploader(self, *a, **kw):
        return []

    def button(self, label, *a, **kw):
        if label in self.clicks:
            self.clicks.discard(label)
            return True
        return False

    form_submit_button = button

    def __getattr__(self, name):  # markdown, caption, title, info, success, ... record text
        def rec(*a, **kw):
            self.log.append(" ".join(str(x) for x in a if isinstance(x, str)))
        return rec


def run_script(fake, monkeypatch, svc):
    mod = types.ModuleType("streamlit")
    mod.__dict__.update({k: getattr(fake, k) for k in dir(fake) if not k.startswith("_")})
    mod.__getattr__ = lambda name: getattr(fake, name)
    monkeypatch.setitem(sys.modules, "streamlit", mod)
    monkeypatch.setattr("agent.service.Service", lambda settings: svc)
    for _ in range(6):  # a click triggers st.rerun(): run again until the script settles
        try:
            runpy.run_path(str(ROOT / "streamlit_app.py"), run_name="__main__")
            return
        except Rerun:
            continue
        except Stop:
            return


def fake_llm(messages, model, s, max_output_tokens=0, json_mode=False):
    if messages[0]["content"].startswith("You are a strict, impartial evaluator"):
        return json.dumps({"grounded": 4, "on_brand": 5, "goal_relevance": 3, "platform_fit": 4, "unsupported_claims": [], "reason": "fine"})
    return json.dumps({"variations": [{"angle": "Benefit-led", "content": "Solar powered factories. #Acme", "sources": [1], "brand_note": "tone ok"}]})


@pytest.fixture
def svc(settings):
    return Service(settings, llm_fn=fake_llm, fetch_fn=lambda url, s, session=None: FetchResult(url, "text/html", HTML),
                   embedder=HashEmbedder(), search_fn=lambda q, n: [], wiki_fn=lambda n: [])


def test_login_required_then_full_journey(svc, monkeypatch):
    monkeypatch.delenv("ALLOW_NO_AUTH", raising=False)
    fake = FakeSt()
    # no login configured in the test settings -> the script must refuse, not open the app
    import dataclasses
    svc.s = dataclasses.replace(svc.s, users={"priya": "secret1"}, access_code="", allow_no_auth=False)
    run_script(fake, monkeypatch, svc)
    assert "user" not in fake.session_state
    fake.values = {"Username": "priya", "Password": "wrong"}
    fake.clicks = {"Log in"}
    run_script(fake, monkeypatch, svc)
    assert "user" not in fake.session_state and any("Incorrect" in x for x in fake.log)
    fake.values["Password"] = "secret1"
    fake.clicks = {"Log in"}
    run_script(fake, monkeypatch, svc)
    assert fake.session_state["user"] == "priya"

    # create a brand
    fake.values.update({"Brand name": "Acme Foods", "Industry cluster": "FMCG & Consumer Brands"})
    fake.clicks = {"Create brand"}
    run_script(fake, monkeypatch, svc)
    brands = svc.list_brands("priya")
    assert [b["name"] for b in brands] == ["Acme Foods"] and fake.session_state["sel_brand"] == brands[0]["brand_id"]

    # paste a URL
    fake.values["One URL per line (max 15)"] = "https://acme.com/about"
    fake.clicks = {"Ingest URLs"}
    run_script(fake, monkeypatch, svc)
    assert svc.list_documents("priya", brands[0]["brand_id"])

    # generate
    fake.values["Campaign goal"] = "Announce the solar factory"
    fake.clicks = {"Generate content"}
    run_script(fake, monkeypatch, svc)
    assert fake.session_state["gen_out"]["out"]["variations"][0]["content"].startswith("Solar powered")
    assert not fake.session_state.get("gen_err")

    # evaluation: setup then run a small balanced subset
    fake.clicks = {"Step 1: Set up the 4 evaluation brands (Swiggy, Marico, HDFC Bank, Infosys)"}
    run_script(fake, monkeypatch, svc)
    assert len(svc.list_brands("priya")) == 5
    fake.values["Test items (32 = full run)"] = 8
    fake.clicks = {"Step 2: Run evaluation"}
    run_script(fake, monkeypatch, svc)
    res = fake.session_state["eval_result"]
    assert res.get("msg", "").startswith("✅ Finished: 8 of 8") and b"grounded" in res["csv"]

    # log out clears the session
    fake.clicks = {"Log out"}
    run_script(fake, monkeypatch, svc)
    assert "user" not in fake.session_state
