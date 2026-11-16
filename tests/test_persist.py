"""Durable storage: brands and eval runs survive a wiped data folder (in-memory stand-in + fake REST session)."""
import json
import shutil

import pytest

from agent import evaluate as ev
from agent import persist
from agent.embeddings import HashEmbedder
from agent.security import FetchResult
from agent.service import Service
from tests.test_evaluate import HTML, fake_llm


def make(settings, sync):
    return Service(settings, llm_fn=fake_llm, fetch_fn=lambda url, s, session=None: FetchResult(url, "text/html", HTML),
                   embedder=HashEmbedder(), search_fn=lambda q, n: [], wiki_fn=lambda n: [], sync=sync)


def test_brands_survive_wipe_and_stay_private(settings):
    mem = persist.MemoryBackend()
    svc = make(settings, mem)
    b = svc.create_brand("priya", "Acme", "Retail")
    svc.ingest_urls("priya", b["brand_id"], ["https://good.com/about"])
    svc.create_brand("rahul", "Other", "Retail")
    shutil.rmtree(settings.data_dir / "workspaces")          # simulate a host restart
    svc2 = make(settings, mem)
    names = [x["name"] for x in svc2.list_brands("priya")]
    assert names == ["Acme"]
    assert svc2.list_documents("priya", b["brand_id"])
    assert svc2.store("priya").retrieve(b["brand_id"], "solar factories", 3, 0.5, HashEmbedder())
    assert [x["name"] for x in svc2.list_brands("rahul")] == ["Other"]


def test_delete_removes_remote_copy(settings):
    mem = persist.MemoryBackend()
    svc = make(settings, mem)
    b = svc.create_brand("priya", "Acme", "Retail")
    svc.delete_brand("priya", b["brand_id"])
    shutil.rmtree(settings.data_dir / "workspaces")
    assert make(settings, mem).list_brands("priya") == []


def test_no_backend_when_not_configured(settings):
    assert persist.from_settings(settings) is None


def test_past_runs_roundtrip_and_restore(settings):
    mem = persist.MemoryBackend()
    svc = make(settings, mem)
    list(ev.setup_brands_stream(svc, "priya"))
    rows = list(ev.evaluate_stream(svc, "priya", "gpt-5-nano", "gpt-5.4-nano", 8))[-1][2]
    params = {"gen_model": "gpt-5-nano", "judge_model": "gpt-5.4-nano", "top_k": 6, "hybrid": 0.5}
    ev.write_csv(svc, "priya", rows, params)
    shutil.rmtree(settings.data_dir / "evals")
    runs = ev.list_past_runs(make(settings, mem), "priya")
    assert len(runs) == 1 and runs[0]["scored"] == 8
    rows2, params2 = ev.load_run(runs[0]["path"])
    assert "Overall" in ev.summary_markdown(rows2, params2) and "4.00" in ev.summary_markdown(rows2, params2)


def test_restore_ignores_path_traversal(settings, tmp_path):
    mem = persist.MemoryBackend()
    mem._put("priya", "workspaces/../../evil.txt", b"x")
    assert mem.restore("priya", "workspaces", tmp_path / "root") == 0
    assert not (tmp_path / "evil.txt").exists()


class FakeResp:
    def __init__(self, data=None, status=200):
        self._d, self.status_code = data, status

    def json(self):
        return self._d

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeRest:
    """Mimics the few PostgREST calls we use."""
    def __init__(self):
        self.headers, self.table = {}, {}

    def post(self, url, json=None, **kw):
        assert "on_conflict=username,path" in url and kw["headers"]["Prefer"].startswith("resolution=merge-duplicates")
        self.table[(json["username"], json["path"])] = json["data_b64"]
        return FakeResp(status=201)

    def _match(self, params):
        u = params["username"].removeprefix("eq.")
        p = params["path"]
        for (uu, pp), v in self.table.items():
            if uu != u:
                continue
            if p.startswith("eq.") and pp == p[3:] or p.startswith("like.") and pp.startswith(p[5:-1].replace("\\_", "_")):
                yield (uu, pp), v

    def get(self, url, params=None, **kw):
        return FakeResp([{"path": k[1]} if params["select"] == "path" else {"data_b64": v} for k, v in self._match(params)])

    def delete(self, url, params=None, **kw):
        for k, _ in list(self._match(params)):
            del self.table[k]
        return FakeResp(status=204)


def test_supabase_backend_speaks_rest(settings):
    rest = FakeRest()
    be = persist.SupabaseBackend("https://x.supabase.co", "svc-key", session=rest)
    assert rest.headers["apikey"] == "svc-key" and rest.headers["Authorization"] == "Bearer svc-key"
    svc = make(settings, be)
    b = svc.create_brand("priya", "Acme", "Retail")
    svc.ingest_urls("priya", b["brand_id"], ["https://good.com/about"])
    assert ("priya", f"workspaces/{b['brand_id']}/meta.json") in rest.table
    shutil.rmtree(settings.data_dir / "workspaces")
    assert [x["name"] for x in make(settings, be).list_brands("priya")] == ["Acme"]
    svc.delete_brand("priya", b["brand_id"])
    assert not rest.table
