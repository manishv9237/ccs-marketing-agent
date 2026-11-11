"""Streamlit front end (same agent/ code as the Gradio app).

Run locally:   streamlit run streamlit_app.py
Streamlit Cloud: main file = streamlit_app.py; put OPENAI_API_KEY and APP_USERS in the app's Secrets.
"""
import logging
import os
import tempfile
import threading
from pathlib import Path

import streamlit as st

# Streamlit Cloud "Secrets" -> environment variables, BEFORE settings are read.
try:
    for _k, _v in dict(st.secrets).items():
        if isinstance(_v, (str, int, float)) and _k not in os.environ:
            os.environ[_k] = str(_v)
except Exception:  # no secrets file (local run with .env): fine
    pass

from agent import evaluate as ev  # noqa: E402
from agent.config import get_settings  # noqa: E402
from agent.extract import SUPPORTED_FILE_TYPES  # noqa: E402
from agent.generate import PLATFORMS  # noqa: E402
from agent.industries import INDUSTRY_CLUSTERS  # noqa: E402
from agent.security import check_login, normalize_username  # noqa: E402
from agent.service import USER_ERRORS, Service  # noqa: E402
from agent.store import StoreError  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
st.set_page_config(page_title="Brand Content Agent", page_icon="✍️", layout="wide")


@st.cache_resource(show_spinner=False)
def get_service() -> Service:
    svc = Service(get_settings())
    threading.Thread(target=lambda: svc.embedder, daemon=True, name="warm-embedder").start()
    return svc


svc = get_service()
s = svc.s
ss = st.session_state


# ------------------------------------------------------------------ login
def login_screen() -> None:
    st.title("Brand Content Agent")
    if not s.users and not s.access_code and not s.allow_no_auth:
        st.error("Not configured: set APP_USERS (username:password list) or APP_ACCESS_CODE in the app's secrets.")
        st.stop()
    st.caption("Log in with the username and password you were given. Your brands are private to your username." if s.users
               else "Enter a display name and the shared access code. Your brands are private to your display name.")
    with st.form("login"):
        name = st.text_input("Username")
        pw = st.text_input("Password" if s.users else "Access code", type="password")
        if st.form_submit_button("Log in", type="primary"):
            user = normalize_username(name)
            if user and check_login(name, pw, s):
                ss["user"] = user
                st.rerun()
            else:
                st.error("Incorrect username or password.")
    st.stop()


if "user" not in ss:
    if s.allow_no_auth and not s.users and not s.access_code:
        ss["user"] = "local"
    else:
        login_screen()
user = ss["user"]


# ---------------------------------------------------------------- helpers
def flash(key: str, msg: str) -> None:
    ss[f"flash_{key}"] = msg


def show_flash(key: str) -> None:
    msg = ss.pop(f"flash_{key}", None)
    if msg:
        st.markdown(msg)


def err(exc: Exception) -> str:
    return f"⚠️ {exc}"


def fmt_results(results) -> str:
    if not results:
        return "Nothing to ingest."
    lines = []
    for r in results:
        name = r.name if len(r.name) < 70 else r.name[:67] + "..."
        lines.append(f"✅ **{name}** — {r.chunks} chunks" + (f" _({r.message})_" if r.message else "") if r.ok else f"❌ **{name}** — {r.message}")
    return "\n\n".join(lines)


def safe(fn, *a, **kw):
    """Run a service call; return (result, error_message)."""
    try:
        return fn(*a, **kw), None
    except USER_ERRORS as exc:
        return None, err(exc)
    except Exception:  # unexpected: do not leak internals
        logging.exception("unexpected error")
        return None, "⚠️ Something went wrong. Please try again."


# ----------------------------------------------------------------- header
u = svc.usage(user)
sem = {True: "semantic + keyword", False: "keyword-only (embedding model unavailable)", None: "loading embedding model…"}[u["semantic_search"]]
top = st.columns([5, 1])
top[0].title("Brand Content Agent")
top[0].caption("Grounded marketing copy for LinkedIn, X, Instagram and email, written from your brand's own documents and web sources, not from guesswork.")
top[1].write(f"Signed in as **{user}**")
if top[1].button("Log out"):
    ss.clear()
    st.rerun()
st.markdown(f"Generations left today: **{u['generations_left']}** · retrieval: {sem}")

brands = svc.list_brands(user)
ids = [b["brand_id"] for b in brands]
labels = {b["brand_id"]: f"{b['name']} · {b['industry']} ({b['docs']} docs)" for b in brands}
if ss.get("sel_brand") not in ids:
    ss["sel_brand"] = ids[0] if ids else None
brand_id = None
if ids:
    brand_id = st.selectbox("Active brand", ids, index=ids.index(ss["sel_brand"]), format_func=lambda i: labels[i])
    ss["sel_brand"] = brand_id
else:
    st.info("No brands yet. Create one in the Brand tab.")

t_brand, t_know, t_gen, t_eval = st.tabs(["1 · Brand", "2 · Knowledge", "3 · Generate", "4 · Evaluation"])

# ------------------------------------------------------------------ brand
with t_brand:
    with st.form("create_brand"):
        c1, c2 = st.columns(2)
        b_name = c1.text_input("Brand name", max_chars=80, placeholder="e.g. Acme Foods")
        b_ind = c2.selectbox("Industry cluster", list(INDUSTRY_CLUSTERS), index=None, placeholder="Choose the closest cluster")
        b_notes = st.text_area("Brand voice & guidelines (optional)", max_chars=8000, height=110,
                               placeholder="Tone of voice, words you always/never use, audience, positioning, taglines…")
        create = st.form_submit_button("Create brand", type="primary")
    with st.expander("What each industry cluster covers"):
        st.markdown("\n".join(f"- **{k}**: {v}" for k, v in INDUSTRY_CLUSTERS.items()))
    if create:
        if not b_ind:
            flash("brand", err(StoreError("Choose an industry cluster.")))
        else:
            b, e = safe(svc.create_brand, user, b_name, b_ind, b_notes)
            if e:
                flash("brand", e)
            else:
                ss["sel_brand"] = b["brand_id"]
                flash("brand", f"✅ Created **{b['name']}**. Next: add knowledge in the Knowledge tab.")
        st.rerun()
    if brand_id and st.button("Delete active brand"):
        _, e = safe(svc.delete_brand, user, brand_id)
        flash("brand", e or "🗑️ Brand deleted.")
        st.rerun()
    show_flash("brand")

# -------------------------------------------------------------- knowledge
with t_know:
    st.markdown("The more real brand material here, the better and more specific the content. **Don't upload confidential documents**: this is a shared public demo.")
    if not brand_id:
        st.info("Create or select a brand first.")
    else:
        with st.expander("A · Upload documents", expanded=True):
            files = st.file_uploader(f"PDF, Word, text or Markdown ({s.max_upload_mb} MB max each)", accept_multiple_files=True,
                                     type=[t.lstrip(".") for t in SUPPORTED_FILE_TYPES])
            if st.button("Ingest files", type="primary") and files:
                tmp = Path(tempfile.mkdtemp(prefix="up-"))
                paths = []
                for f in files:
                    p = tmp / Path(f.name).name
                    p.write_bytes(f.getbuffer())
                    paths.append(str(p))
                res, e = safe(svc.ingest_files, user, brand_id, paths)
                flash("know", e or fmt_results(res))
                st.rerun()
        with st.expander("B · Discover on the web", expanded=True):
            st.caption("Searches the web and Wikipedia for the active brand and lists candidate pages. You choose what gets ingested.")
            domain = st.text_input("Brand website (optional, improves results)", placeholder="acme.com")
            if st.button("Find sources"):
                with st.spinner("Searching…"):
                    out, e = safe(svc.discover, user, brand_id, domain)
                if e:
                    flash("know", e)
                else:
                    cands, warns = out
                    ss["cands"] = {c.url: c.label for c in cands}
                    ss["cands_pre"] = ([c.url for c in cands if c.origin in ("wikipedia", "site")][:4] + [c.url for c in cands if c.origin == "search"][:3])
                    flash("know", "\n\n".join([f"⚠️ {w}" for w in warns] + [f"Found **{len(cands)}** candidate sources. Untick anything irrelevant, then press Ingest selected."
                                                                           if cands else "No sources found. Try adding the brand's website domain."]))
                st.rerun()
            if ss.get("cands"):
                picked = st.multiselect("Candidate sources", list(ss["cands"]), default=ss.get("cands_pre", []), format_func=lambda u_: ss["cands"][u_])
                if st.button("Ingest selected", type="primary"):
                    if not picked:
                        flash("know", err(ValueError("Tick at least one source.")))
                    else:
                        with st.spinner("Ingesting…"):
                            res, e = safe(svc.ingest_urls, user, brand_id, list(picked))
                        flash("know", e or fmt_results(res))
                        ss.pop("cands", None)
                    st.rerun()
        with st.expander("C · Paste URLs"):
            urls = st.text_area("One URL per line (max 15)", height=100, placeholder="https://acme.com/about")
            if st.button("Ingest URLs", type="primary"):
                lst = [x for x in urls.replace(",", "\n").splitlines() if x.strip()]
                with st.spinner("Ingesting…"):
                    res, e = safe(svc.ingest_urls, user, brand_id, lst)
                flash("know", e or fmt_results(res))
                st.rerun()
        show_flash("know")

        st.subheader("What this brand knows")
        docs, e = safe(svc.list_documents, user, brand_id)
        if e:
            st.markdown(e)
        elif not docs:
            st.caption("No knowledge yet. Upload files, paste URLs or discover sources on the web.")
        else:
            st.dataframe([{"Source": (d["source"] or d["name"])[:90], "Type": d["kind"], "Chunks": d["chunks"]} for d in docs], hide_index=True, use_container_width=True)
            dmap = {d["doc_id"]: f"{d['name'][:60]} ({d['kind']})" for d in docs}
            rc = st.columns([4, 1])
            gone = rc[0].selectbox("Remove a document", list(dmap), index=None, format_func=lambda i: dmap[i], placeholder="Pick a document")
            if rc[1].button("Remove") and gone:
                _, e = safe(svc.delete_document, user, brand_id, gone)
                flash("know", e or "🗑️ Removed.")
                st.rerun()

# --------------------------------------------------------------- generate
with t_gen:
    if not brand_id:
        st.info("Create or select a brand first.")
    else:
        c1, c2, c3 = st.columns(3)
        platform = c1.selectbox("Platform", list(PLATFORMS), format_func=lambda k: PLATFORMS[k]["label"])
        model = c2.selectbox("Model", list(s.allowed_models), index=list(s.allowed_models).index(s.default_model))
        n = c3.slider("Variations", 1, 5, 2)
        goal = st.text_area("Campaign goal", max_chars=500, height=70, placeholder="e.g. Drive sign-ups for the summer launch")
        brief = st.text_area("Brief / context (optional)", max_chars=2000, height=90, placeholder="Key message, offer, audience, dates, things to mention or avoid…")
        with st.expander("Advanced retrieval"):
            g_topk = st.slider("Sources to retrieve (top-k)", 2, 12, s.default_top_k)
            g_hybrid = st.slider("Hybrid weight (1 = semantic only, 0 = keyword only)", 0.0, 1.0, float(s.default_hybrid), 0.1)
        if st.button("Generate content", type="primary"):
            with st.spinner("Generating and fact-checking (this can take up to two minutes)…"):
                out, e = safe(svc.generate, user, brand_id, platform, goal, brief, int(n), model, int(g_topk), float(g_hybrid))
            ss["gen_out"] = {"out": out, "platform": platform} if out else None
            ss["gen_err"] = e
            st.rerun()
        if ss.get("gen_err"):
            st.markdown(ss["gen_err"])
        g = ss.get("gen_out")
        if g:
            out = g["out"]
            st.caption(f"Model **{out['model']}** · retrieval {out['retrieval_ms']:.0f} ms · generation {out['generation_ms'] / 1000:.1f} s · "
                       f"{len(out['sources'])} sources · {'hybrid' if out['semantic'] else 'keyword-only'} search")
            for v in out["variations"]:
                cites = ", ".join(f"S{x}" for x in v["sources"]) or "—"
                st.markdown(f"**Variation {v['number']} · {v['angle']}**  \n_{v['length']} · sources: {cites}_")
                if v["brand_note"]:
                    st.markdown(f"> {v['brand_note']}")
                if v.get("unsupported"):
                    st.warning("Not found in your sources (check before posting): " + "; ".join(v["unsupported"]))
                elif v.get("verified"):
                    st.success("Fact-check: claims match the sources")
                st.text_area(f"{PLATFORMS[g['platform']]['label']} post {v['number']} (select and copy)", value=v["content"], height=180, key=f"post_{v['number']}")
            with st.expander("Sources used (check every claim against these)"):
                for i, sr in enumerate(out["sources"], 1):
                    snippet = sr["text"][:320].replace("\n", " ")
                    st.markdown(f"**S{i}** · `{sr['source'][:80]}` · score {sr['score']}\n> {snippet}{'…' if len(sr['text']) > 320 else ''}")

# ------------------------------------------------------------- evaluation
with t_eval:
    st.markdown("Tests the agent on **4 brands from different sectors x 4 platforms x 4 goal types** (32 posts, each scored 1 to 5 by a *different* model as judge). "
                "Step 1 loads the brands from their official websites and Wikipedia. Step 2 runs the test and gives a CSV. "
                "**Don't click anything else while a step is running.**")
    if st.button("Step 1: Set up the 4 evaluation brands (Swiggy, Marico, HDFC Bank, Infosys)"):
        box = st.empty()
        try:
            for text in ev.setup_brands_stream(svc, user):
                box.markdown(text)
        except USER_ERRORS as exc:
            box.markdown(err(exc))
        except Exception:
            logging.exception("eval setup failed")
            box.markdown("⚠️ Something went wrong during setup. Please try again.")
    c1, c2 = st.columns(2)
    e_gen = c1.selectbox("Generator model", list(s.allowed_models), index=list(s.allowed_models).index(s.default_model), key="e_gen")
    judge_opts = list(s.allowed_models)
    judge_default = next((m for m in ("gpt-5.4-nano", "gpt-5-nano", "gpt-6-luna", "gpt-5.6-luna") if m in judge_opts and m != s.default_model), s.default_model)
    e_judge = c2.selectbox("Judge model (must differ)", judge_opts, index=judge_opts.index(judge_default), key="e_judge")
    d1, d2, d3 = st.columns(3)
    e_n = d1.slider("Test items (32 = full run)", 4, ev.MAX_ITEMS, ev.MAX_ITEMS, 4)
    e_topk = d2.slider("Sources to retrieve (top-k)", 2, 12, s.default_top_k, key="e_topk")
    e_hybrid = d3.slider("Hybrid weight (1 = semantic only, 0 = keyword only)", 0.0, 1.0, float(s.default_hybrid), 0.1, key="e_hybrid")
    if st.button("Step 2: Run evaluation", type="primary"):
        params = {"gen_model": e_gen, "judge_model": e_judge, "top_k": int(e_topk), "hybrid": float(e_hybrid)}
        bar = st.progress(0.0, text="Starting…")
        rows, problem = [], None
        try:
            for done, total, rows in ev.evaluate_stream(svc, user, e_gen, e_judge, int(e_n), int(e_topk), float(e_hybrid)):
                bar.progress(done / max(total, 1), text=f"Scored {done} of {total} items…")
        except USER_ERRORS as exc:
            problem = err(exc)
        except Exception:
            logging.exception("evaluation failed")
            problem = "⚠️ Something went wrong during the evaluation. Please try again."
        ok = [r for r in rows if not r["error"]]
        if problem:
            bar.empty()
            ss["eval_result"] = {"problem": problem}
        elif not ok:
            bar.empty()
            ss["eval_result"] = {"problem": "⚠️ No item could be scored. " + "; ".join(sorted({r["error"] for r in rows}))[:300]}
        else:
            bar.empty()
            path = ev.write_csv(svc, user, rows, params)
            ss["eval_result"] = {"msg": f"✅ Finished: {len(ok)} of {len(rows)} items scored.", "summary": ev.summary_markdown(rows, params),
                                 "csv": Path(path).read_bytes(), "name": Path(path).name}
        st.rerun()
    res = ss.get("eval_result")
    if res:
        if res.get("problem"):
            st.markdown(res["problem"])
        else:
            st.success(res["msg"])
            st.markdown(res["summary"])
            st.download_button("Download results (CSV)", res["csv"], file_name=res["name"], mime="text/csv")

st.caption("Demo build. Generated copy is a draft: verify facts against the sources before publishing. Content you add is stored on the server and may be visible to the operators.")
