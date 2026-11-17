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
from agent.accounts import AccountError  # noqa: E402
from agent.limits import LimitReached  # noqa: E402
from agent.security import check_login, normalize_username  # noqa: E402
from agent.service import USER_ERRORS, Service  # noqa: E402
from agent.store import StoreError  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
st.set_page_config(page_title="BrandVoice AI", page_icon="💬", layout="wide")


@st.cache_resource(show_spinner=False)
def get_service() -> Service:
    svc = Service(get_settings())
    threading.Thread(target=lambda: svc.embedder, daemon=True, name="warm-embedder").start()
    return svc


svc = get_service()
s = svc.s
ss = st.session_state


# ------------------------------------------------------------------ look and feel
# Hide Streamlit's whole toolbar (Share, star, edit, GitHub, three-dot menu and its Print/Record items).
# The light/dark choice is offered on the page itself (theme_switcher below), which drives Streamlit's own
# theme setting through the hidden menu. One flush-left line so Markdown never shows the CSS as text.
_CSS = ("<style>"
        "[data-testid=\"stToolbarActions\"], [data-testid=\"stAppDeployButton\"], [data-testid=\"stDecoration\"], .stAppDeployButton, footer {display: none !important;} "
        "[data-testid=\"stMainMenu\"] {visibility: hidden !important; position: absolute !important;} "
        "[data-baseweb=\"popover\"]:has([data-testid^=\"stMainMenuItem\"]) {opacity: 0 !important; pointer-events: none !important;} "
        ".block-container {padding-top: 3.6rem; max-width: 1080px;} "
        "h1, h2, h3 {letter-spacing: -0.01em;} "
        ".bca-kicker {font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase; opacity: 0.6; margin: 0 0 0.15rem 0;} "
        ".bca-title {font-size: 2rem; font-weight: 700; line-height: 1.15; margin: 0 0 0.2rem 0;} "
        ".bca-sub {opacity: 0.75; margin: 0 0 0.9rem 0;} "
        ".bca-steps {display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 0.2rem 0 1rem 0;} "
        ".bca-step {border: 1px solid rgba(128,128,128,0.35); border-radius: 999px; padding: 0.18rem 0.75rem; font-size: 0.82rem;} "
        ".bca-step.done {border-color: #2e8b57; color: #2e8b57; font-weight: 600;} "
        ".bca-step.now {border-color: #5b4bdb; color: #5b4bdb; font-weight: 600;} "
        ".bca-strip {display: flex; gap: 1.4rem; flex-wrap: wrap; border: 1px solid rgba(128,128,128,0.3); border-radius: 12px; padding: 0.6rem 1rem; margin: 0.4rem 0 1rem 0; font-size: 0.85rem;} "
        ".bca-strip b {display: block; font-size: 1.05rem;} "
        ".bca-pill {display: inline-block; border-radius: 999px; padding: 0.05rem 0.6rem; font-size: 0.75rem; font-weight: 600;} "
        ".bca-good {background: rgba(46,139,87,0.16); color: #2e8b57;} "
        ".bca-ok {background: rgba(214,137,16,0.16); color: #d68910;} "
        ".bca-thin {background: rgba(192,57,43,0.16); color: #5b4bdb;} "
        "div[data-testid=\"stForm\"], div[data-testid=\"stExpander\"] {border-radius: 12px !important;} "
        "div[data-testid=\"stTabs\"] button[role=\"tab\"] {font-weight: 600; padding: 0.6rem 1rem;} "
        ".stButton > button, .stDownloadButton > button, [data-testid=\"stFormSubmitButton\"] > button {border-radius: 10px; font-weight: 600;} "
        "textarea {border-radius: 10px !important;} "
        "[data-testid=\"stChatMessage\"] {border: 1px solid rgba(128,128,128,0.25); border-radius: 16px; padding: 0.8rem 1rem; margin-bottom: 0.7rem;} "
        "[data-testid=\"stChatInput\"] {border-radius: 16px;}"
        "</style>")
st.markdown(_CSS, unsafe_allow_html=True)

try:
    from streamlit.components.v1 import html as _components_html  # noqa: E402
except Exception:  # not available in the test stand-in
    def _components_html(*a, **k):
        return None

_THEME_JS = """<script>(async () => {
  const d = window.parent.document, sleep = t => new Promise(r => setTimeout(r, t));
  const btn = d.querySelector('[data-testid="stMainMenuButton"]'); if (!btn) return;
  btn.click();
  for (let i = 0; i < 20; i++) {
    const it = d.querySelector('[data-testid="stMainMenuItem-theme-__T__"]');
    if (it) { it.click(); break; }
    await sleep(100);
  }
  await sleep(250); btn.click();
})();</script>"""


def theme_switcher() -> None:
    """System / Light / Dark shown directly on the page (no three-dot menu)."""
    opts = ["System", "Light", "Dark"]
    choice = st.radio("Theme", opts, index=opts.index(ss.get("theme_applied", "System")), horizontal=True, label_visibility="collapsed") or "System"
    if ss.get("theme_applied", "System") != choice:
        ss["theme_applied"] = choice
        _components_html(_THEME_JS.replace("__T__", choice), height=0)


# ------------------------------------------------------------------ login / sign up
def _open_session(uid: str, display: str) -> None:
    ss["user"] = uid
    ss["display"] = display
    st.rerun()


def login_screen() -> None:
    can_login_env = bool(s.users or s.access_code)
    if not can_login_env and not s.signup_enabled and not s.allow_no_auth:
        st.error("Not configured: enable sign-up or set APP_USERS / APP_ACCESS_CODE in the app's secrets.")
        st.stop()
    _sp, _th = st.columns([3, 2])
    with _th:
        theme_switcher()
    left, mid, right = st.columns([1, 2, 1])
    with mid:
        st.markdown('<p class="bca-kicker">BrandVoice AI</p><p class="bca-title">BrandVoice AI</p>'
                    '<p class="bca-sub">Your AI partner for smarter, on-brand marketing</p>', unsafe_allow_html=True)
        tab_in, tab_up = st.tabs(["Log in", "Sign up"]) if s.signup_enabled else (st.container(), None)
        with tab_in:
            with st.form("login"):
                ident = st.text_input("Email or username")
                pw = st.text_input("Password", type="password")
                if st.form_submit_button("Log in", type="primary"):
                    ident = (ident or "").strip()
                    if "@" in ident:
                        acct = svc.accounts.verify(ident, pw)
                        if acct:
                            _open_session(acct["id"], acct["name"])
                    else:
                        uname = normalize_username(ident)
                        if uname and check_login(ident, pw, s):
                            _open_session(uname, uname)
                    st.error("Incorrect email/username or password.")
        if tab_up is not None:
            with tab_up:
                with st.form("signup"):
                    c1, c2 = st.columns(2)
                    first = c1.text_input("First name")
                    last = c2.text_input("Last name")
                    email = st.text_input("Email address")
                    pw1 = st.text_input("Create password (at least 8 characters)", type="password")
                    pw2 = st.text_input("Confirm password", type="password")
                    code = st.text_input("Invite code") if s.signup_code else ""
                    if st.form_submit_button("Sign up", type="primary"):
                        acct = None
                        try:
                            svc.limiter.consume("_signup", "signup", s.signup_per_day)  # blunt abuse cap
                            try:
                                acct = svc.accounts.sign_up(first, last, email, pw1, pw2, code)
                            except AccountError:
                                svc.limiter.refund("_signup", "signup")
                                raise
                        except (AccountError, LimitReached) as exc:
                            st.error(str(exc))
                        if acct:
                            _open_session(acct["id"], acct["name"])
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
_sp2, _th2 = st.columns([4, 2])
with _th2:
    theme_switcher()
top = st.columns([5, 1])
top[0].markdown('<p class="bca-kicker">Your AI partner for smarter, on-brand marketing</p><p class="bca-title">BrandVoice AI</p>'
                '<p class="bca-sub">Grounded marketing copy for LinkedIn, X, Instagram and email, written from your brand\'s own documents and web sources, not from guesswork.</p>',
                unsafe_allow_html=True)
top[1].write(f"Signed in as **{ss.get('display', user)}**")
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

_docs_now, _ = safe(svc.list_documents, user, brand_id) if brand_id else (None, None)
_n_docs = len(_docs_now or [])
_s1 = "done" if brand_id else "now"
_s2 = "done" if _n_docs else ("now" if brand_id else "")
_s3 = "now" if (brand_id and _n_docs) else ""
st.markdown('<div class="bca-steps">'
            f'<span class="bca-step {_s1}">{"✓ " if brand_id else "1 "}Brand kit</span>'
            f'<span class="bca-step {_s2}">{"✓ " if _n_docs else "2 "}Source library</span>'
            f'<span class="bca-step {_s3}">3 Chat</span></div>', unsafe_allow_html=True)

t_gen, t_brand, t_know, t_eval = st.tabs(["Chat", "Brand kit", "Source library", "Evaluation"])

# ------------------------------------------------------------------ brand
with t_brand:
    with st.form("create_brand"):
        c1, c2 = st.columns(2)
        b_name = c1.text_input("Brand name", max_chars=80, placeholder="e.g. Acme Foods")
        b_ind = c2.selectbox("Industry cluster", list(INDUSTRY_CLUSTERS), index=None, placeholder="Choose the closest cluster")
        k1, k2 = st.columns(2)
        b_aud = k1.text_input("Target audience", max_chars=200, placeholder="e.g. urban families, 25-40")
        b_pos = k2.text_input("Positioning in one line", max_chars=200, placeholder="e.g. everyday health, made simple")
        t1, t2, t3 = st.columns(3)
        tone_a = t1.select_slider("Tone: register", ["Formal", "Balanced", "Casual"], value="Balanced")
        tone_b = t2.select_slider("Tone: energy", ["Reserved", "Balanced", "Bold"], value="Balanced")
        tone_c = t3.select_slider("Tone: feel", ["Factual", "Balanced", "Emotive"], value="Balanced")
        b_avoid = st.text_input("Words to avoid", max_chars=200, placeholder="comma separated, e.g. cheap, guaranteed")
        b_notes = st.text_area("Brand voice & guidelines (optional)", max_chars=8000, height=110,
                               placeholder="Tone of voice, words you always/never use, audience, positioning, taglines…")
        create = st.form_submit_button("Create brand", type="primary")
    with st.expander("What each industry cluster covers"):
        st.markdown("\n".join(f"- **{k}**: {v}" for k, v in INDUSTRY_CLUSTERS.items()))
    if create:
        if not b_ind:
            flash("brand", err(StoreError("Choose an industry cluster.")))
        else:
            _kit = [f"Target audience: {b_aud}" if b_aud else "", f"Positioning: {b_pos}" if b_pos else "",
                    "Tone: " + ", ".join(x for x in (tone_a, tone_b, tone_c) if x and x != "Balanced") if any(x and x != "Balanced" for x in (tone_a, tone_b, tone_c)) else "",
                    f"Words to avoid: {b_avoid}" if b_avoid else "", b_notes or ""]
            b, e = safe(svc.create_brand, user, b_name, b_ind, "\n".join(x for x in _kit if x))
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

        st.subheader("Source library")
        st.caption("Content can only claim what these sources say. Status shows where the library is thin before you generate.")
        docs, e = safe(svc.list_documents, user, brand_id)
        if e:
            st.markdown(e)
        elif not docs:
            st.caption("No knowledge yet. Upload files, paste URLs or discover sources on the web.")
        else:
            def _status(n):
                return "Good" if n >= 20 else ("Adequate" if n >= 8 else "Thin")
            st.dataframe([{"Source": (d["source"] or d["name"])[:90], "Type": d["kind"], "Chunks": d["chunks"], "Status": _status(d["chunks"])} for d in docs],
                         hide_index=True, use_container_width=True)
            _total = sum(d["chunks"] for d in docs)
            st.markdown(f'Total **{_total}** chunks from **{len(docs)}** sources &nbsp; <span class="bca-pill bca-{ {"Good": "good", "Adequate": "ok", "Thin": "thin"}[_status(_total)] }">{_status(_total)}</span>', unsafe_allow_html=True)
            if _total < 20:
                st.warning("Thin library: very little to ground claims on. Add more documents or pages, or expect cautious, general copy.")
            dmap = {d["doc_id"]: f"{d['name'][:60]} ({d['kind']})" for d in docs}
            rc = st.columns([4, 1])
            gone = rc[0].selectbox("Remove a document", list(dmap), index=None, format_func=lambda i: dmap[i], placeholder="Pick a document")
            if rc[1].button("Remove") and gone:
                _, e = safe(svc.delete_document, user, brand_id, gone)
                flash("know", e or "🗑️ Removed.")
                st.rerun()

# ------------------------------------------------------------------- chat
def render_out(out: dict, platform: str, key: str) -> None:
    st.caption(f"Model **{out['model']}** · retrieval {out['retrieval_ms']:.0f} ms · generation {out['generation_ms'] / 1000:.1f} s · "
               f"{len(out['sources'])} sources · {'hybrid' if out['semantic'] else 'keyword-only'} search")
    _ver = sum(1 for v in out["variations"] if v.get("verified") and not v.get("unsupported"))
    st.markdown(f'**{_ver} of {len(out["variations"])}** variations fully grounded in your sources'
                + (" · review the flagged ones before posting" if _ver < len(out["variations"]) else ""))
    for v in out["variations"]:
        cites = ", ".join(f"S{x}" for x in v["sources"]) or "—"
        st.markdown(f"**Variation {v['number']} · {v['angle']}**  \n_{v['length']} · sources: {cites}_")
        if v["brand_note"]:
            st.markdown(f"> {v['brand_note']}")
        if v.get("unsupported"):
            st.warning("Not found in your sources (check before posting): " + "; ".join(v["unsupported"]))
        elif v.get("verified"):
            st.success("Fact-check: claims match the sources")
        st.text_area(f"{PLATFORMS[platform]['label']} post {v['number']} (select and copy)", value=v["content"], height=180, key=f"{key}_post_{v['number']}")
    with st.expander("Sources used (check every claim against these)"):
        for i, sr in enumerate(out["sources"], 1):
            snippet = sr["text"][:320].replace("\n", " ")
            st.markdown(f"**S{i}** · `{sr['source'][:80]}` · score {sr['score']}\n> {snippet}{'…' if len(sr['text']) > 320 else ''}")


with t_gen:
    with st.chat_message("assistant"):
        if not brand_id:
            st.markdown("Hi, I'm **BrandVoice AI**. Let's start by creating your brand in the **Brand kit** tab, then I'll help you write on-brand posts.")
        elif not _n_docs:
            st.markdown("Brand is ready. Next, add some sources in the **Source library** tab (documents, web pages or URLs) so I can ground every claim.")
        else:
            st.markdown("Hi, I'm **BrandVoice AI**. Pick a platform below, then tell me the campaign goal in the box at the bottom, for example: "
                        "*Drive sign-ups for the summer launch*. I'll write variations from your sources and fact-check them.")
    if brand_id:
        with st.expander("Chat settings"):
            c1, c2, c3 = st.columns(3)
            platform = c1.selectbox("Platform", list(PLATFORMS), format_func=lambda k: PLATFORMS[k]["label"])
            model = c2.selectbox("Model", list(s.allowed_models), index=list(s.allowed_models).index(s.default_model))
            n = c3.slider("Variations", 1, 5, 2)
            brief = st.text_area("Brief / context (optional)", max_chars=2000, height=90, placeholder="Key message, offer, audience, dates, things to mention or avoid…")
            g_topk = st.slider("Sources to retrieve (top-k)", 2, 12, s.default_top_k)
            g_hybrid = st.slider("Hybrid weight (1 = semantic only, 0 = keyword only)", 0.0, 1.0, float(s.default_hybrid), 0.1)
        hist = ss.setdefault("chat", {}).setdefault(brand_id, [])
        for i, m in enumerate(hist):
            with st.chat_message(m["role"]):
                if m["role"] == "user":
                    st.markdown(m["text"])
                elif m.get("error"):
                    st.markdown(m["error"])
                else:
                    render_out(m["out"], m["platform"], f"h{i}")
        prompt = st.chat_input("Describe your campaign goal, e.g. Drive sign-ups for the summer launch")
        if prompt:
            hist.append({"role": "user", "text": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Writing and fact-checking (this can take up to two minutes)…"):
                    out, e = safe(svc.generate, user, brand_id, platform, prompt, brief, int(n), model, int(g_topk), float(g_hybrid))
            if out:
                hist.append({"role": "assistant", "out": out, "platform": platform})
            else:
                hist.append({"role": "assistant", "error": e or "⚠️ Nothing was generated."})
            ss["gen_out"] = {"out": out, "platform": platform} if out else None
            ss["gen_err"] = e
            st.rerun()
        if hist and st.button("New chat"):
            hist.clear()
            st.rerun()

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

    st.divider()
    with st.expander("Past runs (saved results)"):
        runs = ev.list_past_runs(svc, user)
        if not runs:
            st.caption("No saved runs yet. Finished runs are saved automatically.")
        else:
            pick = st.selectbox("Run", runs, format_func=lambda r: f"{r['name']} · top-k {r['params']['top_k']} · hybrid {r['params']['hybrid']} · {r['scored']} scored", key="past_run")
            rows_p, params_p = ev.load_run(pick["path"])
            st.markdown(ev.summary_markdown(rows_p, params_p))
            st.download_button("Download this run (CSV)", Path(pick["path"]).read_bytes(), file_name=pick["name"], mime="text/csv", key="past_dl")

st.caption("Demo build. Generated copy is a draft: verify facts against the sources before publishing. Content you add is stored on the server and may be visible to the operators.")
