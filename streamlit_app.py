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
from agent import review  # noqa: E402
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
        "[data-testid=\"stMainMenu\"] {display: none !important;} "
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
        ".st-key-topnav {background: #1c1a2e; border-radius: 12px; padding: 0.35rem 0.8rem; margin-bottom: 1.1rem;} "
        ".st-key-topnav [data-testid=\"stRadio\"] > label {display: none;} "
        ".st-key-topnav div[role=\"radiogroup\"] {display: flex; flex-wrap: wrap; align-items: center; gap: 0.2rem; width: 100%;} "
        ".st-key-topnav div[role=\"radiogroup\"]::before {content: \"BrandVoice AI\"; font-weight: 700; color: #ffffff; margin-right: 1.2rem; letter-spacing: 0.01em;} "
        ".st-key-topnav label[data-baseweb=\"radio\"] {margin: 0; padding: 0.3rem 0.8rem; border-radius: 8px; cursor: pointer;} "
        ".st-key-topnav label[data-baseweb=\"radio\"] > div:first-child {display: none;} "
        ".stApp .st-key-topnav label[data-baseweb=\"radio\"] p {color: #cfcbe8 !important; font-weight: 500; margin: 0;} "
        ".st-key-topnav label[data-baseweb=\"radio\"]:has(input:checked) {background: #5b4bdb;} "
        ".stApp .st-key-topnav label[data-baseweb=\"radio\"]:has(input:checked) p {color: #ffffff !important; font-weight: 600;} "
        ".st-key-topnav div[role=\"radiogroup\"] > label:last-child {margin-left: auto;} "
        ".bca-swatch {display: inline-block; width: 22px; height: 22px; border-radius: 6px; border: 1px solid rgba(128,128,128,0.4); margin-right: 6px; vertical-align: middle;} "
        "[data-testid=\"stChatMessage\"] {border: 1px solid rgba(128,128,128,0.25); border-radius: 16px; padding: 0.8rem 1rem; margin-bottom: 0.7rem;} "
        "[data-testid=\"stChatInput\"] {border-radius: 16px;}"
        "</style>")
st.markdown(_CSS, unsafe_allow_html=True)

_DARK = ("<style>:root {color-scheme: dark;} .stApp, [data-testid=\"stAppViewContainer\"], [data-testid=\"stHeader\"], [data-testid=\"stMain\"] {background-color: #0e1117 !important; color: #fafafa !important;} "
         ".stApp p, .stApp span, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp div[data-testid=\"stMarkdownContainer\"], .stApp div[data-testid=\"stCaptionContainer\"] {color: #fafafa !important;} "
         ".stApp input, .stApp textarea, .stApp [data-baseweb=\"select\"] > div, .stApp [data-baseweb=\"input\"], .stApp [data-baseweb=\"textarea\"] {background-color: #262730 !important; color: #fafafa !important;} "
         ".stApp [data-baseweb=\"popover\"] *, .stApp ul[role=\"listbox\"] {background-color: #262730 !important; color: #fafafa !important;} "
         "div[data-testid=\"stForm\"], div[data-testid=\"stExpander\"], [data-testid=\"stChatMessage\"], .bca-strip, .bca-step {border-color: rgba(250,250,250,0.25) !important;} "
         "[data-testid=\"stChatMessage\"] {background-color: #161a23 !important;} [data-testid=\"stBottom\"], [data-testid=\"stBottomBlockContainer\"] {background-color: #0e1117 !important;} "
         "[data-testid=\"stAlert\"] {background-color: #1b2030 !important;} button[kind=\"secondary\"] {background-color: #262730 !important; color: #fafafa !important; border-color: rgba(250,250,250,0.25) !important;}</style>")
_LIGHT = ("<style>:root {color-scheme: light;} .stApp, [data-testid=\"stAppViewContainer\"], [data-testid=\"stHeader\"], [data-testid=\"stMain\"] {background-color: #ffffff !important; color: #1f2328 !important;} "
          ".stApp p, .stApp span, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp div[data-testid=\"stMarkdownContainer\"], .stApp div[data-testid=\"stCaptionContainer\"] {color: #1f2328 !important;} "
          ".stApp input, .stApp textarea, .stApp [data-baseweb=\"select\"] > div, .stApp [data-baseweb=\"input\"], .stApp [data-baseweb=\"textarea\"] {background-color: #f6f6f4 !important; color: #1f2328 !important;} "
          ".stApp [data-baseweb=\"popover\"] *, .stApp ul[role=\"listbox\"] {background-color: #ffffff !important; color: #1f2328 !important;} "
          "[data-testid=\"stBottom\"], [data-testid=\"stBottomBlockContainer\"] {background-color: #ffffff !important;}</style>")


def theme_switcher() -> None:
    """System / Light / Dark shown on the page. Applied with CSS here, so there is no popup and no dependence on Streamlit's own menu."""
    opts = ["System", "Light", "Dark"]
    choice = st.radio("Theme", opts, index=opts.index(ss.get("theme_applied", "System")), horizontal=True, label_visibility="collapsed", key="theme_choice") or "System"
    ss["theme_applied"] = choice
    apply_theme()


def apply_theme() -> None:
    choice = ss.get("theme_applied", "System")
    if choice == "Dark":
        st.markdown(_DARK, unsafe_allow_html=True)
    elif choice == "Light":
        st.markdown(_LIGHT, unsafe_allow_html=True)


# ------------------------------------------------------------------ login / sign up
def _open_session(uid: str, display: str, full: str = "") -> None:
    ss["user"] = uid
    ss["display"] = display
    ss["full"] = full or display
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
        st.markdown('<p class="bca-title">BrandVoice AI</p>'
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
                            _open_session(acct["id"], acct["name"], acct.get("full_name", ""))
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
                            _open_session(acct["id"], acct["name"], acct.get("full_name", ""))
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



# ------------------------------------------------------------------- chat
_STATUS_ICON = {"grounded": "🟢", "flagged": "🔴", "unchecked": "⚪"}
_RECIPE_KEYS = ("g_chan", "g_model", "g_n", "g_brief", "g_angles", "g_topk", "g_hyb")


def _apply_recipe(data: dict) -> None:
    for k in _RECIPE_KEYS:
        if k in data:
            ss[k] = data[k]


def _do_rewrite(key: str, v: dict, out: dict, platform: str, brand_id_: str) -> None:
    instr = ss.get(f"{key}_instr", "")
    new, e = safe(svc.rewrite, user, brand_id_, platform, v, instr, out["sources"], out["model"])
    if e:
        ss[f"{key}_rw_err"] = e
        return
    new["history"] = list(v.get("history", [])) + [v["content"]]
    v.clear()
    v.update(new)
    ss.pop(f"{key}_rw_err", None)


def render_out(out: dict, platform: str, key: str, brand_id_: str) -> None:
    st.caption(f"{PLATFORMS[platform]['label']} · model **{out['model']}** · retrieval {out['retrieval_ms']:.0f} ms · generation {out['generation_ms'] / 1000:.1f} s · "
               f"{len(out['sources'])} sources · {'hybrid' if out['semantic'] else 'keyword-only'} search")
    n_ok = sum(1 for v in out["variations"] if v.get("verified") and not v.get("unsupported"))
    picked = 0
    for v in out["variations"]:
        vk = f"{key}_{v['number']}"
        cites = ", ".join(f"S{x}" for x in v["sources"]) or "—"
        st.markdown(f"**Variation {v['number']} · {v['angle']}**  \n_{v['length']} · sources: {cites}_")
        if v["brand_note"]:
            st.markdown(f"> {v['brand_note']}")
        paras = review.paragraph_status(v["content"], v.get("unsupported", []), bool(v.get("verified")))
        n_g = sum(1 for p in paras if p["status"] == "grounded")
        if paras and any(p["status"] != "unchecked" for p in paras):
            st.markdown(f"**Verification** · {n_g} of {len(paras)} paragraphs grounded")
            for p in paras:
                st.markdown(f"{_STATUS_ICON[p['status']]} {p['text']}")
        if v.get("unsupported"):
            st.warning("Not found in your sources (check before posting): " + "; ".join(v["unsupported"]))
        elif v.get("verified"):
            st.success("Fact-check: claims match the sources")
        st.text_area(f"{PLATFORMS[platform]['label']} post {v['number']} (select and copy)", value=v["content"], height=180, key=f"{vk}_post")
        with st.expander("Sources behind this piece"):
            for i in v["sources"]:
                if 1 <= i <= len(out["sources"]):
                    sr = out["sources"][i - 1]
                    st.markdown(f"**S{i}** · `{sr['source'][:80]}`\n> {sr['text'][:320].replace(chr(10), ' ')}{'…' if len(sr['text']) > 320 else ''}")
        with st.expander("Guided rewrite"):
            st.text_input("Describe the change (e.g. shorter and friendlier)", key=f"{vk}_instr", max_chars=300)
            st.button("Rewrite", key=f"{vk}_rw", on_click=_do_rewrite, args=(vk, v, out, platform, brand_id_))
            if ss.get(f"{vk}_rw_err"):
                st.markdown(ss[f"{vk}_rw_err"])
            if v.get("history"):
                st.caption(f"History: {len(v['history'])} earlier version(s)")
                for j, old in enumerate(v["history"], 1):
                    st.text_area(f"Version {j}", value=old, height=100, key=f"{vk}_hist_{j}")
        if st.checkbox("Pick this one", key=f"{vk}_pick"):
            picked += 1
    st.caption(f"{n_ok} of {len(out['variations'])} fully grounded · {picked} picked · review the flagged paragraphs before posting")
    with st.expander("Sources used (check every claim against these)"):
        for i, sr in enumerate(out["sources"], 1):
            snippet = sr["text"][:320].replace("\n", " ")
            st.markdown(f"**S{i}** · `{sr['source'][:80]}` · score {sr['score']}\n> {snippet}{'…' if len(sr['text']) > 320 else ''}")



# ----------------------------------------------------------------- shell (top bar + pages)
u = svc.usage(user)
sem = {True: "semantic + keyword", False: "keyword-only", None: "loading embedding model…"}[u["semantic_search"]]
brands = svc.list_brands(user)
ids = [b["brand_id"] for b in brands]
labels = {b["brand_id"]: f"{b['name']} · {b['industry']}" for b in brands}
if ss.get("sel_brand") not in ids:
    ss["sel_brand"] = ids[0] if ids else None

PAGES = ["Add brand", "Sources", "Brand kit", "Brief", "Account"]
if "goto" in ss:
    ss["nav"] = ss.pop("goto")
if ss.get("nav") not in PAGES:
    ss["nav"] = "Brief" if ids else "Add brand"
full_name = ss.get("full") or ss.get("display") or user
with st.container(key="topnav"):
    page = st.radio("Navigation", PAGES, horizontal=True, key="nav", label_visibility="collapsed",
                    format_func=lambda p: full_name if p == "Account" else p) or PAGES[0]


def brand_picker(label: str = "Brand") -> str | None:
    """Dropdown of all created brands; remembers the choice across pages."""
    if not ids:
        st.info("No brands yet. Use **Add brand** first.")
        return None
    bid = st.selectbox(label, ids, index=ids.index(ss["sel_brand"]), format_func=lambda i: labels[i])
    ss["sel_brand"] = bid
    return bid


_INDUSTRIES = list(INDUSTRY_CLUSTERS)
_TONES = [("formal_casual", "Formal", "Casual"), ("reserved_bold", "Reserved", "Bold"), ("factual_emotive", "Factual", "Emotive")]


def kit_fields(prefix: str, brand: dict | None = None):
    """Brand kit inputs (inside a form). Returns (name, industry, kit dict)."""
    kit = (brand or {}).get("kit") or {}
    tone0 = kit.get("tone") or {}
    cols0 = kit.get("colours") or []
    c1, c2 = st.columns(2)
    name = c1.text_input("Brand name", value=(brand or {}).get("name", ""), max_chars=80, placeholder="e.g. Marico", key=f"{prefix}_name")
    cur_ind = (brand or {}).get("industry")
    ind = c2.selectbox("Industry", _INDUSTRIES, index=(_INDUSTRIES.index(cur_ind) if cur_ind in _INDUSTRIES else None),
                       placeholder="Choose the closest cluster", key=f"{prefix}_ind")
    aud = st.text_input("Target audience", value=kit.get("audience", ""), max_chars=200, placeholder="e.g. Indian households, 25-45, value-conscious", key=f"{prefix}_aud")
    pos = st.text_input("Positioning in one line", value=kit.get("positioning", ""), max_chars=200, placeholder="e.g. Everyday wellness brands families have trusted", key=f"{prefix}_pos")
    st.markdown("**Tone**")
    tone = {}
    for key_, left, right in _TONES:
        a, b_, c = st.columns([1, 5, 1])
        a.markdown(f"<div style='text-align:right;padding-top:0.3rem'>{left}</div>", unsafe_allow_html=True)
        tone[key_] = b_.slider(f"{left} to {right}", 0, 100, int(tone0.get(key_, 50)), label_visibility="collapsed", key=f"{prefix}_tone_{key_}")
        c.markdown(f"<div style='padding-top:0.3rem'>{right}</div>", unsafe_allow_html=True)
    st.markdown("**Brand colours**")
    use_cols = st.checkbox("Include brand colours", value=bool(cols0), key=f"{prefix}_usecols")
    defaults = (cols0 + ["#00843d", "#f4a900", "#1c1a17", "#5b4bdb"])[:4]
    pickers = st.columns(4)
    colours = [(pickers[i].color_picker(f"Colour {i + 1}", defaults[i], key=f"{prefix}_col{i}") or defaults[i]) for i in range(4)]
    avoid = st.text_input("Words to avoid", value=kit.get("avoid", ""), max_chars=200, placeholder="comma separated, e.g. cheap, miracle, guaranteed", key=f"{prefix}_avoid")
    notes = st.text_area("Brand voice & guidelines (optional)", value=kit.get("notes", ""), max_chars=8000, height=110, key=f"{prefix}_notes",
                         placeholder="Tone of voice, words you always/never use, taglines…")
    return name, ind, {"audience": aud, "positioning": pos, "tone": tone, "colours": colours if use_cols else [], "avoid": avoid, "notes": notes}


# ================================================================== ADD BRAND
if page == "Add brand":
    st.markdown("## Add brand")
    st.caption("Everything generated here is checked against this kit. Set it once; it governs every brief afterwards.")
    show_flash("brand")
    with st.form("create_brand"):
        b_name, b_ind, b_kit = kit_fields("add")
        create = st.form_submit_button("Save brand and continue", type="primary")
    with st.expander("What each industry cluster covers"):
        st.markdown("\n".join(f"- **{k}**: {v}" for k, v in INDUSTRY_CLUSTERS.items()))
    if create:
        if not b_ind:
            flash("brand", err(StoreError("Choose an industry.")))
        else:
            b, e = safe(svc.create_brand, user, b_name, b_ind, "", b_kit)
            if e:
                flash("brand", e)
            else:
                ss["sel_brand"] = b["brand_id"]
                flash("know", f"✅ Saved **{b['name']}**. Now add its sources.")
                ss["goto"] = "Sources"
        st.rerun()

# ================================================================== SOURCES
elif page == "Sources":
    st.markdown("## Sources")
    brand_id = brand_picker("Brand")
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
            st.markdown("##### Coverage by goal")
            cov, _e = safe(svc.goal_coverage, user, brand_id)
            if cov:
                st.dataframe(cov, hide_index=True, use_container_width=True)
                for row in cov:
                    if row["Status"] == "Thin":
                        st.markdown(f"**{row['Goal']} is thin.** {row['Chunks']} chunks won't support specific claims. Add a relevant document, or expect cautious, general copy on that goal.")
            dmap = {d["doc_id"]: f"{d['name'][:60]} ({d['kind']})" for d in docs}
            rc = st.columns([4, 1])
            gone = rc[0].selectbox("Remove a document", list(dmap), index=None, format_func=lambda i: dmap[i], placeholder="Pick a document")
            if rc[1].button("Remove") and gone:
                _, e = safe(svc.delete_document, user, brand_id, gone)
                flash("know", e or "🗑️ Removed.")
                st.rerun()


# ================================================================== BRAND KIT
elif page == "Brand kit":
    st.markdown("## Brand kit")
    st.caption("All your brands. Edit any of them here; changes apply to the next brief.")
    show_flash("kit")
    if not brands:
        st.info("No brands yet. Use **Add brand** first.")
    for b in brands:
        bid = b["brand_id"]
        kit = b.get("kit") or {}
        full_b = svc.store(user).get_brand(bid)
        kit = full_b.get("kit") or {}
        with st.container(border=True):
            h1, h2, h3 = st.columns([5, 1, 1])
            h1.markdown(f"**{b['name']}** · {b['industry']} · {b['docs']} sources")
            if h2.button("Edit", key=f"edit_{bid}"):
                ss["edit_brand"] = None if ss.get("edit_brand") == bid else bid
                ss.pop("confirm_del", None)
                st.rerun()
            if h3.button("Delete", key=f"del_{bid}"):
                ss["confirm_del"] = bid
                ss.pop("edit_brand", None)
                st.rerun()
            if kit.get("audience"):
                st.markdown(f"**Audience:** {kit['audience']}")
            if kit.get("positioning"):
                st.markdown(f"**Positioning:** {kit['positioning']}")
            if kit.get("tone"):
                t_ = kit["tone"]
                st.caption(" · ".join(f"{l}–{r}: {t_.get(k, 50)}" for k, l, r in _TONES))
            if kit.get("colours"):
                st.markdown("".join(f'<span class="bca-swatch" style="background:{c}"></span>' for c in kit["colours"]), unsafe_allow_html=True)
            if kit.get("avoid"):
                st.markdown(f"**Words to avoid:** {kit['avoid']}")
            if ss.get("confirm_del") == bid:
                st.warning(f"Delete {b['name']} and all its sources? This cannot be undone.")
                y, n_ = st.columns([1, 1])
                if y.button("Yes, delete", key=f"yes_{bid}", type="primary"):
                    _, e = safe(svc.delete_brand, user, bid)
                    ss.pop("confirm_del", None)
                    flash("kit", e or f"🗑️ Deleted {b['name']}.")
                    st.rerun()
                if n_.button("Cancel", key=f"no_{bid}"):
                    ss.pop("confirm_del", None)
                    st.rerun()
            if ss.get("edit_brand") == bid:
                with st.form(f"edit_form_{bid}"):
                    e_name, e_ind, e_kit = kit_fields(f"ed_{bid}", full_b)
                    save = st.form_submit_button("Save changes", type="primary")
                if save:
                    _, e = safe(svc.update_brand, user, bid, e_name, e_ind or b["industry"], e_kit)
                    if not e:
                        ss.pop("edit_brand", None)
                    flash("kit", e or f"✅ Updated {e_name.strip()}.")
                    st.rerun()

# ================================================================== BRIEF
elif page == "Brief":
    _brand_name = next((b["name"] for b in brands if b["brand_id"] == ss.get("sel_brand")), "")
    st.markdown(f"## New brief" + (f" · {_brand_name}" if _brand_name else "") + " · draft")
    desk, chat = st.columns([1, 2])
    with desk:
        brand_id = brand_picker("Brand")
        _docs_now, _ = safe(svc.list_documents, user, brand_id) if brand_id else (None, None)
        _n_docs = len(_docs_now or [])
        st.caption(f"{_n_docs} sources · Today's quota: {u['generations_left']} left · retrieval: {sem}")
        if brand_id:
            recipes, _re = safe(svc.list_recipes, user, brand_id)
            recipes = recipes or {}
            _plat_keys = list(PLATFORMS)
            chans = st.multiselect("Channels", _plat_keys, default=_plat_keys[:1], format_func=lambda k: PLATFORMS[k]["label"], key="g_chan")
            model = st.selectbox("Model", list(s.allowed_models), index=list(s.allowed_models).index(s.default_model), key="g_model")
            n = st.slider("Variations per channel", 1, 5, 2, key="g_n")
            st.caption("Each variation takes a different angle, so three gives a benefit-led, a story-led and a problem/solution version.")
            angles = st.text_input("Angles to use (optional)", max_chars=200, placeholder="e.g. festive gifting, family story", key="g_angles")
            brief = st.text_area("Brief / context (optional)", max_chars=2000, height=90, placeholder="Key message, offer, audience, dates, things to mention or avoid…", key="g_brief")
            g_topk = st.slider("Sources to retrieve (top-k)", 2, 12, s.default_top_k, key="g_topk")
            g_hybrid = st.slider("Hybrid weight (1 = semantic only, 0 = keyword only)", 0.0, 1.0, float(s.default_hybrid), 0.1, key="g_hyb")
            _pieces = max(1, len(chans or [1])) * int(n)
            st.markdown('<div class="bca-strip">'
                        f'<div>This run<b>{_pieces} pieces</b>{len(chans or [1])} × {int(n)}</div>'
                        f'<div>Est. time<b>~{max(1, len(chans or [1]))} to {2 * max(1, len(chans or [1]))} min</b></div></div>', unsafe_allow_html=True)
            st.markdown("**Reusable** · save this brief as a recipe.")
            rname = st.text_input("Recipe name", max_chars=40, placeholder="e.g. Festive launch kit")
            if st.button("Save as recipe"):
                _, e = safe(svc.save_recipe, user, brand_id, rname, {"g_chan": list(chans or []), "g_model": model, "g_n": int(n), "g_brief": brief,
                                                                      "g_angles": angles, "g_topk": int(g_topk), "g_hyb": float(g_hybrid)})
                flash("recipe", e or f"✅ Saved recipe **{rname.strip()[:40]}**.")
                st.rerun()
            if recipes:
                pick_r = st.selectbox("Saved recipes", list(recipes), index=None, placeholder="Pick a recipe")
                p2, p3 = st.columns(2)
                p2.button("Load recipe", on_click=lambda: _apply_recipe(recipes.get(pick_r, {})) if pick_r else None)
                if p3.button("Delete recipe") and pick_r:
                    _, e = safe(svc.delete_recipe, user, brand_id, pick_r)
                    flash("recipe", e or "🗑️ Recipe deleted.")
                    st.rerun()
            show_flash("recipe")
    with chat:
        with st.chat_message("assistant"):
            if not brand_id:
                st.markdown("Hi, I'm **BrandVoice AI**. Start with **Add brand**, then add its **Sources**, and I'll help you write on-brand posts here.")
            elif not _n_docs:
                st.markdown("Brand is ready. Add some **Sources** (documents, web pages or URLs) so I can ground every claim.")
            else:
                st.markdown("Hi, I'm **BrandVoice AI**. Choose your channels on the left, then tell me the campaign goal below, for example: "
                            "*Drive sign-ups for the summer launch*. I'll write variations for each channel from your sources and fact-check them.")
        if brand_id:
            hist = ss.setdefault("chat", {}).setdefault(brand_id, [])
            for i, m in enumerate(hist):
                with st.chat_message(m["role"]):
                    if m["role"] == "user":
                        st.markdown(m["text"])
                    elif m.get("error"):
                        st.markdown(m["error"])
                    else:
                        render_out(m["out"], m["platform"], f"h{i}", brand_id)
            prompt = st.chat_input("Describe your campaign goal, e.g. Drive sign-ups for the summer launch")
            if prompt:
                hist.append({"role": "user", "text": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                targets = list(chans or _plat_keys[:1])
                full_brief = (brief or "") + (f"\nAngles to use: {angles}" if angles else "")
                last_out, last_e = None, None
                with st.chat_message("assistant"):
                    bar = st.progress(0.0, text="Starting…")
                    for k, ch in enumerate(targets):
                        bar.progress(k / len(targets), text=f"Writing {PLATFORMS[ch]['label']} ({k + 1} of {len(targets)}), then fact-checking…")
                        out, e = safe(svc.generate, user, brand_id, ch, prompt, full_brief, int(n), model, int(g_topk), float(g_hybrid))
                        if out:
                            hist.append({"role": "assistant", "out": out, "platform": ch})
                            last_out = {"out": out, "platform": ch}
                        else:
                            hist.append({"role": "assistant", "error": e or "⚠️ Nothing was generated."})
                            last_e = e
                            break
                    bar.empty()
                ss["gen_out"] = last_out
                ss["gen_err"] = last_e
                st.rerun()
            if hist and st.button("New chat"):
                hist.clear()
                st.rerun()

# ================================================================== ACCOUNT
else:
    st.markdown("## Your account")
    show_flash("acct")
    acct = svc.accounts.get(user)
    if acct:
        with st.form("profile"):
            p1, p2 = st.columns(2)
            f_first = p1.text_input("First name", value=acct["first_name"], max_chars=40, key="pf_first")
            f_last = p2.text_input("Last name", value=acct["last_name"], max_chars=40, key="pf_last")
            st.text_input("Email", value=acct["email"], disabled=True, key="pf_email")
            if st.form_submit_button("Save details", type="primary"):
                try:
                    r = svc.accounts.update_profile(user, f_first, f_last)
                    ss["display"], ss["full"] = r["name"], r["full_name"]
                    flash("acct", "✅ Details saved.")
                except AccountError as exc:
                    flash("acct", err(exc))
                st.rerun()
        with st.form("password"):
            st.markdown("**Change password**")
            old_pw = st.text_input("Current password", type="password", key="pw_old")
            new_pw = st.text_input("New password (at least 8 characters)", type="password", key="pw_new")
            new_pw2 = st.text_input("Confirm new password", type="password", key="pw_new2")
            if st.form_submit_button("Change password"):
                try:
                    svc.accounts.change_password(user, old_pw, new_pw, new_pw2)
                    flash("acct", "✅ Password changed.")
                except AccountError as exc:
                    flash("acct", err(exc))
                st.rerun()
    else:
        st.info("This login is managed by the project team, so its name and password can't be edited here.")
    st.markdown("**Appearance**")
    theme_switcher()
    if st.button("Log out"):
        ss.clear()
        st.rerun()
    with st.expander("Evaluation (project testing tool)"):
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


if page != "Account":
    apply_theme()
st.caption("Demo build. Generated copy is a draft: verify facts against the sources before publishing. Content you add is stored on the server and may be visible to the operators.")
