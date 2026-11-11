"""Gradio interface. All logic lives in agent.service; handlers only translate.

Per-user isolation: Gradio's login gives every request `request.username`
(a display name + the shared access code), and every service call is scoped to it.
"""
# NOTE: do NOT add `from __future__ import annotations` here. Gradio finds the `request: gr.Request`
# argument by inspecting the real annotation object; stringified annotations make it silently
# skip the injection and every handler fails with a missing-argument error.
import gradio as gr

from agent import evaluate as ev
from agent.extract import SUPPORTED_FILE_TYPES
from agent.generate import PLATFORMS
from agent.security import normalize_username
from agent.service import USER_ERRORS, IngestResult, Service
from agent.store import StoreError

MAX_VARIATIONS = 5

INDUSTRY_CLUSTERS = {
    "FMCG & Consumer Brands": "FMCG, packaged foods, beverages, dairy, personal care, cosmetics, home care",
    "Retail, E-commerce & D2C": "Retail, online marketplaces, quick commerce, D2C, fashion, lifestyle, jewellery",
    "BFSI": "Banking, insurance, NBFCs, fintech, wealth management, financial services",
    "Technology & Digital Products": "SaaS, IT services, AI, software, cybersecurity, telecom",
    "Healthcare & Life Sciences": "Pharma, hospitals, diagnostics, medical devices, biotechnology, wellness",
    "Automotive & Mobility": "Cars, two-wheelers, commercial vehicles, EVs, auto components, mobility",
    "Industrial & Manufacturing": "Manufacturing, machinery, construction equipment, chemicals, electronics, industrial automation",
    "Energy & Sustainability": "Oil & gas, power, renewable energy, EV charging, sustainability solutions",
    "Real Estate & Infrastructure": "Real estate, construction, cement, building materials, infrastructure, EPC",
    "Logistics & Transportation": "Logistics, supply chain, shipping, aviation, warehousing, transportation",
    "B2B & Professional Services": "Consulting, legal, accounting, recruitment, enterprise services, BPO/KPO",
    "Media & Entertainment": "OTT, films, music, gaming, sports, publishing, entertainment",
    "Travel & Hospitality": "Hotels, airlines, tourism, restaurants, travel platforms",
    "Education & EdTech": "Schools, colleges, coaching, EdTech, skill development",
    "Public Sector & Social Impact": "Government, NGOs, CSR, public awareness, social campaigns",
}
INDUSTRY_HINT_DEFAULT = "<small style='opacity:.7'>Choose the closest cluster. What each one covers will show here.</small>"


def industry_hint(cluster):
    subs = INDUSTRY_CLUSTERS.get(cluster or "")
    return f"<small style='opacity:.7'>Covers: {subs}</small>" if subs else INDUSTRY_HINT_DEFAULT


def _copy_textbox(**kw):
    """Textbox with a copy button, tolerant of Gradio API differences between versions."""
    for extra in ({"buttons": ["copy"]}, {"show_copy_button": True}, {}):
        try:
            return gr.Textbox(**kw, **extra)
        except TypeError:
            continue
    return gr.Textbox(**kw)


def build_ui(svc: Service) -> gr.Blocks:
    s = svc.s

    # ------------------------------------------------------------------ helpers
    def who(request: gr.Request) -> str:
        user = normalize_username(getattr(request, "username", None))
        if user:
            return user
        if s.allow_no_auth:
            return "local"
        raise gr.Error("Please sign in.")

    def err(exc: Exception) -> str:
        return f"⚠️ {exc}"

    def header(user: str) -> str:
        u = svc.usage(user)
        sem = {True: "semantic + keyword", False: "keyword-only (embedding model unavailable)", None: "loading embedding model…"}[u["semantic_search"]]
        return f"Signed in as **{user}** · generations left today: **{u['generations_left']}** · retrieval: {sem}"

    def brand_state(user: str, selected: str | None = None):
        """(choices, selected_id). Never read attributes back from a gr.* component created in a
        handler - Gradio 6 does not expose them; keep the plain values separately."""
        brands = svc.list_brands(user)
        choices = [(f"{b['name']} · {b['industry']} ({b['docs']} docs)", b["brand_id"]) for b in brands]
        ids = [c[1] for c in choices]
        return choices, (selected if selected in ids else (ids[0] if ids else None))

    def brand_dropdown(user: str, selected: str | None = None):
        choices, value = brand_state(user, selected)
        return gr.Dropdown(choices=choices, value=value)

    def docs_view(user: str, brand_id: str | None):
        if not brand_id:
            return "*Create or select a brand first.*", gr.Dropdown(choices=[], value=None)
        try:
            docs = svc.list_documents(user, brand_id)
        except USER_ERRORS as exc:
            return err(exc), gr.Dropdown(choices=[], value=None)
        if not docs:
            return "*No knowledge yet. Upload files, paste URLs or discover sources on the web.*", gr.Dropdown(choices=[], value=None)
        rows = ["| Source | Type | Chunks |", "|---|---|---|"]
        for d in docs:
            label = (d["source"] or d["name"]).replace("|", "/")[:90]
            rows.append(f"| {label} | {d['kind']} | {d['chunks']} |")
        choices = [(f"{d['name'][:60]} ({d['kind']})", d["doc_id"]) for d in docs]
        return "\n".join(rows), gr.Dropdown(choices=choices, value=None)

    def fmt_results(results: list[IngestResult]) -> str:
        if not results:
            return "Nothing to ingest."
        lines = []
        for r in results:
            name = r.name if len(r.name) < 70 else r.name[:67] + "..."
            if r.ok:
                lines.append(f"✅ **{name}** — {r.chunks} chunks" + (f" _({r.message})_" if r.message else ""))
            else:
                lines.append(f"❌ **{name}** — {r.message}")
        return "\n\n".join(lines)

    # ----------------------------------------------------------------- handlers
    def on_load(request: gr.Request):
        user = who(request)
        choices, selected = brand_state(user)
        docs_md, doc_dd = docs_view(user, selected)
        return header(user), gr.Dropdown(choices=choices, value=selected), docs_md, doc_dd

    def on_brand_change(brand_id, request: gr.Request):
        return docs_view(who(request), brand_id)

    def on_industry_change(industry, request: gr.Request):
        who(request)
        return industry_hint(industry)

    # ------------------------------------------------------------- evaluation
    def on_eval_setup(request: gr.Request):
        user = who(request)
        try:
            for text in ev.setup_brands_stream(svc, user):
                yield text, brand_dropdown(user)
        except USER_ERRORS as exc:
            yield err(exc), brand_dropdown(user)
        except Exception:  # unexpected: do not leak internals
            yield "⚠️ Something went wrong during setup. Please try again.", brand_dropdown(user)

    def on_eval_run(gen_model, judge_model, n_items, top_k, hybrid, request: gr.Request):
        user = who(request)
        params = {"gen_model": gen_model, "judge_model": judge_model, "top_k": int(top_k), "hybrid": float(hybrid)}
        rows = []
        try:
            for done, total, rows in ev.evaluate_stream(svc, user, gen_model, judge_model, int(n_items), int(top_k), float(hybrid)):
                yield f"⏳ Scored {done} of {total} items…", "", None
            ok = [r for r in rows if not r["error"]]
            if not ok:
                errs = "; ".join(sorted({r["error"] for r in rows}))[:300]
                yield f"⚠️ No item could be scored. {errs}", "", None
                return
            yield f"✅ Finished: {len(ok)} of {len(rows)} items scored.", ev.summary_markdown(rows, params), ev.write_csv(svc, user, rows, params)
        except USER_ERRORS as exc:
            yield err(exc), "", None
        except Exception:
            yield "⚠️ Something went wrong during the evaluation. Please try again.", "", None

    def on_create(name, industry, notes, request: gr.Request):
        user = who(request)
        try:
            if not (industry or "").strip():
                raise StoreError("Choose an industry cluster.")
            b = svc.create_brand(user, name, industry, notes)
            status = f"✅ Created **{b['name']}**. Next: add knowledge in the **Knowledge** tab."
            choices, selected = brand_state(user, b["brand_id"])
        except USER_ERRORS as exc:
            status = err(exc)
            choices, selected = brand_state(user)
        docs_md, doc_dd = docs_view(user, selected)
        return status, gr.Dropdown(choices=choices, value=selected), docs_md, doc_dd, header(user)

    def on_delete_brand(brand_id, request: gr.Request):
        user = who(request)
        try:
            if not brand_id:
                raise ValueError("Select a brand first.")
            svc.delete_brand(user, brand_id)
            status = "🗑️ Brand deleted."
        except (ValueError, *USER_ERRORS) as exc:
            status = err(exc)
        choices, selected = brand_state(user)
        docs_md, doc_dd = docs_view(user, selected)
        return status, gr.Dropdown(choices=choices, value=selected), docs_md, doc_dd

    def after_ingest(user, brand_id, results):
        docs_md, doc_dd = docs_view(user, brand_id)
        return fmt_results(results), docs_md, doc_dd, brand_dropdown(user, brand_id)

    def on_upload(brand_id, files, request: gr.Request):
        user = who(request)
        if not brand_id:
            return (err(ValueError("Select a brand first.")), *docs_view(user, None), brand_dropdown(user))
        try:
            paths = [f if isinstance(f, str) else getattr(f, "name", str(f)) for f in (files or [])]
            return after_ingest(user, brand_id, svc.ingest_files(user, brand_id, paths))
        except USER_ERRORS as exc:
            return (err(exc), *docs_view(user, brand_id), brand_dropdown(user, brand_id))

    def on_paste_urls(brand_id, text, request: gr.Request):
        user = who(request)
        if not brand_id:
            return (err(ValueError("Select a brand first.")), *docs_view(user, None), brand_dropdown(user))
        try:
            urls = [u for u in (text or "").replace(",", "\n").splitlines() if u.strip()]
            return after_ingest(user, brand_id, svc.ingest_urls(user, brand_id, urls))
        except USER_ERRORS as exc:
            return (err(exc), *docs_view(user, brand_id), brand_dropdown(user, brand_id))

    def on_discover(brand_id, domain, request: gr.Request):
        user = who(request)
        if not brand_id:
            return err(ValueError("Select a brand first.")), gr.CheckboxGroup(choices=[], value=[])
        try:
            cands, warns = svc.discover(user, brand_id, domain)
        except USER_ERRORS as exc:
            return err(exc), gr.CheckboxGroup(choices=[], value=[])
        notes = [f"⚠️ {w}" for w in warns]
        if not cands:
            return "\n\n".join(notes + ["No sources found. Try adding the brand's website domain."]), gr.CheckboxGroup(choices=[], value=[])
        choices = [(c.label, c.url) for c in cands]
        preselect = [c.url for c in cands if c.origin in ("wikipedia", "site")][:4] + [c.url for c in cands if c.origin == "search"][:3]
        msg = f"Found **{len(cands)}** candidate sources. Untick anything irrelevant, then press **Ingest selected**."
        return "\n\n".join(notes + [msg]), gr.CheckboxGroup(choices=choices, value=preselect)

    def on_ingest_selected(brand_id, selected, request: gr.Request):
        user = who(request)
        if not brand_id:
            return (err(ValueError("Select a brand first.")), *docs_view(user, None), brand_dropdown(user))
        if not selected:
            return (err(ValueError("Tick at least one source.")), *docs_view(user, brand_id), brand_dropdown(user, brand_id))
        try:
            return after_ingest(user, brand_id, svc.ingest_urls(user, brand_id, list(selected)))
        except USER_ERRORS as exc:
            return (err(exc), *docs_view(user, brand_id), brand_dropdown(user, brand_id))

    def on_delete_doc(brand_id, doc_id, request: gr.Request):
        user = who(request)
        try:
            if not (brand_id and doc_id):
                raise ValueError("Pick a document to remove.")
            svc.delete_document(user, brand_id, doc_id)
            status = "🗑️ Removed."
        except (ValueError, *USER_ERRORS) as exc:
            status = err(exc)
        docs_md, doc_dd = docs_view(user, brand_id)
        return status, docs_md, doc_dd, brand_dropdown(user, brand_id)

    def on_generate(brand_id, platform, goal, brief, n, model, top_k, hybrid, request: gr.Request):
        user = who(request)
        hidden = [gr.Markdown(visible=False), gr.Textbox(visible=False)] * MAX_VARIATIONS

        def fail(msg):
            return [msg, *hidden, "", header(user)]

        if not brand_id:
            return fail(err(ValueError("Select a brand first.")))
        try:
            out = svc.generate(user, brand_id, platform, goal, brief, int(n), model, int(top_k), float(hybrid))
        except USER_ERRORS as exc:
            return fail(err(exc))
        except Exception:  # unexpected: do not leak internals on a public site
            return fail("⚠️ Something went wrong while generating. Please try again.")

        slots: list = []
        for i in range(MAX_VARIATIONS):
            if i < len(out["variations"]):
                v = out["variations"][i]
                cites = ", ".join(f"S{x}" for x in v["sources"]) or "—"
                meta = f"**Variation {v['number']} · {v['angle']}** &nbsp; _{v['length']} · sources: {cites}_"
                if v["brand_note"]:
                    meta += f"\n\n> {v['brand_note']}"
                if v.get("unsupported"):
                    meta += "\n\n⚠️ **Not found in your sources (check before posting):** " + "; ".join(v["unsupported"])
                elif v.get("verified"):
                    meta += "\n\n✅ Fact-check: claims match the sources"
                slots += [gr.Markdown(value=meta, visible=True), _copy_textbox(value=v["content"], visible=True, lines=8, label=f"{PLATFORMS[platform]['label']} post {v['number']}")]
            else:
                slots += hidden[:2]
        src_lines = []
        for i, sr in enumerate(out["sources"], 1):
            snippet = sr["text"][:320].replace("\n", " ")
            src_lines.append(f"**S{i}** · `{sr['source'][:80]}` · score {sr['score']}\n> {snippet}{'…' if len(sr['text']) > 320 else ''}")
        stats = (f"Model **{out['model']}** · retrieval {out['retrieval_ms']:.0f} ms · generation {out['generation_ms'] / 1000:.1f} s · "
                 f"{len(out['sources'])} sources · {'hybrid' if out['semantic'] else 'keyword-only'} search")
        return [stats, *slots, "\n\n".join(src_lines), header(user)]

    # ----------------------------------------------------------------------- UI
    with gr.Blocks(title="Brand Content Agent") as demo:
        gr.Markdown("# Brand Content Agent\nGrounded marketing copy for LinkedIn, X, Instagram and email — written from **your brand's own documents and web sources**, not from guesswork.")
        hdr = gr.Markdown()
        brand_dd = gr.Dropdown(label="Active brand", choices=[], interactive=True)

        with gr.Tabs():
            with gr.Tab("1 · Brand"):
                with gr.Row():
                    b_name = gr.Textbox(label="Brand name", placeholder="e.g. Acme Foods", max_length=80)
                    b_ind = gr.Dropdown(choices=list(INDUSTRY_CLUSTERS), value=None, label="Industry cluster", filterable=False)
                b_hint = gr.Markdown(value=INDUSTRY_HINT_DEFAULT)
                with gr.Accordion("What each industry cluster covers", open=False):
                    gr.Markdown("\n".join(f"<small style='opacity:.75'><b>{k}</b>: {v}</small>  " for k, v in INDUSTRY_CLUSTERS.items()))
                b_notes = gr.Textbox(label="Brand voice & guidelines (optional)", lines=4, max_length=8000,
                                     placeholder="Tone of voice, words you always/never use, audience, positioning, taglines…")
                with gr.Row():
                    b_create = gr.Button("Create brand", variant="primary")
                    b_delete = gr.Button("Delete active brand", variant="stop")
                b_status = gr.Markdown()

            with gr.Tab("2 · Knowledge"):
                gr.Markdown("The more real brand material here, the better and more specific the content. **Don't upload confidential documents** — this is a shared public demo.")
                with gr.Accordion("A · Upload documents", open=True):
                    k_files = gr.File(label=f"PDF, Word, text or Markdown ({s.max_upload_mb} MB max each)", file_count="multiple",
                                      file_types=list(SUPPORTED_FILE_TYPES), type="filepath")
                    k_upload = gr.Button("Ingest files", variant="primary")
                with gr.Accordion("B · Discover on the web", open=True):
                    gr.Markdown("Searches the web and Wikipedia for the active brand and lists candidate pages. **You choose** what gets ingested.")
                    k_domain = gr.Textbox(label="Brand website (optional, improves results)", placeholder="acme.com")
                    k_discover = gr.Button("Find sources")
                    k_checks = gr.CheckboxGroup(label="Candidate sources", choices=[])
                    k_ingest_sel = gr.Button("Ingest selected", variant="primary")
                with gr.Accordion("C · Paste URLs", open=False):
                    k_urls = gr.Textbox(label="One URL per line (max 15)", lines=4, placeholder="https://acme.com/about")
                    k_paste = gr.Button("Ingest URLs", variant="primary")
                k_status = gr.Markdown()
                gr.Markdown("### What this brand knows")
                k_docs = gr.Markdown()
                with gr.Row():
                    k_doc_dd = gr.Dropdown(label="Remove a document", choices=[], interactive=True)
                    k_doc_del = gr.Button("Remove", variant="stop")

            with gr.Tab("3 · Generate"):
                with gr.Row():
                    g_platform = gr.Dropdown(label="Platform", choices=[(v["label"], k) for k, v in PLATFORMS.items()], value="linkedin")
                    g_model = gr.Dropdown(label="Model", choices=list(s.allowed_models), value=s.default_model)
                    g_n = gr.Slider(1, MAX_VARIATIONS, value=2, step=1, label="Variations")
                g_goal = gr.Textbox(label="Campaign goal", lines=2, max_length=500, placeholder="e.g. Drive sign-ups for the summer launch")
                g_brief = gr.Textbox(label="Brief / context (optional)", lines=3, max_length=2000, placeholder="Key message, offer, audience, dates, things to mention or avoid…")
                with gr.Accordion("Advanced retrieval", open=False):
                    g_topk = gr.Slider(2, 12, value=s.default_top_k, step=1, label="Sources to retrieve (top-k)")
                    g_hybrid = gr.Slider(0, 1, value=s.default_hybrid, step=0.1, label="Hybrid weight (1 = semantic only, 0 = keyword only)")
                g_go = gr.Button("Generate content", variant="primary")
                g_stats = gr.Markdown()
                slot_components = []
                for _ in range(MAX_VARIATIONS):
                    m = gr.Markdown(visible=False)
                    t = _copy_textbox(visible=False, lines=8, interactive=False)
                    slot_components += [m, t]
                with gr.Accordion("Sources used (check every claim against these)", open=False):
                    g_sources = gr.Markdown()

            with gr.Tab("4 · Evaluation"):
                gr.Markdown("Tests the agent on **4 brands from different sectors x 4 platforms x 4 goal types** (32 posts, each scored 1 to 5 by a *different* model as judge). "
                            "Step 1 loads the brands from their official websites and Wikipedia. Step 2 runs the test and gives a CSV.")
                e_setup = gr.Button("Step 1: Set up the 4 evaluation brands (Swiggy, Marico, HDFC Bank, Infosys)")
                e_setup_status = gr.Markdown()
                with gr.Row():
                    e_gen = gr.Dropdown(label="Generator model", choices=list(s.allowed_models), value=s.default_model)
                    e_judge = gr.Dropdown(label="Judge model (must differ)", choices=list(s.allowed_models),
                                          value=next((m for m in ("gpt-5.4-nano", "gpt-5-nano", "gpt-6-luna", "gpt-5.6-luna") if m in s.allowed_models and m != s.default_model), s.default_model))
                with gr.Row():
                    e_n = gr.Slider(4, ev.MAX_ITEMS, value=ev.MAX_ITEMS, step=4, label="Test items (32 = full run)")
                    e_topk = gr.Slider(2, 12, value=s.default_top_k, step=1, label="Sources to retrieve (top-k)")
                    e_hybrid = gr.Slider(0, 1, value=s.default_hybrid, step=0.1, label="Hybrid weight (1 = semantic only, 0 = keyword only)")
                e_run = gr.Button("Step 2: Run evaluation", variant="primary")
                e_progress = gr.Markdown()
                e_summary = gr.Markdown()
                e_file = gr.File(label="Results (CSV)", interactive=False)

        gr.Markdown("<small>Demo build. Generated copy is a draft: verify facts against the sources before publishing. Content you add is stored on the server and may be visible to the operators.</small>")

        # ------------------------------------------------------------- wiring
        docs_outputs = [k_status, k_docs, k_doc_dd, brand_dd]
        demo.load(on_load, None, [hdr, brand_dd, k_docs, k_doc_dd])
        brand_dd.change(on_brand_change, [brand_dd], [k_docs, k_doc_dd])
        b_ind.change(on_industry_change, [b_ind], [b_hint])
        b_create.click(on_create, [b_name, b_ind, b_notes], [b_status, brand_dd, k_docs, k_doc_dd, hdr])
        b_delete.click(on_delete_brand, [brand_dd], [b_status, brand_dd, k_docs, k_doc_dd])
        k_upload.click(on_upload, [brand_dd, k_files], docs_outputs)
        k_paste.click(on_paste_urls, [brand_dd, k_urls], docs_outputs)
        k_discover.click(on_discover, [brand_dd, k_domain], [k_status, k_checks])
        k_ingest_sel.click(on_ingest_selected, [brand_dd, k_checks], docs_outputs)
        k_doc_del.click(on_delete_doc, [brand_dd, k_doc_dd], docs_outputs)
        g_go.click(on_generate, [brand_dd, g_platform, g_goal, g_brief, g_n, g_model, g_topk, g_hybrid],
                   [g_stats, *slot_components, g_sources, hdr], concurrency_limit=3)
        e_setup.click(on_eval_setup, None, [e_setup_status, brand_dd], concurrency_limit=1)
        e_run.click(on_eval_run, [e_gen, e_judge, e_n, e_topk, e_hybrid], [e_progress, e_summary, e_file], concurrency_limit=1)
    return demo
