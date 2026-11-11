---
title: Brand Content Agent
emoji: ✍️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Brand Content Agent

A RAG agent that writes **brand-grounded** marketing content (LinkedIn, X, Instagram, email) from a brand's own documents and web sources.

**Workflow:** create a brand → add knowledge (upload PDF/Word/text, *discover sources on the web*, paste URLs) → pick platform + goal → get 1-5 variations, each with its character count, the sources it drew from, and a brand-consistency note.

```
upload / URL / web discovery -> extract text -> chunk -> embed -> per-user store
goal + brief -> hybrid retrieval (semantic + BM25) -> grounded prompt -> LLM -> JSON -> variations
```

## Run locally (Windows CMD)

```
cd marketing-agent-public
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env`: set `OPENAI_API_KEY` and `APP_ACCESS_CODE`. For local-only testing you may instead add `ALLOW_NO_AUTH=1` to skip the login screen. Then:

```
python app.py
```

Open http://127.0.0.1:7860 . Run the tests with `pip install pytest` then `pytest`.

If `pip install` fails on `fastembed` or `ddgs` (very new Python versions), install the rest without them. The app falls back to `sentence-transformers` for embeddings and to Wikipedia + site pages for discovery.

### Logins for several people (one link)

Everyone opens the same address. To give each person their own account, add this line to `.env` (or to the host's secrets) and restart:

```
APP_USERS=manish:choose-a-password,priya:another-password,rahul:third-password
```

Usernames are lowercase letters/digits; passwords need 6+ characters and no commas. Each person's brands stay private to their username. When `APP_USERS` is set it replaces the shared `APP_ACCESS_CODE`.

## Streamlit version

The same agent also has a Streamlit front end. Locally: `pip install -r requirements.txt` then `streamlit run streamlit_app.py`. On Streamlit Community Cloud: New app -> this repo, branch `main`, main file `streamlit_app.py`, then in Advanced settings -> Secrets paste:

```
OPENAI_API_KEY = "your-key"
APP_USERS = "manish:choose-a-password,professor:another-password"
```

Free hosting wipes stored brands when the app restarts, so re-run the Evaluation tab's Step 1 after a restart.

## Evaluation tab

Tab "4 · Evaluation" tests the agent on 4 brands from different sectors (Swiggy, Marico, HDFC Bank, Infosys) x 4 platforms (LinkedIn, X, Instagram, email) x 4 campaign goals = 32 posts. Step 1 loads each brand's sources from the web. Step 2 generates every post with one model and scores it 1-5 (grounded, on-brand, goal relevance, platform fit) with a different model as judge, then gives summaries by brand, platform and goal plus a CSV. Top-k and the keyword/semantic blend can be changed to compare retrieval settings. Scores come from a language-model judge: guidance, not proof. Limits: `EVAL_PER_USER_DAY` (default 400), `EVAL_GLOBAL_DAY` (default 800), optional `EVAL_USERS`.

## Deploy publicly

The app is one container serving on `$PORT` (default 7860). Two easy hosts:

### Hugging Face Spaces (free, simplest for a demo)
1. Create a new **Space** → SDK **Docker** → visibility Public.
2. Push this folder to the Space's git repo (the README front-matter above configures it).
3. Space → **Settings → Variables and secrets** → add **secrets** `OPENAI_API_KEY` and `APP_ACCESS_CODE`.
4. Wait for the build. Share the Space URL plus the access code with your audience.

### Render
1. Push the folder to GitHub (the `.env` file is git-ignored).
2. Render → **New → Blueprint** → pick the repo (`render.yaml` is picked up).
3. Enter `OPENAI_API_KEY` and `APP_ACCESS_CODE` when prompted.

## Protecting a public deployment

| Risk | Mitigation built in |
|---|---|
| Strangers spending your API credits | Login (display name + shared `APP_ACCESS_CODE`), per-user and global **daily generation caps** (`GEN_PER_USER_DAY`, `GEN_GLOBAL_DAY`), refund on failed calls |
| Leaked API key | Key read only from environment, sent only in the HTTPS header, never logged or shown; `.env` is git- and docker-ignored |
| Server-side request forgery via "paste a URL" | Only public http(s) on ports 80/443; every URL **and redirect hop** is resolved and rejected if private/loopback/link-local/metadata; size, type and time limits |
| Users seeing each other's brands | Data is namespaced per display name; brand ids are validated hex (no path traversal) |
| Prompt injection from scraped pages | Sources are fenced and the model is told they are data, not instructions; output is plain text in a sanitised Markdown/Textbox |
| Disk/abuse | Limits on brands, documents, chunks, upload size, fetch size |

Honest limits: the shared access code is a *gate*, not per-person identity (anyone with the code can type any display name). Use a long random code and change it after the demo. DNS-rebinding between validation and connection is a residual SSRF risk; on a hardened deployment add network egress rules. Rotate the OpenAI key after the demo.

## Configuration

See `.env.example`. Key settings: `ALLOWED_MODELS`, `DEFAULT_MODEL`, `EMBEDDER`, `SEARCH_PROVIDER` (`tavily`/`brave` give better web search than the default free DuckDuckGo), `DATA_DIR`.

## Known limitations

- Pages rendered entirely by JavaScript expose little text to a plain fetch (the app warns you). Upload a brand PDF/Word file for those.
- On free hosts the disk is ephemeral: brands reset when the container restarts. Mount a persistent disk and set `DATA_DIR` for permanence.
- Generated copy is a draft; always verify facts against the listed sources.

## Layout

```
app.py            entry point (login, queue, launch)
ui.py             Gradio interface (thin; calls the service)
agent/service.py  workflow + validation + usage limits
agent/store.py    per-user store, hybrid retrieval (BM25 + cosine)
agent/discover.py web discovery (search / Wikipedia / site pages)
agent/security.py login + SSRF-safe fetching
agent/llm.py      OpenAI client that adapts to newer model parameters
agent/generate.py platform rules, prompt, JSON parsing, length checks
tests/            48 offline tests
```
