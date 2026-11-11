# CCS project: Brand Content Agent (DMS Group 9): status as of 11 Nov 2026

## What it is
A RAG-based marketing content agent. A user logs in, creates a brand, adds knowledge (uploads, web/Wikipedia discovery, pasted URLs) and generates platform-specific posts (LinkedIn, X, Instagram, email) grounded in that brand's sources, with a second-model fact-check.

## Built and working (tested on the laptop with Gradio)
- Login with individual accounts (`APP_USERS`), per-user private brands, daily usage limits.
- Brand tab with 15 industry clusters; Knowledge tab; Generate tab (retrieval top-k and keyword/semantic blend adjustable); fact-check flags claims not found in sources.
- Evaluation tab: 4 brands (Swiggy, Marico, HDFC Bank, Infosys) x 4 platforms x 4 goals = 32 posts. Generator gpt-5-nano, judge gpt-5.4-nano (must differ). Scores 1-5 on grounded, on-brand, goal relevance, platform fit; CSV export.
- No database or separate backend: the Python app calls the OpenAI API directly; data is stored as files in a data folder (wiped on restart on free hosts).
- 60 automated tests pass (sandbox run).

## Evaluation results (32 items each; LLM-judge scores, guidance only)
| Run | Top-k | Hybrid | Scored | Grounded | On-brand | Goal rel. | Platform fit |
|---|---|---|---|---|---|---|---|
| Baseline | 6 | 0.5 | 32 | 3.81 | 4.34 | 4.03 | 4.31 |
| Keyword-only | 6 | 0.0 | 31 | 4.06 | 4.52 | 4.06 | 4.16 |
| Semantic-only | 6 | 1.0 | 32 | 3.59 | 4.47 | 3.75 | 4.22 |
| Fewer sources | 3 | 0.5 | 31 | 3.68 | 4.39 | 3.77 | 4.32 |
| More sources | 10 | 0.5 | 30 | 3.60 | 4.63 | 4.17 | 4.30 |
Failed items excluded from means. Differences of about 0.2 are within noise. Weakest: "sustainability" goal; Infosys/HDFC Bank had only 1-2 of 8 pages loaded.

## Code and hosting
- GitHub repo: https://github.com/manishv9237/ccs-marketing-agent (2 commits pushed; `.env`, `venv`, `data` not included).
- Hugging Face Spaces: NOT usable (Gradio/Docker Spaces need a paid plan on this account). Nothing was created or paid for.
- Streamlit front end (`streamlit_app.py`) written this session and tested only against a stand-in (Streamlit could not be installed in the sandbox). Streamlit 1.67 was installed on the laptop; running it for real was just starting. NOT yet pushed to GitHub, NOT yet deployed.
- Gradio app (`app.py`) is kept unchanged as the fallback and the planned demo.

## Not done / open
1. Run and fix the Streamlit app locally; then commit (GitHub Desktop), then create the app on Streamlit Cloud (main file `streamlit_app.py`; add `OPENAI_API_KEY` and `APP_USERS` in Secrets yourself).
2. Short report; finish-today doc still describes the old 6-brand/48-item plan.
3. One-user workflow validation (a real person; record answers).
4. Demo backup screenshots (Swiggy LinkedIn, Marico Instagram).
5. Rotate the API key and change logins after the demo.

## Honest notes for the report
- 3 of the 6 survey responses are reconstructed (not original); report original-only figures separately.
- Scores come from an LLM judge; a sample should be read by hand.
- Known weaknesses: ~100 s per generation; fact-check is inconsistent; JS-heavy sites give thin text; free hosts reset stored data.
