# CCS project: Brand Content Agent (DMS Group 9): status as of 16 Nov 2026

## What it is
A RAG-based marketing content agent. A user signs up or logs in, creates a brand, adds knowledge (uploads, web/Wikipedia discovery, pasted URLs) and generates platform-specific posts (LinkedIn, X, Instagram, email) grounded in that brand's sources, with a second-model fact-check.

## Built and working (Streamlit Cloud)
- **Login and sign-up page:** two tabs. Log in with email or username plus password; sign up with first name, last name, email and password. Accounts created by `APP_USERS` still work. Each account has private brands and daily usage limits.
- **Account storage:** one file per account with a salted scrypt password hash (never the password). Stored on the server's disk, and also copied to Supabase once Supabase is connected (see "Not done"). Options: `SIGNUP_ENABLED`, `SIGNUP_CODE` (invite code), `SIGNUP_PER_DAY`.
- **Brand tab** with 15 industry clusters; **Knowledge tab**; **Generate tab** (variations, retrieval top-k and keyword/semantic blend adjustable); fact-check flags claims not found in sources.
- **Evaluation tab:** 4 brands (Swiggy, Marico, HDFC Bank, Infosys) x 4 platforms x 2 goals per brand-platform pair = 32 posts (8 per brand, platform and goal). Generator gpt-5-nano, judge gpt-5.4-nano (must differ). Scores 1-5 on grounded, on-brand, goal relevance, platform fit, plus a length check; CSV export.
- **Past runs:** a section in the Evaluation tab lists saved evaluation CSVs and shows their summary tables.
- **Look:** Streamlit's Share, star, edit, GitHub icons and the three-dot menu are hidden with CSS and `toolbarMode = "viewer"`. A System / Light / Dark selector on the page replaces the menu (it drives the hidden menu's theme buttons; the button clicks were tested on the live page, the selector itself still needs checking after deploy). The black "Manage app" button is added by Streamlit for the app owner and cannot be removed from code.
- No separate backend: the Python app calls the OpenAI API directly. Data is stored as files in a data folder, which free hosts wipe on restart (see "Not done").
- 76 automated tests pass (sandbox run, against a stand-in for Streamlit and a fake Supabase; not run against a real Supabase project).

## Evaluation results (32 items each; LLM-judge scores, guidance only)
| Run | Top-k | Hybrid | Scored | Grounded | On-brand | Goal rel. | Platform fit |
|---|---|---|---|---|---|---|---|
| Baseline | 6 | 0.5 | 32 | 3.81 | 4.34 | 4.03 | 4.31 |
| Keyword-only | 6 | 0.0 | 31 | 4.06 | 4.52 | 4.06 | 4.16 |
| Semantic-only | 6 | 1.0 | 32 | 3.59 | 4.47 | 3.75 | 4.22 |
| Fewer sources | 3 | 0.5 | 31 | 3.68 | 4.39 | 3.77 | 4.32 |
| More sources | 10 | 0.5 | 30 | 3.60 | 4.63 | 4.17 | 4.30 |

Failed items excluded from means. Differences of about 0.2 are within noise (one run per setting, one judge). Weakest: "sustainability" goal; Infosys/HDFC Bank had only 1-2 of 8 pages loaded locally.

## Code and hosting
- GitHub repo: https://github.com/manishv9237/ccs-marketing-agent (`.env`, `venv`, `data` not included). Latest pushed commit includes sign-up, account storage and the Supabase code; the toolbar/theme changes follow in the next push.
- Live on Streamlit Cloud: https://ccs-marketing-agent.streamlit.app/ (app is set to public: anyone can open the page, only logged-in users can use it).
- Gradio app (`app.py`) kept unchanged as local fallback.
- A friend/teammate works on the code through GitHub (a `.devcontainer` folder was added by them).

## Not done / open
1. **Supabase is not connected.** The code is ready (`agent/persist.py`, `supabase/schema.sql`), but until a Supabase project exists and `SUPABASE_URL` / `SUPABASE_SERVICE_KEY` are added to Streamlit Secrets, new accounts and brands are lost when the free host restarts. Steps are in the README.
2. **Set `SIGNUP_CODE`** in Streamlit Secrets before sharing the link widely; otherwise anyone can sign up and use the shared API key (daily limits still apply).
3. Short report: draft exists (problem, method, results table, limits); still to add the interview and questionnaire findings, the one-user validation and a hand-read check of about 10 judged items.
4. One-user workflow validation (a real person; record their actual answers).
5. Demo backup screenshots (Swiggy LinkedIn, Marico Instagram).
6. Rotate the API key and change logins after the demo.

## Honest notes for the report
- Scores come from an LLM judge; read a sample by hand before trusting them.
- Questionnaire: report only responses actually collected from real respondents, with the number of respondents; take figures from the original questionnaire file.
- Known weaknesses: about 100 s per generation; fact-check is inconsistent; JS-heavy sites give thin text (probable cause of the thin HDFC/Infosys sources); free hosts reset stored data unless Supabase is connected.
- Survey features not built: images/video, ad-copy fields and Meta/Google Ads integration, competitor analysis, ad-performance learning, bulk generation.
