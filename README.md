# Self-Evaluating Lesson Content Generator

Streamlit app for generating and reviewing a standalone beginner lesson with Gemini. The writer and evaluator use LangChain's Gemini integration; LangGraph controls the generate → evaluate → retry flow; SQLite stores lessons, attempt-level evaluator JSON, and categorized failure feedback.

## Requirements and setup (Windows PowerShell)

Python 3.10 or newer. A Gemini API key with API access is required for lesson generation and review.

```powershell
cd "$HOME\OneDrive\Desktop\Nxtwave"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env  # only if you do not already have a configured .env
```

Put the key in `GEMINI_API_KEY` in `.env`. Keep `.env` private; it is listed in `.gitignore`. Never paste the key into source code, screenshots, a commit, or this chat. `GEMINI_MODEL` and `SQLITE_PATH` are optional. Relative database paths are resolved under the project folder.

## Run

```powershell
cd "$HOME\OneDrive\Desktop\Nxtwave"
\.venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`). Enter a topic of 3–200 characters and select **Generate lesson**. Generation and evaluation both require working network access to the Gemini API.

## Architecture and design decisions

- `app.py`: Streamlit interface, progress, rubric, retry/rejection log, memory insights, and saved-run viewer.
- `llm_client.py`: LangChain `ChatGoogleGenerativeAI` calls for both writer and evaluator. It prefers Gemini structured JSON output and falls back to strict JSON text parsing. Credentials are loaded from `.env` by `config.py` and are redacted from UI/log errors.
- `prompts.py`: separate, constrained writer and reviewer prompts. The lesson is written for a 12th-grade learner with limited English and no prior topic knowledge; it defines terms, includes everyday examples, and follows a standalone lesson structure.
- `evaluation.py`: validates the six PASS/FAIL checks, recomputes `failed_checks`, rejection reasons and overall result from individual checks, and chooses the best fully evaluated attempt. An overall PASS cannot override a failed or missing check.
- `graph/workflow.py`: LangGraph nodes load memory → generate → evaluate → conditionally retry (maximum two rewrites) → finalize. A pass stops immediately. Invalid or unavailable evaluator output is never a pass.
- `memory/store.py`: SQLite tables for generation records, every reviewed attempt, and one feedback row per failed criterion. Related notes are retrieved into the next generator prompt. The default is `data/lessons.db`.

Keeping the earliest attempt when passing-check counts tie avoids replacing a lesson with an equally scored rewrite. Only fully evaluated attempts are eligible as the final lesson; an API/evaluator error is disclosed.

## Checks and example output shape

The evaluator checks factual accuracy and groundedness, beginner-friendly language, real-world examples, clear language without unexplained jargon, key topic coverage, and coherent teaching flow. Stored JSON has this shape (illustrative structure only, not an evaluation result):

```json
{
  "checks": {
    "accuracy_and_groundedness": {"result": "PASS", "reason": "..."},
    "beginner_friendly_language": {"result": "FAIL", "reason": "..."}
  },
  "failed_checks": ["beginner_friendly_language"],
  "rejection_reasons": ["Beginner-friendly language: ..."],
  "overall": "FAIL",
  "pass_count": 5
}
```

The app presents all six checks in the UI. A response with missing checks, malformed JSON, or a Gemini request error is marked unevaluated/error, never successful.

[`examples/rag_run_example.md`](examples/rag_run_example.md) records the generated lesson and actual structured evaluation from the local RAG end-to-end run performed for this project. It is one model result, not a guaranteed future response.

## Labeled retry demonstration

Select **Run rejection demo**. It makes real Gemini generation and evaluation calls, then deliberately changes the first attempt's beginner-friendly check to a clearly marked `DEMO ONLY` failure so the retry path can be shown on demand. The app retries using that rejection and evaluates the rewrite normally. Demo runs are not written to SQLite, so a fabricated check cannot pollute saved history or future memory. This is a demonstration control, not a genuine evaluator failure.

## Automated tests

```powershell
\.venv\Scripts\python.exe -m pytest
```

Tests use fake model callables and temporary SQLite files, so they make no Gemini requests. They cover PASS consistency, invalid JSON, early stop, two-retry cap, best attempt, rejection feedback in rewrite prompts, persistence across closing/reopening the database, retrieval into later prompts, and non-persistent demo behavior.

The pytest cache is disabled because this project may live in a OneDrive folder with restrictive cache-directory behavior. Pytest uses the normal system temporary folder for test databases.

## Manual 15–20 minute demonstration video outline

1. (1–2 min) Introduce the assignment and show the project folders/files; point out that `.env` is private and `.gitignore` excludes it.
2. (2 min) Explain the architecture: Streamlit UI → LangGraph → LangChain Gemini writer/reviewer → SQLite memory.
3. (2 min) Show `.env.example`, `requirements.txt`, and setup/launch commands. Do not show `.env` contents or the API key.
4. (4–5 min) Generate a lesson on **Retrieval-Augmented Generation (RAG)**. Show the generated explanation and each of the six evaluator results. State only what the displayed evaluator actually returned.
5. (3 min) Open attempt history and rejection reasons. If a genuine check fails, show how the specific feedback is included in a rewrite; otherwise use the clearly labeled **Run rejection demo** to show the same control flow without presenting it as a real failure.
6. (2 min) Show saved history and memory insight counts. Close/reopen the app and show the saved record remains in `data/lessons.db`.
7. (2–3 min) Run `python -m pytest` and explain the retry-cap, JSON, memory, and demo tests.
8. (1 min) Summarize limitations: the evaluator is an LLM review rather than a fact database; a valid key, model access, quota, and network are required; inspect factual claims before teaching.

## Limitations

- Accuracy is judged by the configured Gemini evaluator and is not independently fact-checked against sources.
- API quotas, model access, network availability, and provider behavior can interrupt a run. Such a run is shown as an error or unevaluated, never a pass.
- Memory retrieval uses lexical topic/reason overlap in local SQLite, not an embedding/vector database.
- The demo intentionally alters the first check and must not be cited as a genuine model finding.
