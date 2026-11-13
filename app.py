"""Streamlit front end for the self-evaluating lesson generator."""

from __future__ import annotations

import html
import json

import streamlit as st

from config import NO_KEY_MESSAGE, api_key_configured, settings_summary
from evaluation import LABELS, is_complete_pass, trusted_outcome
from graph.state import clean_topic
from graph.workflow import build_lesson_graph, run_lesson_graph
from memory.store import MemoryStore
from safety import public_error

PASSED = "passed"
BEST_EFFORT = "best_effort"


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1120px;}
        .eyebrow {
            font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase;
            color: #1f4b4a; font-weight: 700; margin: 0;
        }
        h1.app-title {margin: 0.15rem 0 0.35rem; font-size: 2.05rem; line-height: 1.15;}
        .lede {margin: 0; color: #3d4a4a; font-size: 1.05rem; max-width: 760px;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_store(path: str) -> MemoryStore:
    return MemoryStore(path)


@st.cache_resource
def cached_store(path: str) -> MemoryStore:
    return get_store(path)


def render_sidebar(store: MemoryStore) -> None:
    summary = settings_summary()
    st.sidebar.subheader("Setup")
    key_label = "configured" if summary["api_key_configured"] else "missing"
    st.sidebar.write(f"API key: {key_label}")
    st.sidebar.write(f"Model: {summary['model']}")
    st.sidebar.caption(summary["database"])
    st.sidebar.write(f"Retries: {summary['max_retries']} maximum")
    st.sidebar.divider()
    st.sidebar.subheader("Saved memory")
    stats = store.stats()
    st.sidebar.write(f"Lessons saved: {stats['generations']}")
    st.sidebar.write(f"Failure notes: {stats['failures']}")
    st.sidebar.write(f"Lessons that passed: {stats['passed']}")
    counts = store.category_counts()
    if counts:
        st.sidebar.caption("Failures by check")
        for key, count in counts:
            st.sidebar.write(f"{LABELS.get(key, key)}: {count}")
    wipe = st.sidebar.checkbox("Delete saved lessons and feedback")
    if st.sidebar.button("Clear memory", disabled=not wipe):
        store.clear()
        st.sidebar.success("Saved memory was cleared.")
        st.rerun()


def rubric_card(label: str, result: str, reason: str) -> None:
    if result == "PASS":
        color, bg, border = "#146c43", "#e8f6ee", "#b7e4c7"
    elif result == "FAIL":
        color, bg, border = "#9b1c1c", "#fdecec", "#f5c2c2"
    else:
        color, bg, border = "#7a5b00", "#fff6db", "#f0e0a8"
    st.markdown(
        f"""
        <div style="background:{bg};border:1px solid {border};border-radius:10px;
                    padding:0.75rem 0.85rem;min-height:150px;margin-bottom:0.75rem;">
          <div style="color:{color};font-weight:700;letter-spacing:0.04em;font-size:0.75rem;">{html.escape(result)}</div>
          <div style="font-weight:650;margin:0.2rem 0 0.35rem;">{html.escape(label)}</div>
          <div style="font-size:0.92rem;line-height:1.4;">{html.escape(reason)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_rubric(evaluation: dict | None, outcome: str) -> None:
    st.subheader("Rubric")
    columns = st.columns(3)
    for index, (key, label) in enumerate(LABELS.items()):
        if evaluation and isinstance(evaluation.get("checks"), dict):
            item = evaluation["checks"].get(key) or {}
            result = item.get("result") or "NOT EVALUATED"
            reason = item.get("reason") or "No reason was recorded."
            if result not in {"PASS", "FAIL"}:
                result = "NOT EVALUATED"
        else:
            result = "NOT EVALUATED"
            reason = "This check was not returned, so it is not a pass."
        with columns[index % 3]:
            rubric_card(label, result, reason)
    if outcome == PASSED and not is_complete_pass(evaluation):
        st.error("This result was marked passed without six PASS checks. It is shown as not passed.")
    elif evaluation and evaluation.get("overall_corrected"):
        st.caption("The overall flag was adjusted so it matches the six checks.")


def render_result(result: dict) -> None:
    evaluation = result.get("final_evaluation")
    outcome = trusted_outcome(result.get("outcome") or "", evaluation)
    disclosure = result.get("disclosure") or ""
    if result.get("demo_mode"):
        st.info(
            "DEMO ONLY: the first rejection was controlled and is not a genuine evaluator finding. "
            "This run was not saved to SQLite."
        )
    if outcome == PASSED and is_complete_pass(evaluation):
        st.success(disclosure)
    elif outcome == "error":
        st.error(disclosure or "The lesson could not be generated.")
    else:
        st.warning(disclosure)

    pass_count = result.get("pass_count") or 0
    metric_cols = st.columns(4)
    metric_cols[0].metric(
        "Demo attempts" if result.get("demo_mode") else "Attempts saved",
        result.get("attempt") or len(result.get("history") or []),
    )
    metric_cols[1].metric("Retries used", result.get("retry_count") or 0)
    metric_cols[2].metric("Checks passed", f"{pass_count} of 6" if evaluation else "—")
    metric_cols[3].metric("Result", outcome.replace("_", " "))

    if result.get("memory_warning"):
        st.caption(f"Memory note: {result['memory_warning']}")

    with st.expander("Generation status", expanded=True):
        for line in result.get("status_log") or []:
            st.write(line)

    lesson = (result.get("final_lesson") or "").strip()
    st.subheader("Lesson")
    if lesson:
        with st.container(border=True):
            st.markdown(lesson)
        st.download_button(
            "Download lesson (.md)",
            data=lesson,
            file_name="lesson.md",
            mime="text/markdown",
        )
    else:
        st.info("No lesson text was produced.")

    st.caption(
        "Accuracy is judged by the same Gemini model, not by a separate fact database. "
        "Read the accuracy reason before you teach from this lesson."
    )
    render_rubric(evaluation, outcome)

    remaining = result.get("remaining_failures") or []
    st.subheader("Remaining failures")
    if remaining:
        for item in remaining:
            st.write(f"**{item.get('label', item.get('key', 'Check'))}.** {item.get('reason', '')}")
    elif outcome == PASSED:
        st.write("None. Every check passed on the lesson shown.")
    else:
        st.write("No completed review is available, so there is no pass to claim.")

    st.subheader("Retry history")
    history = result.get("history") or []
    if not history:
        st.write("No completed attempts.")
    else:
        rows = []
        shown = result.get("best_attempt_number")
        for item in history:
            ev = item.get("evaluation") or {}
            failed = ev.get("failed_checks") or []
            rows.append(
                {
                    "Attempt": item.get("attempt"),
                    "Result": ev.get("overall") or "UNEVALUATED",
                    "Checks passed": f"{ev.get('pass_count', 0)} of 6" if ev else "Not evaluated",
                    "Failed checks": ", ".join(LABELS.get(key, key) for key in failed) or ("—" if ev else "Not evaluated"),
                    "Shown": "Yes" if item.get("attempt") == shown else "",
                }
            )
        st.table(rows)
        if len(history) > 1:
            for item in history:
                label = f"Attempt {item.get('attempt')} lesson"
                if item.get("attempt") == shown:
                    label += " (shown)"
                with st.expander(label):
                    st.markdown(item.get("lesson") or "")

    st.subheader("Rejection log")
    logged = False
    for item in history:
        reasons = (item.get("evaluation") or {}).get("rejection_reasons") or []
        if not reasons:
            continue
        logged = True
        st.markdown(f"**Attempt {item.get('attempt')}**")
        for reason in reasons:
            st.write(f"- {reason}")
    if not logged:
        if outcome == PASSED:
            st.write("No checks failed, so the rejection log is empty.")
        else:
            st.write("No rejection reasons were recorded.")

    st.subheader("Memory insights")
    hits = result.get("memory_hits") or []
    if not hits:
        st.write("No related past failures were used in this prompt.")
    else:
        st.write("These earlier failures were placed in the generation prompt:")
        for hit in hits:
            label = LABELS.get(hit.get("category", ""), hit.get("category", "Check"))
            st.write(f"- **{label}** on “{hit.get('topic', '')}”: {hit.get('reason', '')}")

    with st.expander("Evaluator JSON for the lesson shown"):
        if evaluation:
            st.json(evaluation)
        else:
            st.write("The evaluator did not return JSON for this lesson.")


def render_saved(store: MemoryStore) -> None:
    st.divider()
    st.subheader("Saved lessons")
    records = store.recent_generations(limit=12)
    if not records:
        st.write("No lessons have been saved yet.")
        return
    options = {
        f"#{row['id']} · {row['topic']} · {row['outcome']} · {row['created_at']}": row["id"]
        for row in records
    }
    choice = st.selectbox("Open a saved run", [" "] + list(options))
    if not choice or choice == " ":
        return
    record = store.get_generation(options[choice])
    if record is None:
        st.write("That saved run could not be read.")
        return
    evaluation = record.get("final_evaluation")
    outcome = trusted_outcome(record.get("outcome") or "", evaluation)
    st.write(record.get("disclosure") or "")
    st.write(
        f"Retries recorded: {record.get('retry_count', 0)}. "
        f"Checks passed on the saved lesson: {record.get('pass_count', 0)} of 6."
    )
    if outcome == PASSED and not is_complete_pass(evaluation):
        st.error("This saved row says passed, but the stored checks do not all pass.")
    if record.get("final_lesson"):
        with st.container(border=True):
            st.markdown(record["final_lesson"])
    if evaluation:
        render_rubric(evaluation, outcome)
    attempts = store.list_attempts(record["id"])
    if attempts:
        st.markdown("**Saved attempts**")
        for attempt in attempts:
            ev = attempt.get("evaluation") or {}
            failed = ", ".join(LABELS.get(key, key) for key in (ev.get("failed_checks") or [])) or "none"
            if ev.get("overall") == "UNEVALUATED":
                st.write(
                    f"Attempt {attempt['attempt_number']}: UNEVALUATED. "
                    f"The review could not finish. {ev.get('error', '')}"
                )
            else:
                st.write(
                    f"Attempt {attempt['attempt_number']}: {ev.get('overall', '—')} "
                    f"({attempt.get('pass_count', 0)} of 6). Failed: {failed}."
                )


def main() -> None:
    st.set_page_config(
        page_title="Self-Evaluating Lesson Content Generator",
        page_icon=":material/menu_book:",
        layout="wide",
    )
    inject_css()
    st.markdown('<p class="eyebrow">Lesson studio</p>', unsafe_allow_html=True)
    st.markdown(
        '<h1 class="app-title">Self-Evaluating Lesson Content Generator</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="lede">Enter a topic. The app writes a standalone beginner lesson, '
        "grades it on six pass/fail checks, and rewrites it up to two times if a check fails.</p>",
        unsafe_allow_html=True,
    )
    summary = settings_summary()
    try:
        store = cached_store(str(summary["database"]))
    except Exception as exc:
        st.error(public_error(exc))
        st.stop()
    render_sidebar(store)

    if not api_key_configured():
        st.info(NO_KEY_MESSAGE)

    topic_input = st.text_input(
        "Topic",
        placeholder="How rain forms",
        help="One subject is enough. Example: what is a fraction.",
    )
    action_col, demo_col = st.columns([2, 1])
    with action_col:
        generate = st.button(
            "Generate lesson",
            type="primary",
            disabled=not summary["api_key_configured"],
        )
    with demo_col:
        demo_generate = st.button(
            "Run rejection demo",
            help="Runs Gemini, then inserts a clearly labeled demo-only failure on attempt 1 to show the retry. Demo data is not saved.",
            disabled=not summary["api_key_configured"],
        )

    if generate or demo_generate:
        try:
            topic = clean_topic(topic_input)
        except ValueError as exc:
            st.warning(str(exc))
        else:
            try:
                from llm_client import make_callables

                generate_fn, evaluate_fn = make_callables()
                graph = build_lesson_graph(store, generate_fn, evaluate_fn)
                with st.status("Working on the lesson...", expanded=True) as status_box:
                    seen: list[str] = []

                    def on_status(snapshot: dict) -> None:
                        message = snapshot.get("status") or ""
                        if message and (not seen or seen[-1] != message):
                            seen.append(message)
                            status_box.write(message)

                    result = run_lesson_graph(
                        graph,
                        topic,
                        model_name=str(summary["model"]),
                        on_status=on_status,
                        demo_mode=demo_generate,
                    )
                    if demo_generate:
                        result["disclosure"] = (
                            "DEMO ONLY: Attempt 1 includes a controlled rejection to demonstrate regeneration. "
                            "This is not a genuine evaluator failure, and this demo was not saved to SQLite."
                        )
                        result["demo_mode"] = True
                    outcome = trusted_outcome(
                        result.get("outcome") or "",
                        result.get("final_evaluation"),
                    )
                    if outcome == "error":
                        status_box.update(label="Generation stopped.", state="error")
                    elif outcome == PASSED:
                        status_box.update(label="All 6 checks passed.", state="complete")
                    else:
                        status_box.update(label="Finished. Review the remaining failures.", state="complete")
                st.session_state["lesson_result"] = result
            except Exception as exc:
                st.error(public_error(exc))

    result = st.session_state.get("lesson_result")
    if result:
        render_result(result)
    elif not (generate or demo_generate):
        left, middle, right = st.columns(3)
        left.markdown("**1. Draft**\n\nA short lesson for a beginner, with examples and defined words.")
        middle.markdown("**2. Review**\n\nSix pass/fail checks. A pass requires every check to pass.")
        right.markdown("**3. Rewrite**\n\nFailed checks are sent back. After two retries, the best draft is shown with its failures.")

    render_saved(store)
    st.caption("Keys stay in the environment. They are not written into the lesson database or this page.")
    _ = json  # kept available for debugging sessions without changing the UI


if __name__ == "__main__":
    main()
