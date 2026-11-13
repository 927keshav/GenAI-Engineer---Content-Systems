"""Persistent lesson history and categorized review failures."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "are",
    "was",
    "how",
    "why",
    "what",
    "does",
    "into",
    "about",
    "your",
    "you",
    "its",
    "have",
    "has",
}


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _tokens(text: str) -> set[str]:
    cleaned: list[str] = []
    for token in re.findall(r"[a-z0-9]+", (text or "").lower()):
        if token in _STOPWORDS or len(token) <= 2:
            continue
        if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        cleaned.append(token)
    return set(cleaned)


def _loads(text: str | None) -> Any:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


class MemoryStore:
    """One local SQLite file. All writes are serialized with a lock."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._create_tables()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def _create_tables(self) -> None:
        with self._lock:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS generations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    model_name TEXT NOT NULL DEFAULT '',
                    final_lesson TEXT,
                    final_evaluation_json TEXT,
                    status TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    overall TEXT NOT NULL,
                    best_attempt INTEGER,
                    pass_count INTEGER NOT NULL DEFAULT 0,
                    remaining_failures_json TEXT NOT NULL DEFAULT '[]',
                    disclosure TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    generation_id INTEGER NOT NULL,
                    attempt_number INTEGER NOT NULL,
                    lesson TEXT NOT NULL,
                    evaluation_json TEXT NOT NULL,
                    pass_count INTEGER NOT NULL,
                    overall TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (generation_id) REFERENCES generations(id)
                );

                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    generation_id INTEGER NOT NULL,
                    attempt_number INTEGER NOT NULL,
                    topic TEXT NOT NULL,
                    category TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (generation_id) REFERENCES generations(id)
                );

                CREATE INDEX IF NOT EXISTS idx_feedback_topic ON feedback(topic);
                CREATE INDEX IF NOT EXISTS idx_attempts_generation ON attempts(generation_id);
                """
            )
            self.conn.commit()

    def clear(self) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM feedback")
            self.conn.execute("DELETE FROM attempts")
            self.conn.execute("DELETE FROM generations")
            self.conn.commit()

    def start_generation(self, topic: str, model_name: str) -> int:
        now = _now()
        with self._lock:
            cursor = self.conn.execute(
                """
                INSERT INTO generations (
                    topic, created_at, updated_at, model_name, status, outcome,
                    overall, disclosure
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (topic, now, now, model_name, "Running.", "in_progress", "PENDING", ""),
            )
            self.conn.commit()
            return int(cursor.lastrowid)

    def save_attempt(
        self,
        *,
        generation_id: int,
        attempt_number: int,
        lesson: str,
        evaluation: dict[str, Any],
    ) -> None:
        now = _now()
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO attempts (
                    generation_id, attempt_number, lesson, evaluation_json,
                    pass_count, overall, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    generation_id,
                    attempt_number,
                    lesson,
                    json.dumps(evaluation, ensure_ascii=False),
                    int(evaluation.get("pass_count") or 0),
                    str(evaluation.get("overall") or "FAIL"),
                    now,
                ),
            )
            self.conn.commit()

    def save_feedback(
        self,
        *,
        generation_id: int,
        attempt_number: int,
        topic: str,
        evaluation: dict[str, Any],
    ) -> None:
        """Store one row per failed check so later lessons can avoid the same mistake."""
        checks = evaluation.get("checks") or {}
        failed = list(evaluation.get("failed_checks") or [])
        if not failed:
            return
        now = _now()
        with self._lock:
            try:
                for key in failed:
                    item = checks.get(key) or {}
                    reason = item.get("reason") if isinstance(item, dict) else ""
                    self.conn.execute(
                        """
                        INSERT INTO feedback (
                            generation_id, attempt_number, topic, category, reason, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            generation_id,
                            attempt_number,
                            topic,
                            key,
                            (reason or "No reason was given.").strip(),
                            now,
                        ),
                    )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def finish_generation(
        self,
        generation_id: int,
        *,
        final_lesson: str,
        final_evaluation: dict[str, Any] | None,
        outcome: str,
        retry_count: int,
        overall: str,
        best_attempt: int | None,
        pass_count: int,
        remaining_failures: list[dict[str, str]],
        disclosure: str,
        status: str,
    ) -> None:
        now = _now()
        evaluation_json = (
            json.dumps(final_evaluation, ensure_ascii=False) if final_evaluation is not None else None
        )
        with self._lock:
            self.conn.execute(
                """
                UPDATE generations
                SET updated_at = ?, final_lesson = ?, final_evaluation_json = ?,
                    status = ?, outcome = ?, retry_count = ?, overall = ?,
                    best_attempt = ?, pass_count = ?, remaining_failures_json = ?,
                    disclosure = ?
                WHERE id = ?
                """,
                (
                    now,
                    final_lesson,
                    evaluation_json,
                    status,
                    outcome,
                    retry_count,
                    overall,
                    best_attempt,
                    pass_count,
                    json.dumps(remaining_failures, ensure_ascii=False),
                    disclosure,
                    generation_id,
                ),
            )
            self.conn.commit()

    def relevant_failures(self, topic: str, limit: int = 6) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT topic, category, reason, created_at
                FROM feedback
                ORDER BY id DESC
                LIMIT 200
                """
            ).fetchall()
        topic_tokens = _tokens(topic)
        topic_text = (topic or "").lower().strip()
        ranked: list[tuple[int, str, sqlite3.Row]] = []
        for row in rows:
            other = (row["topic"] or "").lower().strip()
            score = 2 * len(topic_tokens & _tokens(row["topic"]))
            score += len(topic_tokens & _tokens(row["reason"]))
            if topic_text and other and (topic_text == other or topic_text in other or other in topic_text):
                score += 8
            if score <= 0:
                continue
            ranked.append((score, row["created_at"] or "", row))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        seen: set[tuple[str, str]] = set()
        results: list[dict[str, Any]] = []
        for _score, _created, row in ranked:
            key = (row["category"], (row["reason"] or "").strip().lower())
            if key in seen:
                continue
            seen.add(key)
            results.append(
                {
                    "topic": row["topic"],
                    "category": row["category"],
                    "reason": row["reason"],
                    "created_at": row["created_at"],
                }
            )
            if len(results) >= limit:
                break
        return results

    def stats(self) -> dict[str, int]:
        with self._lock:
            generations = self.conn.execute("SELECT COUNT(*) FROM generations").fetchone()[0]
            failures = self.conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
            passed = self.conn.execute(
                "SELECT COUNT(*) FROM generations WHERE outcome = ?",
                ("passed",),
            ).fetchone()[0]
        return {
            "generations": int(generations),
            "failures": int(failures),
            "passed": int(passed),
        }

    def category_counts(self) -> list[tuple[str, int]]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT category, COUNT(*) AS total
                FROM feedback
                GROUP BY category
                ORDER BY total DESC, category ASC
                """
            ).fetchall()
        return [(row["category"], int(row["total"])) for row in rows]

    def recent_generations(self, limit: int = 8) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT id, topic, created_at, outcome, retry_count, overall,
                       best_attempt, pass_count, disclosure
                FROM generations
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_generation(self, generation_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM generations WHERE id = ?",
                (generation_id,),
            ).fetchone()
        if row is None:
            return None
        record = dict(row)
        record["final_evaluation"] = _loads(record.get("final_evaluation_json"))
        record["remaining_failures"] = _loads(record.get("remaining_failures_json")) or []
        return record

    def list_attempts(self, generation_id: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT attempt_number, lesson, evaluation_json, pass_count, overall, created_at
                FROM attempts
                WHERE generation_id = ?
                ORDER BY attempt_number ASC
                """,
                (generation_id,),
            ).fetchall()
        attempts = []
        for row in rows:
            item = dict(row)
            item["evaluation"] = _loads(item.get("evaluation_json"))
            attempts.append(item)
        return attempts
