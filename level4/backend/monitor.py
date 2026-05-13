"""
Monitoring module -- reusable across all levels.
Logs structured events to SQLite for later visualization.
"""

import json
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).parent / "monitoring.db"


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                level       INTEGER NOT NULL,
                timestamp   REAL NOT NULL,
                event_type  TEXT NOT NULL,
                agent_name  TEXT,
                data        TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id    TEXT PRIMARY KEY,
                question      TEXT NOT NULL,
                status        TEXT NOT NULL,
                started_at    REAL,
                completed_at  REAL,
                agent_outputs TEXT,
                full_report   TEXT,
                scorecard     TEXT,
                revisions     TEXT
            )
        """)
        conn.commit()


def log_event(
    session_id: str,
    event_type: str,
    level: int = 4,
    agent_name: str | None = None,
    data: dict | None = None,
):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO events (session_id, level, timestamp, event_type, agent_name, data)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                level,
                time.time(),
                event_type,
                agent_name,
                json.dumps(data) if data else None,
            ),
        )
        conn.commit()


def get_events(session_id: str | None = None) -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        if session_id:
            rows = conn.execute(
                "SELECT * FROM events WHERE session_id = ? ORDER BY timestamp",
                (session_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM events ORDER BY timestamp DESC LIMIT 200"
            ).fetchall()
        return [dict(r) for r in rows]


def save_session(
    session_id: str,
    question: str,
    status: str,
    started_at: float | None,
    agent_outputs: dict | None = None,
    full_report: str = "",
    scorecard: dict | None = None,
    revisions: dict | None = None,
):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO sessions
                (session_id, question, status, started_at, completed_at,
                 agent_outputs, full_report, scorecard, revisions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                question,
                status,
                started_at,
                time.time(),
                json.dumps(agent_outputs) if agent_outputs else None,
                full_report,
                json.dumps(scorecard) if scorecard else None,
                json.dumps(revisions) if revisions else None,
            ),
        )
        conn.commit()


def get_session(session_id: str) -> dict | None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        for key in ("agent_outputs", "scorecard", "revisions"):
            if d.get(key):
                try:
                    d[key] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d


def list_sessions(limit: int = 20) -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT session_id, question, status, started_at, completed_at,
                   scorecard
            FROM sessions
            ORDER BY completed_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            if d.get("scorecard"):
                try:
                    sc = json.loads(d["scorecard"])
                    d["overall_score"] = sc.get("overall_score")
                except (json.JSONDecodeError, TypeError):
                    d["overall_score"] = None
            else:
                d["overall_score"] = None
            del d["scorecard"]
            results.append(d)
        return results


# Initialize DB on import
init_db()
