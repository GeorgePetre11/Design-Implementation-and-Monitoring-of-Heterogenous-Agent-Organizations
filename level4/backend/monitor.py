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


# Initialize DB on import
init_db()
