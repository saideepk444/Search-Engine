import json
import sqlite3
from contextlib import contextmanager
from typing import Generator

import config


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


@contextmanager
def get_conn() -> Generator[sqlite3.Connection, None, None]:
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    import os
    os.makedirs(config.DATA_DIR, exist_ok=True)
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS pages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                url         TEXT UNIQUE NOT NULL,
                title       TEXT,
                body        TEXT,
                links_json  TEXT DEFAULT '[]',
                crawled_at  TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS doc_stats (
                doc_id      INTEGER PRIMARY KEY REFERENCES pages(id),
                token_count INTEGER NOT NULL,
                avg_dl      REAL    -- filled in after full corpus is known
            );

            CREATE TABLE IF NOT EXISTS index_entries (
                token   TEXT    NOT NULL,
                doc_id  INTEGER NOT NULL REFERENCES pages(id),
                tf      REAL    NOT NULL,
                positions TEXT  DEFAULT '[]',
                PRIMARY KEY (token, doc_id)
            );
            CREATE INDEX IF NOT EXISTS idx_token ON index_entries(token);

            CREATE TABLE IF NOT EXISTS pagerank (
                doc_id  INTEGER PRIMARY KEY REFERENCES pages(id),
                score   REAL NOT NULL DEFAULT 0.0
            );
        """)


# --- pages ---

def insert_page(url: str, title: str, body: str, links: list[str]) -> int | None:
    with get_conn() as conn:
        try:
            cur = conn.execute(
                "INSERT INTO pages (url, title, body, links_json) VALUES (?, ?, ?, ?)",
                (url, title, body, json.dumps(links)),
            )
            return cur.lastrowid
        except sqlite3.IntegrityError:
            return None  # already crawled


def get_all_pages() -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute("SELECT id, url, title, body, links_json FROM pages").fetchall()


def get_page_count() -> int:
    with get_conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]


# --- index ---

def upsert_index_entries(entries: list[tuple]) -> None:
    """entries: [(token, doc_id, tf, positions_json)]"""
    with get_conn() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO index_entries (token, doc_id, tf, positions) VALUES (?, ?, ?, ?)",
            entries,
        )


def upsert_doc_stats(stats: list[tuple], avg_dl: float) -> None:
    """stats: [(doc_id, token_count)]"""
    with get_conn() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO doc_stats (doc_id, token_count, avg_dl) VALUES (?, ?, ?)",
            [(doc_id, tc, avg_dl) for doc_id, tc in stats],
        )


def get_postings(token: str) -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT doc_id, tf FROM index_entries WHERE token = ?", (token,)
        ).fetchall()


def get_doc_stats(doc_id: int) -> sqlite3.Row | None:
    with get_conn() as conn:
        return conn.execute(
            "SELECT token_count, avg_dl FROM doc_stats WHERE doc_id = ?", (doc_id,)
        ).fetchone()


def get_avg_dl() -> float:
    with get_conn() as conn:
        row = conn.execute("SELECT avg_dl FROM doc_stats LIMIT 1").fetchone()
        return row["avg_dl"] if row else 0.0


def get_doc_count() -> int:
    return get_page_count()


# --- pagerank ---

def upsert_pagerank(scores: dict[int, float]) -> None:
    with get_conn() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO pagerank (doc_id, score) VALUES (?, ?)",
            scores.items(),
        )


def get_pagerank_scores() -> dict[int, float]:
    with get_conn() as conn:
        rows = conn.execute("SELECT doc_id, score FROM pagerank").fetchall()
        return {r["doc_id"]: r["score"] for r in rows}
