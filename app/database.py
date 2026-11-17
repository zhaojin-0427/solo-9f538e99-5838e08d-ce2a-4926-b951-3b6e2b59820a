"""SQLite schema and connection management.

Revision numbers are unsigned 64-bit integers (1 .. 2**64-1).  SQLite only
supports *signed* 64-bit integers, so revisions are stored as canonical
decimal TEXT and all new/old comparisons happen as Python ``int`` values in
the service layer.  Databases created by older versions (INTEGER revision
columns) are migrated in-place on startup via a table rebuild; every stored
fragment and publication row is preserved byte-for-byte apart from the
numeric-to-text encoding of the revision column.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .config import DB_PATH

# Bumped whenever the schema shape changes; drives startup migrations.
SCHEMA_VERSION = 2

# Canonical positive decimal integer, no sign, no leading zeroes, at most
# 20 digits (2**64 - 1 = 18446744073709551615 has 20 digits).  The service
# layer additionally validates the uint64 upper bound in Pydantic.
REVISION_CHECK = (
    "revision TEXT NOT NULL CHECK ("
    "revision GLOB '[1-9]*' "
    "AND revision NOT GLOB '*[^0-9]*' "
    "AND length(revision) BETWEEN 1 AND 20)"
)

SCHEMA = f"""
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS fragments (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    station      TEXT    NOT NULL,
    minute_utc   TEXT    NOT NULL,
    fragment_no  INTEGER NOT NULL,
    {REVISION_CHECK},
    start_sample INTEGER NOT NULL CHECK (start_sample >= 0 AND start_sample < 60),
    samples      TEXT    NOT NULL,
    content_hash TEXT    NOT NULL,
    UNIQUE (station, minute_utc, fragment_no)
);

CREATE INDEX IF NOT EXISTS idx_fragments_minute
    ON fragments (station, minute_utc);

CREATE TABLE IF NOT EXISTS publications (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    station          TEXT    NOT NULL,
    minute_utc       TEXT    NOT NULL,
    wave             TEXT    NOT NULL,
    fragment_set_hash TEXT   NOT NULL,
    stale            INTEGER NOT NULL DEFAULT 0 CHECK (stale IN (0, 1)),
    created_at       TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_publications_minute
    ON publications (station, minute_utc);

CREATE TABLE IF NOT EXISTS publication_items (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    publication_id INTEGER NOT NULL REFERENCES publications(id) ON DELETE CASCADE,
    fragment_no    INTEGER NOT NULL,
    {REVISION_CHECK},
    start_sample   INTEGER NOT NULL,
    samples        TEXT    NOT NULL,
    content_hash   TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pub_items_pub
    ON publication_items (publication_id);
"""

# Rebuild every revision-bearing table, copying values verbatim except for
# casting the INTEGER revision into canonical decimal text.  The DDL runs in
# one explicit transaction (SQLite supports transactional DDL), so a crash or
# error rolls the whole migration back instead of leaving a half-rebuilt
# schema.  foreign_keys must be toggled outside the transaction — the pragma
# is a no-op inside one — which also lets publications be rebuilt while its
# child table publication_items still references it.
MIGRATION_SQL = """
PRAGMA foreign_keys = OFF;
BEGIN;

CREATE TABLE fragments_new (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    station      TEXT    NOT NULL,
    minute_utc   TEXT    NOT NULL,
    fragment_no  INTEGER NOT NULL,
    revision     TEXT    NOT NULL,
    start_sample INTEGER NOT NULL CHECK (start_sample >= 0 AND start_sample < 60),
    samples      TEXT    NOT NULL,
    content_hash TEXT    NOT NULL,
    UNIQUE (station, minute_utc, fragment_no)
);
INSERT INTO fragments_new
    (id, station, minute_utc, fragment_no, revision, start_sample, samples, content_hash)
SELECT id, station, minute_utc, fragment_no, CAST(revision AS TEXT),
       start_sample, samples, content_hash
FROM fragments;
DROP TABLE fragments;
ALTER TABLE fragments_new RENAME TO fragments;
CREATE INDEX idx_fragments_minute ON fragments (station, minute_utc);

CREATE TABLE publications_new (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    station          TEXT    NOT NULL,
    minute_utc       TEXT    NOT NULL,
    wave             TEXT    NOT NULL,
    fragment_set_hash TEXT   NOT NULL,
    stale            INTEGER NOT NULL DEFAULT 0 CHECK (stale IN (0, 1)),
    created_at       TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
INSERT INTO publications_new
    (id, station, minute_utc, wave, fragment_set_hash, stale, created_at)
SELECT id, station, minute_utc, wave, fragment_set_hash, stale, created_at
FROM publications;
DROP TABLE publications;
ALTER TABLE publications_new RENAME TO publications;
CREATE INDEX idx_publications_minute ON publications (station, minute_utc);

CREATE TABLE publication_items_new (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    publication_id INTEGER NOT NULL REFERENCES publications(id) ON DELETE CASCADE,
    fragment_no    INTEGER NOT NULL,
    revision       TEXT    NOT NULL,
    start_sample   INTEGER NOT NULL,
    samples        TEXT    NOT NULL,
    content_hash   TEXT    NOT NULL
);
INSERT INTO publication_items_new
    (id, publication_id, fragment_no, revision, start_sample, samples, content_hash)
SELECT id, publication_id, fragment_no, CAST(revision AS TEXT),
       start_sample, samples, content_hash
FROM publication_items;
DROP TABLE publication_items;
ALTER TABLE publication_items_new RENAME TO publication_items;
CREATE INDEX idx_pub_items_pub ON publication_items (publication_id);

PRAGMA user_version = 2;
COMMIT;
PRAGMA foreign_keys = ON;
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    # Writers take the write lock up front so concurrent uploads/publishes
    # serialize instead of failing with SQLITE_BUSY on commit.
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def _revision_column_is_text(conn: sqlite3.Connection) -> bool:
    """True when ``fragments.revision`` already has TEXT affinity (new schema)."""
    for col in conn.execute("PRAGMA table_info(fragments)").fetchall():
        if col["name"] == "revision":
            return col["type"].upper() == "TEXT"
    return False


def _needs_revision_migration(conn: sqlite3.Connection) -> bool:
    # A pre-existing database with the old INTEGER revision column must be
    # rebuilt; a brand-new database has no fragments table at this point.
    table_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'fragments'"
    ).fetchone()
    return table_exists is not None and not _revision_column_is_text(conn)


def init_db(db_path: Path | None = None) -> None:
    """Create or migrate the schema (safe to run on every startup)."""
    path = db_path or DB_PATH
    conn = _connect(path)
    try:
        if _needs_revision_migration(conn):
            conn.executescript(MIGRATION_SQL)
            violations = conn.execute("PRAGMA foreign_key_check").fetchall()
            if violations:  # pragma: no cover - defensive integrity guard
                raise RuntimeError(
                    f"revision migration broke referential integrity: {violations!r}"
                )
        conn.executescript(SCHEMA)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()
    finally:
        conn.close()


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    """Yield a request-scoped connection; the service layer owns commits."""
    conn = _connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()
