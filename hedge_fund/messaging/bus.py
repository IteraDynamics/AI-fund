"""
SQLite-backed message bus with full audit logging.

Messages are stored in two tables:
  - messages: the active inbox/outbox queue
  - audit_log: immutable append-only record of every message ever sent

Agents call send() to publish and receive() to pull pending messages from
their inbox. All operations are thread-safe via SQLite WAL mode.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator, Optional

from hedge_fund.config import AUDIT_DB_PATH
from hedge_fund.models.schemas import AgentMessage, MessagePriority


_lock = threading.Lock()

PRIORITY_ORDER = {
    MessagePriority.URGENT: 0,
    MessagePriority.HIGH: 1,
    MessagePriority.NORMAL: 2,
    MessagePriority.LOW: 3,
}


@contextmanager
def _get_conn(db_path: str = AUDIT_DB_PATH) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_bus(db_path: str = AUDIT_DB_PATH) -> None:
    """Create tables if they don't exist. Call once at startup."""
    with _get_conn(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS messages (
                message_id   TEXT PRIMARY KEY,
                sender       TEXT NOT NULL,
                recipient    TEXT NOT NULL,
                subject      TEXT NOT NULL,
                body         TEXT NOT NULL,
                payload_json TEXT,
                priority     TEXT NOT NULL DEFAULT 'normal',
                thread_id    TEXT,
                requires_reply INTEGER DEFAULT 0,
                timestamp    TEXT NOT NULL,
                read_at      TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_messages_recipient
                ON messages (recipient, read_at);

            CREATE TABLE IF NOT EXISTS audit_log (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id   TEXT NOT NULL,
                sender       TEXT NOT NULL,
                recipient    TEXT NOT NULL,
                subject      TEXT NOT NULL,
                body         TEXT NOT NULL,
                payload_json TEXT,
                priority     TEXT NOT NULL,
                thread_id    TEXT,
                timestamp    TEXT NOT NULL,
                event_type   TEXT NOT NULL   -- 'sent' | 'received' | 'replied'
            );

            CREATE INDEX IF NOT EXISTS idx_audit_timestamp
                ON audit_log (timestamp);
            CREATE INDEX IF NOT EXISTS idx_audit_sender
                ON audit_log (sender);
            CREATE INDEX IF NOT EXISTS idx_audit_recipient
                ON audit_log (recipient);
        """)


def send(
    sender: str,
    recipient: str,
    subject: str,
    body: str,
    payload: Optional[dict] = None,
    priority: MessagePriority = MessagePriority.NORMAL,
    thread_id: Optional[str] = None,
    requires_reply: bool = False,
    db_path: str = AUDIT_DB_PATH,
) -> str:
    """Publish a message to a recipient's inbox. Returns the message_id."""
    message_id = str(uuid.uuid4())
    ts = datetime.utcnow().isoformat()
    payload_json = json.dumps(payload) if payload else None

    with _lock:
        with _get_conn(db_path) as conn:
            conn.execute(
                """INSERT INTO messages
                   (message_id, sender, recipient, subject, body,
                    payload_json, priority, thread_id, requires_reply, timestamp)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (message_id, sender, recipient, subject, body,
                 payload_json, priority.value, thread_id,
                 int(requires_reply), ts),
            )
            conn.execute(
                """INSERT INTO audit_log
                   (message_id, sender, recipient, subject, body,
                    payload_json, priority, thread_id, timestamp, event_type)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (message_id, sender, recipient, subject, body,
                 payload_json, priority.value, thread_id, ts, "sent"),
            )
    return message_id


def receive(
    agent_id: str,
    limit: int = 10,
    mark_read: bool = True,
    db_path: str = AUDIT_DB_PATH,
) -> list[AgentMessage]:
    """Fetch unread messages for agent_id, ordered by priority then timestamp."""
    with _lock:
        with _get_conn(db_path) as conn:
            rows = conn.execute(
                """SELECT * FROM messages
                   WHERE recipient = ? AND read_at IS NULL
                   ORDER BY
                     CASE priority
                       WHEN 'urgent' THEN 0
                       WHEN 'high'   THEN 1
                       WHEN 'normal' THEN 2
                       WHEN 'low'    THEN 3
                     END,
                     timestamp ASC
                   LIMIT ?""",
                (agent_id, limit),
            ).fetchall()

            if mark_read and rows:
                now = datetime.utcnow().isoformat()
                ids = [r["message_id"] for r in rows]
                conn.execute(
                    f"UPDATE messages SET read_at=? WHERE message_id IN ({','.join('?'*len(ids))})",
                    [now, *ids],
                )
                for row in rows:
                    conn.execute(
                        """INSERT INTO audit_log
                           (message_id, sender, recipient, subject, body,
                            payload_json, priority, thread_id, timestamp, event_type)
                           VALUES (?,?,?,?,?,?,?,?,?,?)""",
                        (row["message_id"], row["sender"], row["recipient"],
                         row["subject"], row["body"], row["payload_json"],
                         row["priority"], row["thread_id"], now, "received"),
                    )

    messages = []
    for row in rows:
        payload = json.loads(row["payload_json"]) if row["payload_json"] else None
        messages.append(
            AgentMessage(
                message_id=row["message_id"],
                sender=row["sender"],
                recipient=row["recipient"],
                subject=row["subject"],
                body=row["body"],
                payload=payload,
                priority=MessagePriority(row["priority"]),
                thread_id=row["thread_id"],
                requires_reply=bool(row["requires_reply"]),
                timestamp=datetime.fromisoformat(row["timestamp"]),
            )
        )
    return messages


def get_audit_log(
    limit: int = 100,
    sender: Optional[str] = None,
    recipient: Optional[str] = None,
    since: Optional[datetime] = None,
    db_path: str = AUDIT_DB_PATH,
) -> list[dict]:
    """Query the immutable audit log."""
    clauses = []
    params: list = []
    if sender:
        clauses.append("sender = ?")
        params.append(sender)
    if recipient:
        clauses.append("recipient = ?")
        params.append(recipient)
    if since:
        clauses.append("timestamp >= ?")
        params.append(since.isoformat())

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)

    with _get_conn(db_path) as conn:
        rows = conn.execute(
            f"SELECT * FROM audit_log {where} ORDER BY timestamp DESC LIMIT ?",
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def get_thread(thread_id: str, db_path: str = AUDIT_DB_PATH) -> list[dict]:
    """Retrieve all messages in a conversation thread."""
    with _get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM audit_log WHERE thread_id=? AND event_type='sent' ORDER BY timestamp",
            (thread_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def queue_depth(agent_id: str, db_path: str = AUDIT_DB_PATH) -> int:
    """Return count of unread messages waiting for an agent."""
    with _get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM messages WHERE recipient=? AND read_at IS NULL",
            (agent_id,),
        ).fetchone()
    return row["cnt"]
