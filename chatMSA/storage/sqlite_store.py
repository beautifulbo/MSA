"""
SQLite-backed conversation storage.

For file-based databases: creates a new connection per call (thread-safe).
For ":memory:": keeps a single persistent connection (tables would be lost otherwise).
"""

import json
import os
import sqlite3
import time
from typing import List, Optional

from chatMSA.models.conversation import Conversation, Message
from chatMSA.storage.base import BaseConversationStore


class SQLiteConversationStore(BaseConversationStore):
    """SQLite implementation of conversation persistence."""

    def __init__(self, db_path: str = "data/chat_msa.db"):
        self.db_path = db_path
        self._persistent_conn: Optional[sqlite3.Connection] = None
        # Ensure parent directory exists (skip for :memory:)
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

    def _get_conn(self) -> sqlite3.Connection:
        """
        Get a database connection.

        For :memory: databases, returns a single persistent connection
        (otherwise tables would be lost between calls).
        For file databases, creates a new connection each time (thread-safe).
        """
        if self.db_path == ":memory:":
            if self._persistent_conn is None:
                self._persistent_conn = sqlite3.connect(":memory:")
                self._persistent_conn.row_factory = sqlite3.Row
                self._persistent_conn.execute("PRAGMA foreign_keys=ON")
            return self._persistent_conn
        else:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            return conn

    def initialize(self) -> None:
        """Create tables if they don't exist."""
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id          TEXT PRIMARY KEY,
                    title       TEXT NOT NULL DEFAULT 'New Chat',
                    created_at  REAL NOT NULL,
                    updated_at  REAL NOT NULL,
                    metadata    TEXT           -- JSON-serialized dict
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id              TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content         TEXT NOT NULL,
                    timestamp       REAL NOT NULL,
                    recall_topk     TEXT,        -- JSON-serialized dict
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                    ON messages(conversation_id, timestamp);

                CREATE INDEX IF NOT EXISTS idx_conversations_updated
                    ON conversations(updated_at DESC);
            """)

    def save_conversation(self, conv: Conversation) -> None:
        """Insert or update conversation metadata."""
        metadata_json = json.dumps(conv.metadata) if conv.metadata else None
        with self._get_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO conversations (id, title, created_at, updated_at, metadata)
                   VALUES (?, ?, ?, ?, ?)""",
                (conv.id, conv.title, conv.created_at, conv.updated_at, metadata_json),
            )

    def load_conversation(self, conv_id: str) -> Optional[Conversation]:
        """Load a full conversation with all messages."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE id = ?", (conv_id,)
            ).fetchone()
            if row is None:
                return None

            messages = self._load_messages(conn, conv_id)
            metadata = json.loads(row["metadata"]) if row["metadata"] else None

            return Conversation(
                id=row["id"],
                title=row["title"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                messages=messages,
                metadata=metadata,
            )

    def list_conversations(self) -> List[Conversation]:
        """List all conversations (metadata only, no messages). Sorted by updated_at DESC."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM conversations ORDER BY updated_at DESC"
            ).fetchall()

            conversations = []
            for row in rows:
                metadata = json.loads(row["metadata"]) if row["metadata"] else None
                # Count messages for the summary
                msg_count = conn.execute(
                    "SELECT COUNT(*) as cnt FROM messages WHERE conversation_id = ?",
                    (row["id"],),
                ).fetchone()["cnt"]
                turn_count = conn.execute(
                    "SELECT COUNT(*) as cnt FROM messages WHERE conversation_id = ? AND role = 'user'",
                    (row["id"],),
                ).fetchone()["cnt"]

                conv = Conversation(
                    id=row["id"],
                    title=row["title"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    messages=[],  # Lightweight: no messages loaded
                    metadata=metadata,
                )
                # Attach counts as metadata for API convenience
                conv._message_count = msg_count   # type: ignore[attr-defined]
                conv._turn_count = turn_count     # type: ignore[attr-defined]
                conversations.append(conv)

            return conversations

    def delete_conversation(self, conv_id: str) -> bool:
        """Delete a conversation and all its messages."""
        with self._get_conn() as conn:
            cursor = conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
            return cursor.rowcount > 0

    def add_message(self, conv_id: str, message: Message) -> None:
        """Append a message to an existing conversation."""
        recall_topk_json = json.dumps(message.recall_topk) if message.recall_topk else None
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO messages (id, conversation_id, role, content, timestamp, recall_topk)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (message.id, conv_id, message.role, message.content, message.timestamp, recall_topk_json),
            )
            # Update conversation's updated_at
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (message.timestamp, conv_id),
            )

    def get_messages(self, conv_id: str) -> List[Message]:
        """Get all messages for a conversation, ordered by timestamp."""
        with self._get_conn() as conn:
            return self._load_messages(conn, conv_id)

    def _load_messages(self, conn: sqlite3.Connection, conv_id: str) -> List[Message]:
        """Internal: load messages from an existing connection."""
        rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC",
            (conv_id,),
        ).fetchall()

        messages = []
        for row in rows:
            recall_topk = json.loads(row["recall_topk"]) if row["recall_topk"] else None
            messages.append(Message(
                id=row["id"],
                role=row["role"],
                content=row["content"],
                timestamp=row["timestamp"],
                recall_topk=recall_topk,
            ))
        return messages

    def close(self) -> None:
        """Close the persistent connection if using :memory:."""
        if self._persistent_conn is not None:
            self._persistent_conn.close()
            self._persistent_conn = None
