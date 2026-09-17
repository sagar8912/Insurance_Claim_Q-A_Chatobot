"""Persistent conversation session management backed by SQLite."""
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

from ..config import settings
from ..utils.logger import get_logger

logger = get_logger("insurance_rag.session_service")


class SessionService:
    """Thread-safe SQLite service for persistent conversation sessions and message history."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or settings.CONVERSATION_DB_PATH
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create and configure a new SQLite connection."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=10.0,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for high concurrency
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_db(self) -> None:
        """Create conversation and messages tables if they do not exist."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
                );
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_conv_created 
                ON messages(conversation_id, created_at);
                """
            )
            conn.commit()
            logger.info(f"Initialized conversation database at {self.db_path}")

    def get_or_create_session(self, conversation_id: Optional[str] = None) -> str:
        """Ensure session exists or create a new one with a unique UUID."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            if conversation_id:
                cursor.execute("SELECT id FROM conversations WHERE id = ?", (conversation_id,))
                row = cursor.fetchone()
                if row:
                    return row["id"]

            new_id = conversation_id or str(uuid.uuid4())
            cursor.execute(
                """
                INSERT OR IGNORE INTO conversations (id, created_at, updated_at)
                VALUES (?, ?, ?)
                """,
                (new_id, now, now),
            )
            conn.commit()
            logger.info(f"Created new conversation session: {new_id}")
            return new_id

    def add_message(self, conversation_id: str, role: str, content: str) -> Dict[str, Any]:
        """Record a user or assistant message to persistent storage."""
        role_clean = role.strip().lower()
        if role_clean not in ("user", "assistant", "system"):
            role_clean = "user"

        conv_id = self.get_or_create_session(conversation_id)
        msg_id = f"msg_{uuid.uuid4().hex[:16]}"
        now = datetime.now(timezone.utc).isoformat()

        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO messages (id, conversation_id, role, content, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (msg_id, conv_id, role_clean, content, now),
            )
            cursor.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conv_id),
            )
            conn.commit()

        return {
            "id": msg_id,
            "conversation_id": conv_id,
            "role": role_clean,
            "content": content,
            "created_at": now,
        }

    def get_messages(self, conversation_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve conversation messages ordered chronologically."""
        if not conversation_id:
            return []

        limit_val = limit if limit and limit > 0 else settings.MAX_HISTORY_MESSAGES

        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            # Fetch the most recent N messages, then order chronologically
            query = """
                SELECT id, conversation_id, role, content, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """
            cursor.execute(query, (conversation_id, limit_val))
            rows = cursor.fetchall()

        messages = [
            {
                "id": row["id"],
                "conversation_id": row["conversation_id"],
                "role": row["role"],
                "content": row["content"],
                "created_at": row["created_at"],
            }
            for row in reversed(rows)
        ]
        return messages

    def format_chat_history_for_prompt(
        self, conversation_id: str, max_messages: Optional[int] = None
    ) -> str:
        """Format recent chat history into readable transcript for LLM context and query rewriting."""
        limit = max_messages or settings.MAX_HISTORY_MESSAGES
        messages = self.get_messages(conversation_id, limit=limit)
        if not messages:
            return ""

        formatted_turns = []
        for msg in messages:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            formatted_turns.append(f"{role_label}: {msg['content'].strip()}")

        return "\n".join(formatted_turns)

    def clear_session(self, conversation_id: str) -> bool:
        """Delete all messages associated with a conversation while keeping the conversation entry."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            now = datetime.now(timezone.utc).isoformat()
            cursor.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
            conn.commit()
            logger.info(f"Cleared messages for conversation: {conversation_id}")
            return True

    def delete_session(self, conversation_id: str) -> bool:
        """Completely delete conversation and its messages."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            conn.commit()
            logger.info(f"Deleted conversation: {conversation_id}")
            return True

    def count_messages(self, conversation_id: str) -> int:
        """Return total number of messages in a conversation."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM messages WHERE conversation_id = ?", (conversation_id,))
            row = cursor.fetchone()
            return row["count"] if row else 0

    def list_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List active conversations with last update time and message count."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT c.id, c.created_at, c.updated_at, COUNT(m.id) as message_count
                FROM conversations c
                LEFT JOIN messages m ON c.id = m.conversation_id
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            return [
                {
                    "id": row["id"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "message_count": row["message_count"],
                }
                for row in rows
            ]


# Singleton instance
_session_service: Optional[SessionService] = None


def get_session_service() -> SessionService:
    global _session_service
    if _session_service is None:
        _session_service = SessionService()
    return _session_service
