"""
Data models for conversations and messages.

Pure dataclasses with no framework dependency — usable by any layer.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Message:
    """A single message in a conversation."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: str = ""            # "user" | "assistant"
    content: str = ""
    timestamp: float = field(default_factory=time.time)
    recall_topk: Optional[Dict] = None  # MSA retrieved doc IDs per layer (optional)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "recall_topk": self.recall_topk,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        return cls(
            id=data["id"],
            role=data["role"],
            content=data["content"],
            timestamp=data["timestamp"],
            recall_topk=data.get("recall_topk"),
        )


@dataclass
class Conversation:
    """A conversation session containing ordered messages."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = "New Chat"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    messages: List[Message] = field(default_factory=list)
    metadata: Optional[Dict] = None  # Extensible: model info, config snapshot, etc.

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def turn_count(self) -> int:
        """Number of complete user-assistant turns."""
        return sum(1 for m in self.messages if m.role == "user")

    def add_message(self, role: str, content: str, recall_topk: Optional[Dict] = None) -> Message:
        """Create and append a message, update timestamp."""
        msg = Message(role=role, content=content, recall_topk=recall_topk)
        self.messages.append(msg)
        self.updated_at = time.time()
        return msg

    def get_history(self, max_turns: Optional[int] = None) -> List[Message]:
        """Return messages, optionally limited to the last N turns."""
        if max_turns is None:
            return list(self.messages)
        # Each turn = 1 user + 1 assistant message
        max_messages = max_turns * 2
        return self.messages[-max_messages:] if max_messages < len(self.messages) else list(self.messages)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [m.to_dict() for m in self.messages],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Conversation":
        return cls(
            id=data["id"],
            title=data["title"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            messages=[Message.from_dict(m) for m in data.get("messages", [])],
            metadata=data.get("metadata"),
        )
