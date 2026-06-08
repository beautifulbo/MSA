"""
Abstract storage interface for conversation persistence.

Implement this interface to swap storage backends (SQLite, PostgreSQL, Redis, etc.)
without changing any business logic.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from chatMSA.models.conversation import Conversation, Message


class BaseConversationStore(ABC):
    """Abstract base class for conversation persistence."""

    @abstractmethod
    def initialize(self) -> None:
        """Create tables / indices if they don't exist. Called once at startup."""
        ...

    @abstractmethod
    def save_conversation(self, conv: Conversation) -> None:
        """Insert or update a conversation (metadata only, not messages)."""
        ...

    @abstractmethod
    def load_conversation(self, conv_id: str) -> Optional[Conversation]:
        """Load a full conversation with all messages. Returns None if not found."""
        ...

    @abstractmethod
    def list_conversations(self) -> List[Conversation]:
        """List all conversations (metadata only, no messages). Sorted by updated_at DESC."""
        ...

    @abstractmethod
    def delete_conversation(self, conv_id: str) -> bool:
        """Delete a conversation and all its messages. Returns True if found and deleted."""
        ...

    @abstractmethod
    def add_message(self, conv_id: str, message: Message) -> None:
        """Append a message to an existing conversation."""
        ...

    @abstractmethod
    def get_messages(self, conv_id: str) -> List[Message]:
        """Get all messages for a conversation, ordered by timestamp."""
        ...

    def close(self) -> None:
        """Release resources. Override if needed."""
        pass
