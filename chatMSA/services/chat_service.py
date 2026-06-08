"""
Chat service — orchestrates conversation management, prompt building,
and MSA engine calls.

This is the core business logic layer. It knows nothing about HTTP or UI.
"""

import time
from typing import List, Optional

from chatMSA.config import ChatConfig
from chatMSA.models.conversation import Conversation, Message
from chatMSA.services.msa_engine_service import MSAEngineService
from chatMSA.services.prompt_builder import PromptBuilder
from chatMSA.storage.base import BaseConversationStore


class ChatService:
    """
    Orchestrates multi-turn conversation on MSA.

    Responsibilities:
        - CRUD for conversations (delegates to store)
        - Build prompts with history (delegates to PromptBuilder)
        - Call MSA engine for generation (delegates to MSAEngineService)
        - Persist messages after each turn
    """

    def __init__(
        self,
        engine: MSAEngineService,
        store: BaseConversationStore,
        config: ChatConfig,
    ):
        self.engine = engine
        self.store = store
        self.config = config
        self.prompt_builder = PromptBuilder(config)

    # ── Conversation CRUD ───────────────────────────────────────

    def create_conversation(self, title: Optional[str] = None) -> Conversation:
        """Create a new conversation and persist it."""
        conv = Conversation(title=title or "New Chat")
        self.store.save_conversation(conv)
        return conv

    def get_conversation(self, conv_id: str) -> Optional[Conversation]:
        """Load a full conversation with all messages."""
        return self.store.load_conversation(conv_id)

    def list_conversations(self) -> List[Conversation]:
        """List all conversations (lightweight, no messages)."""
        return self.store.list_conversations()

    def delete_conversation(self, conv_id: str) -> bool:
        """Delete a conversation and all its messages."""
        return self.store.delete_conversation(conv_id)

    def rename_conversation(self, conv_id: str, title: str) -> Optional[Conversation]:
        """Rename a conversation. Returns updated conversation or None if not found."""
        conv = self.store.load_conversation(conv_id)
        if conv is None:
            return None
        conv.title = title
        self.store.save_conversation(conv)
        return conv

    # ── Chat ────────────────────────────────────────────────────

    def send_message(self, conv_id: str, user_message: str) -> Message:
        """
        Send a user message and get the assistant's response.

        Flow:
            1. Load conversation history from store
            2. Persist the user message
            3. Build prompt with PromptBuilder (history + current query)
            4. Call MSAEngineService.generate()
            5. Parse response, create assistant Message
            6. Persist the assistant message
            7. Return assistant Message

        Raises:
            ValueError: If conversation not found.
            RuntimeError: If engine is not ready.
        """
        # 1. Load conversation
        conv = self.store.load_conversation(conv_id)
        if conv is None:
            raise ValueError(f"Conversation not found: {conv_id}")

        # 2. Persist user message
        user_msg = Message(role="user", content=user_message)
        self.store.add_message(conv_id, user_msg)
        conv.messages.append(user_msg)

        # 3. Build prompt with history
        history = conv.get_history(max_turns=self.config.max_history_turns)
        # Remove the just-added user message from history (it's the current query)
        history_without_current = history[:-1]
        prompt = self.prompt_builder.build(history_without_current, user_message)

        # 4. Generate response
        raw_response, recall_topk = self.engine.generate(prompt)

        # 5. Parse response
        assistant_content = self._parse_response(raw_response)

        # 6. Persist assistant message
        assistant_msg = Message(
            role="assistant",
            content=assistant_content,
            recall_topk=recall_topk if recall_topk else None,
        )
        self.store.add_message(conv_id, assistant_msg)

        # Auto-generate title from first user message if title is default
        if conv.title == "New Chat" and conv.turn_count == 1:
            auto_title = user_message[:50] + ("..." if len(user_message) > 50 else "")
            conv.title = auto_title
            self.store.save_conversation(conv)

        return assistant_msg

    def _parse_response(self, raw_response: str) -> str:
        """
        Extract the final answer from MSA's raw output.

        MSA outputs structured text with <think>...</think> blocks
        and "The answer to the question is:" markers. We extract
        the clean answer for display.
        """
        # Try to extract from structured output
        if "The answer to the question is:" in raw_response:
            answer = raw_response.split("The answer to the question is:")[-1]
            # Clean up common artifacts
            answer = answer.replace("</think>", "").strip()
            if "Answer:" in answer:
                answer = answer.split("Answer:")[-1].strip()
            return answer

        # Fallback: remove <think> blocks
        if "<think>" in raw_response and "</think>" in raw_response:
            before = raw_response.split("<think>")[0]
            after = raw_response.split("</think>")[-1]
            cleaned = (before + after).strip()
            if cleaned:
                return cleaned

        # Last resort: return as-is
        return raw_response.strip()
