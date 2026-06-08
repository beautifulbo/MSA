"""
Prompt construction with conversation history.

Assembles multi-turn history + current query into a single prompt string
that MSA's template system can process.
"""

from typing import List

from chatMSA.config import ChatConfig
from chatMSA.models.conversation import Message


class PromptBuilder:
    """Builds prompts that include conversation history for MSA inference."""

    def __init__(self, config: ChatConfig):
        self.config = config

    def build(self, history: List[Message], current_query: str) -> str:
        """
        Construct a single prompt string with conversation history.

        MSA's _apply_template() wraps the output in Qwen chat format,
        so we only produce the raw text content here.

        Args:
            history: Previous messages (user + assistant pairs).
            current_query: The latest user question.

        Returns:
            A prompt string ready to pass to MSAEngine.generate().

        Format example:
            Previous conversation:
            User: What is MSA?
            Assistant: MSA is a scalable sparse attention framework...

            User: How does it scale to 100M tokens?
        """
        if not history:
            return current_query

        parts = []

        # Build conversation history section
        history_lines = []
        for msg in history:
            prefix = "User" if msg.role == "user" else "Assistant"
            history_lines.append(f"{prefix}: {msg.content}")

        if history_lines:
            parts.append("Previous conversation:")
            parts.append("\n".join(history_lines))
            parts.append("")  # Blank line separator

        # Append current query
        parts.append(current_query)

        return "\n".join(parts)

    # TODO: Implement token-aware truncation.
    #
    # Currently we rely on max_history_turns to limit context size,
    # but this doesn't account for actual token counts. A proper implementation
    # would:
    #   1. Accept the tokenizer as a dependency (or from MSAEngineService)
    #   2. Count tokens in the assembled history
    #   3. Progressively remove oldest turns until under max_context_tokens
    #   4. Optionally add a "[earlier history truncated]" marker
    #
    # def build_with_token_budget(self, history, current_query, tokenizer):
    #     ...
