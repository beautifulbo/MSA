"""
Gradio frontend for chatMSA.

Layout:
    - Left sidebar: conversation list with "New Chat" button
    - Main area: chat history + input box
    - Status bar: engine status indicator
"""

from typing import List, Tuple

import gradio as gr

from chatMSA.services.chat_service import ChatService


def create_gradio_app(chat_service: ChatService) -> gr.Blocks:
    """
    Create and return the Gradio Blocks application.

    The returned app can be mounted onto FastAPI via gr.mount_gradio_app()
    or launched standalone.
    """
    with gr.Blocks(
        title="chatMSA",
        theme=gr.themes.Soft(),
        css="""
        .sidebar { min-width: 250px; max-width: 300px; }
        .conv-item { padding: 8px 12px; cursor: pointer; border-radius: 6px; margin: 2px 0; }
        .conv-item:hover { background: #e8e8e8; }
        .conv-item.active { background: #d0e0ff; font-weight: bold; }
        """,
    ) as app:

        # ── State ───────────────────────────────────────────────
        current_conv_id = gr.State(value=None)

        # ── Layout ──────────────────────────────────────────────
        gr.Markdown("# 💬 chatMSA\n*Multi-turn conversation on Memory Sparse Attention*")

        with gr.Row():
            # Left sidebar
            with gr.Column(scale=1, elem_classes="sidebar"):
                new_chat_btn = gr.Button("➕ New Chat", variant="primary", size="sm")
                gr.Markdown("### Conversations")
                conv_list = gr.HTML(value=_render_conv_list(chat_service, None))

            # Main chat area
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=500,
                    type="messages",
                    show_copy_button=True,
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Type your message...",
                        show_label=False,
                        scale=9,
                        container=False,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                status_text = gr.Markdown("*Ready*")

        # ── Event Handlers ──────────────────────────────────────

        def on_new_chat():
            """Create a new conversation and switch to it."""
            conv = chat_service.create_conversation()
            return (
                conv.id,                                    # current_conv_id
                [],                                         # chatbot (empty)
                _render_conv_list(chat_service, conv.id),   # conv_list
                gr.update(value=""),                        # msg_input
            )

        def on_select_conv(evt: gr.SelectData):
            """Switch to a selected conversation."""
            conv_id = evt.value
            conv = chat_service.get_conversation(conv_id)
            if conv is None:
                return (None, [], _render_conv_list(chat_service, None), "")
            messages = _conv_to_chatbot(conv)
            return (
                conv_id,
                messages,
                _render_conv_list(chat_service, conv_id),
                gr.update(value=""),
            )

        def on_send_message(user_text: str, conv_id: str, history: list):
            """Send a message and get the response."""
            if not user_text.strip():
                return history, gr.update(value=""), _render_conv_list(chat_service, conv_id), conv_id

            # Create conversation if none selected
            if conv_id is None:
                conv = chat_service.create_conversation()
                conv_id = conv.id

            # Add user message to chatbot immediately
            history = history + [{"role": "user", "content": user_text}]

            try:
                # Call chat service (this blocks until MSA responds)
                assistant_msg = chat_service.send_message(conv_id, user_text)
                history = history + [{"role": "assistant", "content": assistant_msg.content}]
                status = f"*Responded at {assistant_msg.timestamp:.0f}*"
            except RuntimeError as e:
                history = history + [{"role": "assistant", "content": f"⚠️ Error: {e}"}]
                status = f"*Error: {e}*"
            except Exception as e:
                history = history + [{"role": "assistant", "content": f"⚠️ Unexpected error: {e}"}]
                status = f"*Error: {e}*"

            return (
                history,
                gr.update(value=""),
                _render_conv_list(chat_service, conv_id),
                conv_id,
                status,
            )

        def on_delete_conv(conv_id: str):
            """Delete the current conversation."""
            if conv_id is not None:
                chat_service.delete_conversation(conv_id)
            return (
                None,
                [],
                _render_conv_list(chat_service, None),
            )

        # Wire up events
        new_chat_btn.click(
            fn=on_new_chat,
            outputs=[current_conv_id, chatbot, conv_list, msg_input],
        )

        # TODO: Conversation selection via HTML click events requires
        # a custom JavaScript component or Gradio's gr.render decorator.
        # Currently, conversation switching works via the Gradio select event.
        # For a production UI, consider:
        #   1. Using gr.render with a radio button list
        #   2. Adding custom JS for clickable conversation items
        #   3. Using gr.Dropdown as a simpler alternative

        send_btn.click(
            fn=on_send_message,
            inputs=[msg_input, current_conv_id, chatbot],
            outputs=[chatbot, msg_input, conv_list, current_conv_id, status_text],
        )

        msg_input.submit(
            fn=on_send_message,
            inputs=[msg_input, current_conv_id, chatbot],
            outputs=[chatbot, msg_input, conv_list, current_conv_id, status_text],
        )

    return app


# ── Helpers ─────────────────────────────────────────────────────

def _conv_to_chatbot(conv) -> list:
    """Convert Conversation messages to Gradio chatbot format."""
    messages = []
    for msg in conv.messages:
        messages.append({"role": msg.role, "content": msg.content})
    return messages


def _render_conv_list(chat_service: ChatService, active_id: str = None) -> str:
    """Render the conversation list as clickable HTML."""
    convs = chat_service.list_conversations()
    if not convs:
        return "<p style='color: #888; padding: 8px;'>No conversations yet</p>"

    html_parts = []
    for conv in convs:
        active_class = "active" if conv.id == active_id else ""
        title = conv.title[:30] + ("..." if len(conv.title) > 30 else "")
        html_parts.append(
            f'<div class="conv-item {active_class}" '
            f'onclick="document.dispatchEvent(new CustomEvent(\'select_conv\', '
            f'{{detail: \'{conv.id}\'}}))">'
            f'{title}</div>'
        )
    return "\n".join(html_parts)
