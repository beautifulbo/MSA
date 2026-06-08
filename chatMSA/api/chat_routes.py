"""
FastAPI routes for conversation management and chat.

Endpoints:
    POST   /api/conversations                -> Create conversation
    GET    /api/conversations                -> List all conversations
    GET    /api/conversations/{id}           -> Get conversation with messages
    DELETE /api/conversations/{id}           -> Delete conversation
    PATCH  /api/conversations/{id}           -> Rename conversation
    POST   /api/conversations/{id}/messages  -> Send message, get response
    GET    /api/health                       -> Engine status
"""

from fastapi import APIRouter, HTTPException

from chatMSA.models.schemas import (
    ConversationCreateRequest,
    ConversationDetail,
    ConversationRenameRequest,
    ConversationSummary,
    HealthResponse,
    MessageResponse,
    MessageSendRequest,
)
from chatMSA.services.chat_service import ChatService
from chatMSA.services.msa_engine_service import MSAEngineService


def create_chat_routes(chat_service: ChatService, engine: MSAEngineService) -> APIRouter:
    """Create and return the chat API router."""
    router = APIRouter(prefix="/api")

    # ── Health ──────────────────────────────────────────────────

    @router.get("/health", response_model=HealthResponse)
    def health():
        status = "ready" if engine.is_ready else ("loading" if engine.is_loading else "error")
        import torch
        gpu_count = len(engine.config.devices) if engine.config.devices else torch.cuda.device_count()
        return HealthResponse(
            status=status,
            model_path=engine.config.model_path,
            gpu_count=gpu_count,
            uptime_seconds=engine.uptime,
            error=engine.error,
        )

    # ── Conversations ───────────────────────────────────────────

    @router.post("/conversations", response_model=ConversationDetail, status_code=201)
    def create_conversation(req: ConversationCreateRequest = None):
        if req is None:
            req = ConversationCreateRequest()
        conv = chat_service.create_conversation(title=req.title)
        return _to_detail(conv)

    @router.get("/conversations", response_model=list)
    def list_conversations():
        convs = chat_service.list_conversations()
        return [_to_summary(c) for c in convs]

    @router.get("/conversations/{conv_id}", response_model=ConversationDetail)
    def get_conversation(conv_id: str):
        conv = chat_service.get_conversation(conv_id)
        if conv is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return _to_detail(conv)

    @router.delete("/conversations/{conv_id}", status_code=204)
    def delete_conversation(conv_id: str):
        deleted = chat_service.delete_conversation(conv_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Conversation not found")

    @router.patch("/conversations/{conv_id}", response_model=ConversationDetail)
    def rename_conversation(conv_id: str, req: ConversationRenameRequest):
        conv = chat_service.rename_conversation(conv_id, req.title)
        if conv is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return _to_detail(conv)

    # ── Messages ────────────────────────────────────────────────

    @router.post("/conversations/{conv_id}/messages", response_model=MessageResponse)
    def send_message(conv_id: str, req: MessageSendRequest):
        try:
            msg = chat_service.send_message(conv_id, req.content)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        return MessageResponse(
            id=msg.id,
            role=msg.role,
            content=msg.content,
            timestamp=msg.timestamp,
            recall_topk=msg.recall_topk,
        )

    return router


# ── Helpers ─────────────────────────────────────────────────────

def _to_summary(conv) -> ConversationSummary:
    msg_count = getattr(conv, "_message_count", conv.message_count)
    turn_count = getattr(conv, "_turn_count", conv.turn_count)
    last_preview = None
    if conv.messages:
        last_preview = conv.messages[-1].content[:100]
    return ConversationSummary(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        message_count=msg_count,
        turn_count=turn_count,
        last_message_preview=last_preview,
    )


def _to_detail(conv) -> ConversationDetail:
    return ConversationDetail(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        messages=[
            MessageResponse(
                id=m.id,
                role=m.role,
                content=m.content,
                timestamp=m.timestamp,
                recall_topk=m.recall_topk,
            )
            for m in conv.messages
        ],
        metadata=conv.metadata,
    )
