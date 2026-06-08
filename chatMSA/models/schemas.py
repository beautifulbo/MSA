"""
Pydantic models for API request/response serialization and validation.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ── Request Models ──────────────────────────────────────────────

class ConversationCreateRequest(BaseModel):
    title: Optional[str] = Field(None, description="Conversation title. Auto-generated if omitted.")
    metadata: Optional[Dict] = Field(None, description="Optional metadata (model info, settings, etc.)")


class ConversationRenameRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200, description="New conversation title")


class MessageSendRequest(BaseModel):
    content: str = Field(..., min_length=1, description="User message content")


# ── Response Models ─────────────────────────────────────────────

class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    timestamp: float
    recall_topk: Optional[Dict] = None


class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: float
    updated_at: float
    message_count: int
    turn_count: int
    last_message_preview: Optional[str] = None


class ConversationDetail(BaseModel):
    id: str
    title: str
    created_at: float
    updated_at: float
    messages: List[MessageResponse]
    metadata: Optional[Dict] = None


class HealthResponse(BaseModel):
    status: str            # "ready" | "loading" | "error"
    model_path: str
    gpu_count: int
    uptime_seconds: float
    error: Optional[str] = None
