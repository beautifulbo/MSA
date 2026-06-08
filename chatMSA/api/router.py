"""
API router aggregation.

Combines all route modules into a single FastAPI application.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from chatMSA.api.chat_routes import create_chat_routes
from chatMSA.services.chat_service import ChatService
from chatMSA.services.msa_engine_service import MSAEngineService


def create_fastapi_app(chat_service: ChatService, engine: MSAEngineService) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        chat_service: The chat business logic service.
        engine: The MSA engine service (for health checks).

    Returns:
        Configured FastAPI app with all routes mounted.
    """
    app = FastAPI(
        title="chatMSA API",
        description="Multi-turn conversation system built on MSA (Memory Sparse Attention)",
        version="0.1.0",
    )

    # CORS — allow Gradio frontend and local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API routes
    chat_router = create_chat_routes(chat_service, engine)
    app.include_router(chat_router)

    return app
