"""
chatMSA entry point.

Launches FastAPI (REST API) with embedded Gradio (chat UI).

Usage:
    python -m chatMSA.app --model_path ckpt/MSA-4B --port 7860
    python -m chatMSA.app --help
"""

import signal
import sys
import pathlib

# Ensure project root is importable
_project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

import gradio as gr
import uvicorn

from chatMSA.config import ChatConfig
from chatMSA.services.chat_service import ChatService
from chatMSA.services.msa_engine_service import MSAEngineService
from chatMSA.storage.sqlite_store import SQLiteConversationStore
from chatMSA.api.router import create_fastapi_app
from chatMSA.frontend.chat_ui import create_gradio_app


def main():
    """Main entry point."""
    # 1. Parse config
    config = ChatConfig.from_args()
    print(f"[chatMSA] Config: model={config.model_path}, db={config.db_path}")
    print(f"[chatMSA] Server: {config.host}:{config.port}")

    # 2. Initialize storage
    store = SQLiteConversationStore(config.db_path)
    store.initialize()
    print(f"[chatMSA] Storage initialized: {config.db_path}")

    # 3. Initialize engine (lazy — heavy loading happens in engine.start())
    engine = MSAEngineService(config)

    # 4. Initialize chat service
    chat_service = ChatService(engine, store, config)

    # 5. Create FastAPI app
    fastapi_app = create_fastapi_app(chat_service, engine)

    # 6. Create and mount Gradio app
    gradio_app = create_gradio_app(chat_service)
    fastapi_app = gr.mount_gradio_app(fastapi_app, gradio_app, path="/")

    # 7. Graceful shutdown
    def shutdown_handler(sig, frame):
        print("\n[chatMSA] Shutting down...")
        engine.stop()
        store.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # 8. Start engine (heavy: model loading + memory prefill)
    #    This runs before the HTTP server so the first request doesn't wait.
    #    TODO: Move engine.start() to a background thread and serve a "loading"
    #    page from Gradio while the engine initializes. This would improve UX
    #    for large models with long startup times.
    print("[chatMSA] Starting MSA engine (this may take a while)...")
    try:
        engine.start()
        print("[chatMSA] Engine ready!")
    except Exception as e:
        print(f"[chatMSA] Engine failed to start: {e}")
        print("[chatMSA] Continuing in degraded mode (chat will return errors)")

    # 9. Launch server
    print(f"[chatMSA] UI: http://{config.host}:{config.port}")
    print(f"[chatMSA] API: http://{config.host}:{config.port}/docs")
    uvicorn.run(fastapi_app, host=config.host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()
