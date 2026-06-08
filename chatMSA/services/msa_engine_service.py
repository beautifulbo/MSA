"""
MSAEngine lifecycle wrapper.

Manages the heavy MSA engine (model loading, multi-GPU workers, prefill)
with lazy initialization and clean shutdown.
"""

import time
import threading
from typing import Optional, Tuple, Dict

import sys
import pathlib

_project_root = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(_project_root))

from chatMSA.config import ChatConfig
from src.msa_service import MSAEngine


class MSAEngineService:
    """
    Wraps MSAEngine with lifecycle management.

    Usage:
        engine_service = MSAEngineService(config)
        engine_service.start()   # Heavy: loads model, prefill memory
        text, topk = engine_service.generate("What is MSA?")
        engine_service.stop()    # Join workers

    Context manager:
        with MSAEngineService(config) as engine:
            text, topk = engine.generate("What is MSA?")
    """

    def __init__(self, config: ChatConfig):
        self.config = config
        self._engine: Optional[MSAEngine] = None
        self._ready = False
        self._loading = False
        self._error: Optional[str] = None
        self._start_time: float = 0
        self._lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def is_loading(self) -> bool:
        return self._loading

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def uptime(self) -> float:
        if self._start_time == 0:
            return 0.0
        return time.time() - self._start_time

    def start(self, memory_file_path: str = "") -> None:
        """
        Initialize the MSA engine. This is expensive (model loading + prefill).

        Args:
            memory_file_path: Path to the memory corpus file (pickle/json).
                              Empty string = no memory corpus (pure chat mode).
        """
        with self._lock:
            if self._ready:
                return
            if self._loading:
                raise RuntimeError("Engine is already loading")

            self._loading = True
            self._error = None

        try:
            generate_config = self.config.to_generate_config()
            model_config = self.config.to_model_config()
            memory_config = self.config.to_memory_config(memory_file_path)

            self._engine = MSAEngine(generate_config, model_config, memory_config)
            self._ready = True
            self._start_time = time.time()
        except Exception as e:
            self._error = str(e)
            raise
        finally:
            self._loading = False

    def stop(self) -> None:
        """Shut down the engine and join worker processes."""
        with self._lock:
            if self._engine is not None:
                self._engine.stop_workers()
                self._engine = None
            self._ready = False
            self._start_time = 0

    def generate(self, prompt: str) -> Tuple[str, Dict]:
        """
        Generate a response for a single prompt.

        Args:
            prompt: The input text (already includes history if applicable).

        Returns:
            Tuple of (generated_text, recall_topk).
            recall_topk maps layer_idx -> list of retrieved doc IDs.

        Raises:
            RuntimeError: If engine is not ready.
        """
        if not self._ready or self._engine is None:
            raise RuntimeError("Engine is not ready. Call start() first.")

        texts, recall_topk, _ = self._engine.generate(
            prompt,
            require_recall_topk=True,
        )
        # texts is a list; single prompt → single result
        response_text = texts[0] if texts else ""
        return response_text, recall_topk

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
