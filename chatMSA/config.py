"""
Unified configuration for chatMSA.

All settings in one place. Converts to MSA's native config types internally.
"""

import argparse
from dataclasses import dataclass, field
from typing import List, Optional

import sys
import pathlib

# Add project root to path so we can import MSA's config types
_project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

from src.config.memory_config import GenerateConfig, ModelConfig, MemoryConfig
from src.utils.template import QWEN3_TEMPLATE, QWEN3_INSTRUCT_TEMPLATE


@dataclass
class ChatConfig:
    """Single source of truth for all chatMSA settings."""

    # ── MSA Engine ──────────────────────────────────────────────
    model_path: str = "ckpt/MSA-4B"
    template: str = "QWEN3_INSTRUCT_TEMPLATE"
    devices: Optional[List[int]] = None  # None = auto-detect all GPUs
    max_generate_tokens: int = 512
    max_batch_size: int = 4
    top_p: float = 0.9
    temperature: float = 0.0
    block_size: int = 2048
    max_chunk_per_block: int = 16384
    max_seq_len: int = 0       # 0 = unlimited
    max_query_seq_len: int = 0  # 0 = unlimited

    # ── Chat ────────────────────────────────────────────────────
    max_history_turns: int = 20       # Max conversation turns kept in prompt context
    max_context_tokens: int = 4096    # Token budget for history + current query
    db_path: str = "data/chat_msa.db"

    # ── Server ──────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 7860

    @property
    def template_dict(self) -> dict:
        """Resolve template name to the actual template dict."""
        templates = {
            "QWEN3_TEMPLATE": QWEN3_TEMPLATE,
            "QWEN3_INSTRUCT_TEMPLATE": QWEN3_INSTRUCT_TEMPLATE,
        }
        if self.template not in templates:
            raise ValueError(f"Unknown template: {self.template}. Choose from {list(templates.keys())}")
        return templates[self.template]

    def to_generate_config(self) -> GenerateConfig:
        """Convert to MSA's GenerateConfig."""
        import torch
        devices = self.devices if self.devices is not None else list(range(torch.cuda.device_count()))
        return GenerateConfig(
            devices=devices,
            template=self.template_dict,
            max_generate_tokens=self.max_generate_tokens,
            max_seq_len=self.max_seq_len,
            max_query_seq_len=self.max_query_seq_len,
            max_batch_size=self.max_batch_size,
            top_p=self.top_p,
            temperature=self.temperature,
            qa_mode=True,
        )

    def to_model_config(self) -> ModelConfig:
        """Convert to MSA's ModelConfig."""
        return ModelConfig(
            model_path=self.model_path,
            doc_top_k=16,           # TODO: make configurable
            pooling_kernel_size=64, # TODO: make configurable
            router_layer_idx="all",
        )

    def to_memory_config(self, memory_file_path: str = "") -> MemoryConfig:
        """Convert to MSA's MemoryConfig."""
        return MemoryConfig(
            block_size=self.block_size,
            pooling_kernel_size=64,  # TODO: make configurable
            slice_chunk_size=self.max_chunk_per_block,
            memory_file_path=memory_file_path,
        )

    @classmethod
    def from_args(cls, args: Optional[List[str]] = None) -> "ChatConfig":
        """Parse CLI arguments into a ChatConfig."""
        parser = argparse.ArgumentParser(description="chatMSA — Multi-turn conversation on MSA")
        parser.add_argument("--model_path", type=str, default="ckpt/MSA-4B")
        parser.add_argument("--template", type=str, default="QWEN3_INSTRUCT_TEMPLATE",
                            choices=["QWEN3_TEMPLATE", "QWEN3_INSTRUCT_TEMPLATE"])
        parser.add_argument("--devices", type=str, default=None,
                            help="Comma-separated GPU IDs, e.g. '0,1,2,3'. Default: all")
        parser.add_argument("--max_generate_tokens", type=int, default=512)
        parser.add_argument("--max_batch_size", type=int, default=4)
        parser.add_argument("--top_p", type=float, default=0.9)
        parser.add_argument("--temperature", type=float, default=0.0)
        parser.add_argument("--block_size", type=int, default=2048)
        parser.add_argument("--max_chunk_per_block", type=int, default=16384)
        parser.add_argument("--max_history_turns", type=int, default=20)
        parser.add_argument("--max_context_tokens", type=int, default=4096)
        parser.add_argument("--db_path", type=str, default="data/chat_msa.db")
        parser.add_argument("--host", type=str, default="0.0.0.0")
        parser.add_argument("--port", type=int, default=7860)

        parsed = parser.parse_args(args)

        devices = None
        if parsed.devices is not None:
            devices = [int(d) for d in parsed.devices.split(",")]

        return cls(
            model_path=parsed.model_path,
            template=parsed.template,
            devices=devices,
            max_generate_tokens=parsed.max_generate_tokens,
            max_batch_size=parsed.max_batch_size,
            top_p=parsed.top_p,
            temperature=parsed.temperature,
            block_size=parsed.block_size,
            max_chunk_per_block=parsed.max_chunk_per_block,
            max_history_turns=parsed.max_history_turns,
            max_context_tokens=parsed.max_context_tokens,
            db_path=parsed.db_path,
            host=parsed.host,
            port=parsed.port,
        )
