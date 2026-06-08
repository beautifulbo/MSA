#!/usr/bin/env python3
"""
simple_msa_demo.py — MSA 完整推理流程示例

功能：
  1. 读取指定目录下的 .txt 文档
  2. Stage 1: 用 MSA 模型编码文档为 chunk-pooled KV cache
  3. Stage 2: 用 router_q_proj/router_k_proj 做完整路由检索
  4. Stage 3: 拼接 template prefix + 选中文档 KV + query，调用 model.generate()

用法：
  python scripts/simple_msa_demo.py \
      --model_path ckpt/MSA-4B \
      --doc_dir /path/to/your/documents \
      --query "你的问题"

与项目完整流程的对应关系：
  本脚本 = PrefillStage1Worker._inference()   (Stage 1)
         + Memory.prefill_stage2()             (Stage 2 路由打分)
         + MSAService.generate()               (Stage 3 生成)
  去掉了多 GPU 通信 (NCCL all-gather/all-to-all)，仅用单 GPU 运行。
"""

import argparse
import os
import sys
import pathlib
import glob
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer

project_path = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_path))

from src.msa.model import MSAForCausalLM, MSAConfig
from src.msa.memory_sparse_attention import MemorySparseAttention
from src.utils.cache import CustomDynamicCache, create_cache
from src.utils.template import QWEN3_INSTRUCT_TEMPLATE


# ============================================================
# 1. 读取文档
# ============================================================

def _read_text_file(path: str) -> str:
    """尝试多种编码读取文本文件。"""
    for encoding in ["utf-8", "gbk", "gb2312", "gb18030", "latin-1"]:
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise UnicodeDecodeError(f"无法识别文件编码: {path}")


def load_documents(doc_dir: str) -> List[str]:
    """读取目录下所有 .txt 文件，返回文档内容列表。"""
    docs = []
    paths = sorted(glob.glob(os.path.join(doc_dir, "*.txt")))
    if not paths:
        raise FileNotFoundError(f"在 {doc_dir} 下未找到 .txt 文件")
    for path in paths:
        content = _read_text_file(path).strip()
        if content:
            docs.append(content)
    print(f"[Stage 0] 加载了 {len(docs)} 篇文档，来自 {doc_dir}")
    return docs


# ============================================================
# 2. Stage 1: 编码文档为 KV cache（完整实现）
# ============================================================

def encode_documents(
    model: MSAForCausalLM,
    tokenizer,
    documents: List[str],
    pooling_kernel_size: int,
    device: str,
) -> Dict:
    """
    Stage 1 完整实现：对所有文档做 forward，得到 chunk-pooled KV cache。

    对应原始代码：
      PrefillStage1Worker._inference() → MemorySparseAttention.forward(stage="prefill_stage1")

    返回一个 dict 包含所有推理产物：
      - kv_caches:              {layer_idx: (K, V)} 每层的 chunk-pooled KV
      - router_k_caches:        {layer_idx: rk} 每层的 router key (decouple_router=True 时)
      - template_prefix_kvcache: {layer_idx: (k, v)} 模板前缀 KV
      - pooled_doc_ids:         {layer_idx: tensor} 每个 chunk 对应的文档 ID
      - chunk_sizes:            每篇文档的 chunk 数量
    """
    msa_config = model.config.msa_config
    router_layer_idx = msa_config.router_layer_idx
    if router_layer_idx == "all":
        router_layers = list(range(model.config.num_hidden_layers))
    else:
        router_layers = [int(i) for i in router_layer_idx.split(",")]

    # ── 拼接所有文档为一个序列 ──
    # 对应 PrefillStage1Worker._prepare_block_inputs()
    all_input_ids = []
    all_attention_mask = []
    all_doc_ids = []
    all_position_ids = []
    chunk_sizes = []

    for doc_idx, doc_text in enumerate(documents):
        doc_inputs = tokenizer(doc_text, add_special_tokens=False)
        doc_token_ids = doc_inputs["input_ids"]
        length = len(doc_token_ids)

        all_input_ids.extend(doc_token_ids)
        all_attention_mask.extend([1] * length)
        # doc_id 从 1 开始（0 留给 query 区域）
        all_doc_ids.extend([doc_idx + 1] * length)
        all_position_ids.extend(list(range(length)))

        n_chunks = (length + pooling_kernel_size - 1) // pooling_kernel_size
        chunk_sizes.append(n_chunks)

    input_ids = torch.LongTensor([all_input_ids]).to(device)
    attention_mask = torch.LongTensor([all_attention_mask]).to(device)
    doc_ids = torch.LongTensor([all_doc_ids]).to(device)
    position_ids = torch.LongTensor([all_position_ids]).to(device)

    # ── 创建 Cache ──
    past_key_values = CustomDynamicCache()
    n_layers = model.config.num_hidden_layers
    for layer_idx in range(n_layers):
        past_key_values.record_kwargs(layer_idx, {"stage": "prefill_stage1"})

    print(f"[Stage 1] 正在编码文档... (共 {sum(chunk_sizes)} chunks, {len(documents)} 篇文档)")

    with torch.no_grad():
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            doc_ids=doc_ids,
        )

    # ── 提取产物 ──
    past_kv = outputs.past_key_values
    result = {
        "kv_caches": {},
        "router_k_caches": {},
        "template_prefix_kvcache": {},
        "pooled_doc_ids": {},
        "chunk_sizes": chunk_sizes,
    }

    for layer_idx in range(n_layers):
        k, v = past_kv.get_kvcache(layer_idx)
        result["kv_caches"][layer_idx] = (k, v)

        # router key (decouple_router=True 时独立存在)
        rk = past_kv.get_router_kcache(layer_idx)
        if rk is not None:
            result["router_k_caches"][layer_idx] = rk

        # 模板前缀 KV
        kwargs = past_kv.cache_kwargs.get(layer_idx, {})
        if "template_prefix_kcache" in kwargs:
            result["template_prefix_kvcache"][layer_idx] = (
                kwargs["template_prefix_kcache"].to(device),
                kwargs["template_prefix_vcache"].to(device),
            )

        # pooled doc ids（每个 chunk 属于哪篇文档）
        if "pooled_doc_ids" in kwargs:
            result["pooled_doc_ids"][layer_idx] = kwargs["pooled_doc_ids"]

    print(f"[Stage 1] 编码完成，KV cache 形状: {result['kv_caches'][router_layers[0]][0].shape}")
    return result


# ============================================================
# 3. Stage 2: 完整路由检索
# ============================================================

def retrieve_documents(
    model: MSAForCausalLM,
    tokenizer,
    query: str,
    stage1_result: Dict,
    top_k: int,
    device: str,
) -> Dict:
    """
    Stage 2 完整实现：用 router_q_proj/router_k_proj 做路由检索。

    对应原始代码：
      MemorySparseAttention.forward(stage="prefill_stage2")
      → Memory.prefill_stage2()  (路由打分)
      → Memory.doc_query()       (选中文档 KV 提取)

    完整流程：
      1. 对 query 做 forward，经过 router layer 时：
         a. router_q_proj 投影 query → routing_q
         b. routing_q 与 router_k (或 pooled K) 做 cosine similarity
         c. head_reduce → query_reduce → chunk_reduce 得到文档分数
         d. top-k 选出文档
      2. 提取选中文档的 K/V

    返回：
      - selected_k:    [1, n_kv_heads, selected_chunks, head_dim]
      - selected_v:    [1, n_kv_heads, selected_chunks, head_dim]
      - scores:        [1, top_k]
      - selected_doc_ids: 选中的文档 ID 列表
      - template_prefix_kvcache: 模板前缀 KV
    """
    msa_config = model.config.msa_config
    router_layer_idx = msa_config.router_layer_idx
    if router_layer_idx == "all":
        router_layers = list(range(model.config.num_hidden_layers))
    else:
        router_layers = [int(i) for i in router_layer_idx.split(",")]

    # 用第一个 router layer 做检索
    layer_idx = router_layers[0]

    # 获取该层的 attention module
    attn_layer = model.model.layers[layer_idx].self_attn
    assert isinstance(attn_layer, MemorySparseAttention), \
        f"Layer {layer_idx} is not MemorySparseAttention, got {type(attn_layer)}"

    # ── 构建 query 输入 ──
    # 对应 MSAService._apply_template()
    template = QWEN3_INSTRUCT_TEMPLATE
    prompt_text = (
        "\nPlease answer the question based on the above historical document information\n\n"
        + query
        + "\nPlease return all documents related to the question\n"
    )

    prompt_inputs = tokenizer(prompt_text, add_special_tokens=False)
    prompt_ids = prompt_inputs["input_ids"]

    # 模板后缀
    pad_token = tokenizer.pad_token
    pad_token_id = tokenizer.pad_token_id
    template_str = template["prompt"].replace("{prompt}", pad_token)
    template_inputs = tokenizer(template_str, add_special_tokens=False)
    pad_index = template_inputs["input_ids"].index(pad_token_id)
    tail_ids = template_inputs["input_ids"][pad_index + 1:]

    # response head
    response_head = "<|im_start|>"
    response_head_ids = tokenizer(response_head, add_special_tokens=False)["input_ids"]

    # 拼接完整序列
    full_ids = prompt_ids + tail_ids + response_head_ids
    input_ids_tensor = torch.LongTensor([full_ids]).to(device)
    seq_len = input_ids_tensor.shape[1]

    attention_mask = torch.ones(1, seq_len, dtype=torch.long, device=device)

    # doc_ids: query 区域 = 0, template 后缀 = -2, response head = -1
    doc_ids = (
        [0] * len(prompt_ids)
        + [-2] * len(tail_ids)
        + [-1] * len(response_head_ids)
    )
    doc_ids_tensor = torch.LongTensor([doc_ids]).to(device)

    # position ids（Global RoPE：从 top_k + 3 开始）
    top_k_actual = min(top_k, stage1_result["kv_caches"][layer_idx][0].shape[2])
    start_pos = 3 + top_k_actual
    position_ids = list(range(start_pos, start_pos + seq_len))
    position_ids_tensor = torch.LongTensor([position_ids]).to(device)

    # ── 创建 Cache 并注入 Stage 1 产物 ──
    past_key_values = CustomDynamicCache()
    n_layers = model.config.num_hidden_layers
    for li in range(n_layers):
        kwargs = {"stage": "prefill_stage2"}
        # 注入 template prefix KV
        if li in stage1_result["template_prefix_kvcache"]:
            t_k, t_v = stage1_result["template_prefix_kvcache"][li]
            kwargs["template_prefix_kcache"] = t_k
            kwargs["template_prefix_vcache"] = t_v
        past_key_values.record_kwargs(li, kwargs)

    # 将 Stage 1 的 KV cache 注入到 past_key_values
    for li in range(n_layers):
        if li in stage1_result["kv_caches"]:
            k, v = stage1_result["kv_caches"][li]
            past_key_values.update(k, v, li)
        if li in stage1_result["router_k_caches"]:
            rk = stage1_result["router_k_caches"][li]
            past_key_values.update_router_kcache(rk, li)
        # 注入 pooled_doc_ids
        if li in stage1_result["pooled_doc_ids"]:
            past_key_values.cache_kwargs[li]["pooled_doc_ids"] = stage1_result["pooled_doc_ids"][li]

    # ── 执行 forward（会触发 MemorySparseAttention 的 prefill_stage2 分支）──
    print(f"[Stage 2] 正在路由检索... (query length: {seq_len}, top_k: {top_k_actual})")

    # 设置 memory_client 让 MemorySparseAttention 能访问 BlockData
    # 对应 MSAService.setup_memory_client()
    _setup_memory_client_for_demo(
        model=model,
        kv_caches=stage1_result["kv_caches"],
        router_k_caches=stage1_result["router_k_caches"],
        template_prefix_kvcache=stage1_result["template_prefix_kvcache"],
        pooled_doc_ids=stage1_result["pooled_doc_ids"],
        chunk_sizes=stage1_result["chunk_sizes"],
        device=device,
    )

    with torch.no_grad():
        outputs = model.model(
            input_ids=input_ids_tensor,
            attention_mask=attention_mask,
            position_ids=position_ids_tensor,
            past_key_values=past_key_values,
            use_cache=True,
            doc_ids=doc_ids_tensor,
        )

    # ── 提取路由结果 ──
    # 从 past_key_values 的 cache_kwargs 中提取 recall_topk
    # 对应 MemorySparseAttention.forward(stage="prefill_stage2") 中的 recall_topk 逻辑
    recall_topk = past_key_values.cache_kwargs[layer_idx].get("recall_topk", None)

    # 提取 compacked KV cache（已组装好的 template + selected docs + query）
    compacked_k = past_key_values.cache_kwargs[layer_idx].get("compacked_key_cache", None)
    compacked_v = past_key_values.cache_kwargs[layer_idx].get("compacked_value_cache", None)

    result = {
        "past_key_values": past_key_values,
        "compacked_k": compacked_k,
        "compacked_v": compacked_v,
        "recall_topk": recall_topk,
        "input_ids": input_ids_tensor,
        "attention_mask": attention_mask,
        "doc_ids": doc_ids_tensor,
        "position_ids": position_ids_tensor,
    }

    if recall_topk:
        for item in recall_topk:
            doc_ids_list = item.get("topk_doc_ids", [])
            scores_list = item.get("score", [])
            print(f"[Stage 2] 选出 {len(doc_ids_list)} 个文档 chunks, 最高分: {max(scores_list):.4f}")
    else:
        print("[Stage 2] 路由完成（无 recall_topk 输出）")

    return result


def _setup_memory_client_for_demo(
    model: MSAForCausalLM,
    kv_caches: Dict,
    router_k_caches: Dict,
    template_prefix_kvcache: Dict,
    pooled_doc_ids: Dict,
    chunk_sizes: List[int],
    device: str,
):
    """
    为 MemorySparseAttention 设置 memory_client，使其能访问文档 KV cache。

    对应 MSAService.setup_memory_client()。
    在完整流程中，MemorySparseAttention 通过 self.memory_client.doc_query()
    访问 BlockData 中的 K/V/rk。这里我们构造一个最小实现。
    """
    # 计算文档元数据
    nr_docs = len(chunk_sizes)
    doc_lens = [0] + chunk_sizes  # 第 0 位是 padding
    doc_offsets = [0]
    for cl in chunk_sizes:
        doc_offsets.append(doc_offsets[-1] + cl)
    doc_ids_list = list(range(nr_docs + 1))  # [0, 1, 2, ..., nr_docs]

    doc_lens_cpu = torch.LongTensor(doc_lens)
    doc_offsets_cpu = torch.LongTensor(doc_offsets)
    doc_ids_tensor = torch.LongTensor(doc_ids_list).to(device)

    # 构造 k_slices 和 slice_desc（对应 Memory._build_k_slices）
    n_layers = len(kv_caches)
    k_slices = {}
    slice_desc_list = []

    for layer_idx in range(n_layers):
        if layer_idx not in kv_caches:
            continue
        k, v = kv_caches[layer_idx]
        # k shape: [1, n_kv_heads, n_chunks, head_dim]
        n_chunks = k.shape[2]
        # 构造 k_slice_t: [1, n_kv_heads, 1, head_dim, n_chunks]
        k_slice_t = k.permute(0, 1, 3, 2).unsqueeze(2)
        k_slices[layer_idx] = [k_slice_t]

        # 简化的 slice_desc
        class SimpleSliceDesc:
            def __init__(self, nr_chunks, nr_docs, doc_ids):
                self.nr_chunks = nr_chunks
                self.nr_docs = nr_docs
                self.local_doc_ids_0 = torch.arange(nr_docs + 1, device=device).unsqueeze(0)
                self.original_doc_ids = doc_ids.unsqueeze(0)
                self.global_doc_ids = doc_ids.unsqueeze(0)
        slice_desc_list.append(SimpleSliceDesc(n_chunks, nr_docs, doc_ids_tensor))

    class DemoMemoryClient:
        """最小 memory_client 实现，让 MemorySparseAttention 能访问文档 KV。"""
        def __init__(self):
            self.blocks = {}
            self.block_desc = type('BlockDesc', (), {
                'nr_docs': nr_docs,
                'doc_lens_cpu': doc_lens_cpu,
                'doc_offsets_cpu': doc_offsets_cpu,
                'doc_ids': doc_ids_tensor,
                'doc_ids_cpu': torch.LongTensor(doc_ids_list),
            })()
            self.k_slices = k_slices
            self.slice_desc = slice_desc_list
            self.model_config = type('MC', (), {'doc_top_k': 16})()
            self.template_prefix_kvcache = template_prefix_kvcache
            self.pooled_doc_ids = pooled_doc_ids
            self.device = device
            self.world_size = 1
            self.num_key_value_groups = (
                model.config.num_attention_heads // model.config.num_key_value_heads
            )
            self.head_reduce_method = msa_config.head_reduce_method
            self.query_reduce_method = msa_config.query_reduce_method
            self.chunk_reduce_method = msa_config.chunk_reduce_method
            self.decouple_router = msa_config.decouple_router
            self.scaling = -1.0 if "INFONCE" in msa_config.aux_loss_method and self.decouple_router else 1.0

            # 初始化 blocks
            for li in range(n_layers):
                from src.msa_service import BlockData
                block_data = BlockData()
                if li in kv_caches:
                    k, v = kv_caches[li]
                    block_data.k = k
                    block_data.v = v
                if li in router_k_caches:
                    block_data.rk = router_k_caches[li]
                self.blocks[li] = block_data

            # 构建 k_slices 和 slice_desc（对应 Memory._build_k_slices）
            self._build_k_slices()

        def _build_k_slices(self):
            """构建分片 K slices 用于路由打分。"""
            msa_config_local = model.config.msa_config
            router_layer_idx_str = msa_config_local.router_layer_idx
            if router_layer_idx_str == "all":
                router_layers_local = list(range(model.config.num_hidden_layers))
            else:
                router_layers_local = [int(i) for i in router_layer_idx_str.split(",")]

            self.k_slices = {}
            self.slice_desc = []
            slice_chunk_size = 16 * 1024  # 默认值

            for li in router_layers_local:
                if li not in self.blocks:
                    continue
                block = self.blocks[li]
                router_k = block.get_router_k()  # rk if exists, else k
                if router_k is None:
                    continue

                n_kv_heads = router_k.shape[1]
                n_chunks = router_k.shape[2]

                # 按 slice_chunk_size 分片
                chunks_done = 0
                k_slices_for_layer = []
                slice_descs = []
                while chunks_done < n_chunks:
                    this_chunk = min(slice_chunk_size, n_chunks - chunks_done)
                    k_slice = router_k[:, :, chunks_done:chunks_done + this_chunk, :]
                    # k_slice_t: [1, n_kv_heads, 1, head_dim, this_chunk]
                    k_slice_t = k_slice.permute(0, 1, 3, 2).unsqueeze(2)
                    k_slices_for_layer.append(k_slice_t)

                    # 构建 slice 描述
                    doc_ids_for_slice = self.pooled_doc_ids.get(li)
                    if doc_ids_for_slice is not None:
                        slice_doc_ids = doc_ids_for_slice[chunks_done:chunks_done + this_chunk]
                    else:
                        slice_doc_ids = torch.arange(1, this_chunk + 1, device=self.device)

                    class SliceDesc:
                        pass
                    sd = SliceDesc()
                    sd.nr_chunks = this_chunk
                    sd.nr_docs = self.block_desc.nr_docs

                    # local_doc_ids_0: [1, nr_chunks] — 每个 chunk 对应的 local doc id
                    sd.local_doc_ids_0 = slice_doc_ids.unsqueeze(0)
                    # original_doc_ids: [1, nr_docs] — local → global 映射
                    sd.original_doc_ids = self.block_desc.doc_ids.unsqueeze(0)
                    slice_descs.append(sd)

                    chunks_done += this_chunk

                self.k_slices[li] = k_slices_for_layer
                if not self.slice_desc:
                    self.slice_desc = slice_descs

        def get_template_prefix_kvcaches(self, layer_idx):
            return self.template_prefix_kvcache[layer_idx]

        def doc_query(self, query_states, query_mask, layer_idx):
            """
            单 GPU 版 doc_query。
            对应 MSAService.doc_query()，去掉了 NCCL 通信。
            """
            bsz, nhead, seqlen, hdim = query_states.shape
            num_kv_groups = self.num_key_value_groups
            top_k = self.model_config.doc_top_k
            dtype = query_states.dtype

            # ── Phase 1: 本地打分 ──
            # 对应 Memory.prefill_stage2()
            total_docs = self.block_desc.nr_docs + 1
            min_val = torch.finfo(dtype).min
            global_doc_scores = torch.full((bsz, total_docs), min_val, dtype=dtype, device=self.device)

            # reshape query for matmul
            query_reshaped = query_states.view(bsz, self.num_key_value_groups, -1, seqlen, hdim) * self.scaling

            for slice_desc, k_slice_t in zip(self.slice_desc, self.k_slices[layer_idx]):
                # attn_scores: [bsz, n_kv_heads, n_groups, seqlen, chunk]
                attn_scores = torch.matmul(query_reshaped, k_slice_t)

                # mask invalid query positions
                routing_mask = ~query_mask
                attn_scores.masked_fill_(routing_mask, min_val)

                # head_reduce
                if self.head_reduce_method == "max":
                    scores = attn_scores.flatten(1, 2).max(dim=1).values  # [bsz, seqlen, chunk]
                elif self.head_reduce_method == "mean":
                    scores = attn_scores.flatten(1, 2).mean(dim=1)
                else:
                    scores = attn_scores.flatten(1, 2).max(dim=1).values

                # query_reduce
                if self.query_reduce_method == "max":
                    scores = scores.max(dim=1).values  # [bsz, chunk]
                elif self.query_reduce_method == "mean":
                    valid_mask = query_mask.squeeze(2).squeeze(1)  # [bsz, seqlen]
                    scores_clean = torch.where(valid_mask.unsqueeze(-1), scores, torch.zeros_like(scores))
                    counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
                    scores = scores_clean.sum(dim=1) / counts
                else:
                    scores = scores.max(dim=1).values

                # chunk → doc scatter
                scatter_indices = slice_desc.local_doc_ids_0.expand(bsz, -1)
                local_doc_scores = torch.full((bsz, slice_desc.nr_docs + 1), min_val, device=self.device, dtype=dtype)

                if self.chunk_reduce_method == "max":
                    local_doc_scores.scatter_reduce_(dim=1, index=scatter_indices, src=scores, reduce="amax", include_self=True)
                elif self.chunk_reduce_method == "mean":
                    doc_sums = torch.zeros_like(local_doc_scores)
                    doc_sums.scatter_reduce_(dim=1, index=scatter_indices, src=scores, reduce="sum", include_self=False)
                    doc_counts = torch.zeros_like(local_doc_scores)
                    ones = torch.ones_like(scores)
                    doc_counts.scatter_reduce_(dim=1, index=scatter_indices, src=ones, reduce="sum", include_self=False)
                    doc_counts = doc_counts.clamp(min=1.0)
                    mean_scores = doc_sums / doc_counts
                    local_doc_scores = torch.where(doc_counts > 0, mean_scores, local_doc_scores)
                else:
                    local_doc_scores.scatter_reduce_(dim=1, index=scatter_indices, src=scores, reduce="amax", include_self=True)

                # local → global
                global_indices = slice_desc.original_doc_ids.expand(bsz, -1)
                global_doc_scores.scatter_reduce_(dim=1, index=global_indices, src=local_doc_scores, reduce="amax", include_self=True)

            # ── Phase 2: Top-K 选择 ──
            k = min(top_k, global_doc_scores.shape[1])
            final_scores, batch_selected_local_ids = torch.topk(global_doc_scores, k=k, dim=1)
            batch_selected_global_ids = self.block_desc.doc_ids[batch_selected_local_ids]

            # ── Phase 3: 提取选中文档的 K/V ──
            block = self.blocks[layer_idx]
            doc_k = block.k  # [1, n_kv_heads, n_chunks, head_dim]
            doc_v = block.v

            # 收集选中 chunks 的 K/V
            # 需要从 pooled_doc_ids 中找到被选中文档对应的所有 chunks
            pooled_ids = self.pooled_doc_ids.get(layer_idx)
            if pooled_ids is not None:
                # 找到属于选中文档的所有 chunk 索引
                selected_doc_set = set(batch_selected_global_ids[0].cpu().tolist())
                chunk_mask = torch.tensor(
                    [pooled_ids[i].item() in selected_doc_set for i in range(pooled_ids.shape[0])],
                    device=self.device,
                )
                selected_chunk_indices = chunk_mask.nonzero(as_tuple=False).squeeze(-1)

                final_k = doc_k[:, :, selected_chunk_indices, :]
                final_v = doc_v[:, :, selected_chunk_indices, :]
                final_selected_doc_ids = batch_selected_global_ids
            else:
                final_k = doc_k
                final_v = doc_v
                final_selected_doc_ids = batch_selected_global_ids

            num_selected = torch.tensor([final_k.shape[2]], device=self.device, dtype=torch.long)

            return final_k, final_v, final_scores, num_selected, final_selected_doc_ids

    # 注入 memory_client 到所有 router layers
    msa_config = model.config.msa_config
    client = DemoMemoryClient()
    for layer in model.model.modules():
        if isinstance(layer, MemorySparseAttention):
            layer.set_memory_client(client)


# ============================================================
# 4. Stage 3: 用 model.generate() 生成回答
# ============================================================

def generate_answer(
    model: MSAForCausalLM,
    tokenizer,
    stage2_result: Dict,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 0.9,
) -> str:
    """
    Stage 3: 调用 model.generate() 生成回答。

    Stage 2 的 MemorySparseAttention 已经将 template prefix KV + 选中文档 KV
    + query KV 组装成了 compacked_key_cache / compacked_value_cache，
    存在 past_key_values 的 cache_kwargs 中。

    Stage 3 的 generate 过程中，MemorySparseAttention 的 "generate" 分支
    会直接使用这个 compacked cache 做 attention。
    """
    past_key_values = stage2_result["past_key_values"]
    input_ids = stage2_result["input_ids"]
    attention_mask = stage2_result["attention_mask"]
    doc_ids = stage2_result["doc_ids"]
    position_ids = stage2_result["position_ids"]

    # 将 stage 切换为 "generate"
    n_layers = model.config.num_hidden_layers
    for layer_idx in range(n_layers):
        past_key_values.record_kwargs(layer_idx, {"stage": "generate"})

    # 重新注入 compacked cache 和其他 kwargs
    # （record_kwargs 会覆盖旧值，需要保留已有的 compacked cache）
    for layer_idx in range(n_layers):
        if layer_idx in stage2_result["past_key_values"].cache_kwargs:
            old_kwargs = stage2_result["past_key_values"].cache_kwargs[layer_idx]
            if "compacked_key_cache" in old_kwargs:
                past_key_values.cache_kwargs[layer_idx]["compacked_key_cache"] = old_kwargs["compacked_key_cache"]
                past_key_values.cache_kwargs[layer_idx]["compacked_value_cache"] = old_kwargs["compacked_value_cache"]
                past_key_values.cache_kwargs[layer_idx]["kv_lengths"] = old_kwargs["kv_lengths"]
                past_key_values.cache_kwargs[layer_idx]["attention_mask"] = old_kwargs["attention_mask"]

    # 设置 generate 所需的 meta
    past_key_values.meta["require_recall_topk"] = False
    past_key_values.meta["qa_mode"] = True
    past_key_values.meta["max_generate_tokens"] = max_new_tokens
    past_key_values.meta["tokenizer"] = tokenizer
    past_key_values.meta["idx_to_doc"] = {}  # 单文档模式无需 doc reference
    past_key_values.meta["pattern"] = r"\[(\d+)\]"
    past_key_values.meta["response_string"] = [""]

    print(f"[Stage 3] 正在生成回答... (max_new_tokens: {max_new_tokens})")

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            doc_ids=doc_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    response = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    return response


# ============================================================
# 5. 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="MSA 完整推理流程演示")
    parser.add_argument("--model_path", type=str, default="ckpt/MSA-4B", help="模型路径")
    parser.add_argument("--doc_dir", type=str, required=True, help="文档目录（包含 .txt 文件）")
    parser.add_argument("--query", type=str, required=True, help="用户问题")
    parser.add_argument("--pooling_kernel_size", type=int, default=64, help="chunk pooling 窗口大小")
    parser.add_argument("--top_k", type=int, default=16, help="检索的 top-k 文档数")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.0, help="采样温度 (0=greedy)")
    parser.add_argument("--top_p", type=float, default=0.9, help="核采样 top-p")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    args = parser.parse_args()

    device = args.device
    if not torch.cuda.is_available():
        device = "cpu"
        print("[WARNING] CUDA 不可用，使用 CPU（会很慢）")

    # ── 加载模型 ──
    print(f"[Init] 加载模型: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = MSAForCausalLM.from_pretrained(
        args.model_path,
        use_cache=True,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
        torch_dtype= torch.bfloat16,
        device_map=device,
    )
    model.eval()
    print("[Init] 模型加载完成")

    # ── 读取文档 ──
    documents = load_documents(args.doc_dir)

    # ── Stage 1: 编码文档 ──
    stage1_result = encode_documents(
        model=model,
        tokenizer=tokenizer,
        documents=documents,
        pooling_kernel_size=args.pooling_kernel_size,
        device=device,
    )

    # ── Stage 2: 路由检索 ──
    stage2_result = retrieve_documents(
        model=model,
        tokenizer=tokenizer,
        query=args.query,
        stage1_result=stage1_result,
        top_k=args.top_k,
        device=device,
    )

    # ── Stage 3: 生成回答 ──
    response = generate_answer(
        model=model,
        tokenizer=tokenizer,
        stage2_result=stage2_result,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # ── 输出 ──
    print("\n" + "=" * 60)
    print(f"问题: {args.query}")
    print("=" * 60)
    print(f"回答:\n{response}")
    print("=" * 60)


if __name__ == "__main__":
    main()
