# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from functools import partial

import torch.nn as nn

import torchtitan.models.deepseek_v3 as dsv3
from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common.attention import FlexAttention
from torchtitan.models.common import Embedding, Linear, RMSNorm, RoPE
from torchtitan.protocols.model_spec import ModelSpec

from .engram import DepthwiseCausalConv1d, Engram
from .model import DeepSeekEngramModel, DeepSeekEngramTransformerBlock
from .parallelize import parallelize_deepseek_engram

__all__ = ["DeepSeekEngramModel", "deepseek_engram_configs"]

_CONV_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}


def _make_engram_config(
    *,
    dim: int,
    vocab_size: int,
    engram_dim: int,
    layer_id: int,
    ngram_orders: list[int],
    table_sizes: list[int],
    num_hash_heads: int,
    kernel_size: int,
    enable_tokenizer_compression: bool = True,
) -> Engram.Config:
    retrieved_dim = len(ngram_orders) * engram_dim
    return Engram.Config(
        dim=dim,
        vocab_size=vocab_size,
        engram_dim=engram_dim,
        ngram_orders=list(ngram_orders),
        table_sizes=list(table_sizes),
        num_hash_heads=num_hash_heads,
        kernel_size=kernel_size,
        enable_tokenizer_compression=enable_tokenizer_compression,
        embedding_param_init=dsv3._EMBEDDING_INIT,
        value_proj=Linear.Config(
            in_features=retrieved_dim,
            out_features=dim,
            param_init=dsv3._LINEAR_INIT,
        ),
        gate_proj=Linear.Config(
            in_features=dim,
            out_features=dim,
            param_init=dsv3._LINEAR_INIT,
        ),
        out_proj=Linear.Config(
            in_features=dim,
            out_features=dim,
            param_init=dsv3._depth_init(layer_id),
        ),
        conv=DepthwiseCausalConv1d.Config(
            channels=dim,
            kernel_size=kernel_size,
            param_init=_CONV_INIT,
        ),
    )


def _convert_layers_with_engram(
    *,
    base_layers,
    dim: int,
    vocab_size: int,
    engram_layer_ids: list[int],
    engram_dim: int,
    ngram_orders: list[int],
    table_sizes: list[int],
    num_hash_heads: int,
    kernel_size: int,
    enable_tokenizer_compression: bool = True,
):
    layers = []
    engram_layer_set = set(engram_layer_ids)
    for layer_id, layer_cfg in enumerate(base_layers):
        engram_cfg = None
        if layer_id in engram_layer_set:
            engram_cfg = _make_engram_config(
                dim=dim,
                vocab_size=vocab_size,
                engram_dim=engram_dim,
                layer_id=layer_id,
                ngram_orders=ngram_orders,
                table_sizes=table_sizes,
                num_hash_heads=num_hash_heads,
                kernel_size=kernel_size,
                enable_tokenizer_compression=enable_tokenizer_compression,
            )

        layers.append(
            DeepSeekEngramTransformerBlock.Config(
                attention=layer_cfg.attention,
                attention_norm=layer_cfg.attention_norm,
                ffn_norm=layer_cfg.ffn_norm,
                feed_forward=layer_cfg.feed_forward,
                moe=layer_cfg.moe,
                engram=engram_cfg,
            )
        )
    return layers


def _debugmodel() -> DeepSeekEngramModel.Config:
    dim = 256
    n_layers = 6
    vocab_size = 2048
    rope_dim = 64

    base_layers = dsv3._build_dsv3_layers(
        n_layers=n_layers,
        n_dense_layers=1,
        dim=dim,
        n_heads=16,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=0.70,
        dense_hidden_dim=1024,
        moe_hidden_dim=256,
        num_experts=8,
        num_shared_experts=2,
        router_top_k=3,
        router_score_func="softmax",
        score_before_experts=False,
    )
    layers = _convert_layers_with_engram(
        base_layers=base_layers,
        dim=dim,
        vocab_size=vocab_size,
        engram_layer_ids=[1, 4],
        engram_dim=128,
        ngram_orders=[2, 3],
        table_sizes=[2048, 4096],
        num_hash_heads=4,
        kernel_size=3,
    )

    return DeepSeekEngramModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=dsv3._EMBEDDING_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=dsv3._NORM_INIT),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=dsv3._output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=rope_dim,
            max_seq_len=4096 * 4,
            theta=10000.0,
            backend="complex",
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
        layers=layers,
    )


def _debugmodel_flex_attn() -> DeepSeekEngramModel.Config:
    dim = 256
    n_layers = 6
    vocab_size = 2048
    rope_dim = 64

    base_layers = dsv3._build_dsv3_layers(
        n_layers=n_layers,
        n_dense_layers=1,
        dim=dim,
        n_heads=16,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=0.70,
        dense_hidden_dim=1024,
        moe_hidden_dim=256,
        num_experts=8,
        num_shared_experts=2,
        router_top_k=3,
        router_score_func="softmax",
        score_before_experts=False,
        inner_attention=FlexAttention.Config(),
        mask_type="block_causal",
    )
    layers = _convert_layers_with_engram(
        base_layers=base_layers,
        dim=dim,
        vocab_size=vocab_size,
        engram_layer_ids=[1, 4],
        engram_dim=128,
        ngram_orders=[2, 3],
        table_sizes=[2048, 4096],
        num_hash_heads=4,
        kernel_size=3,
    )

    return DeepSeekEngramModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=dsv3._EMBEDDING_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=dsv3._NORM_INIT),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=dsv3._output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=rope_dim,
            max_seq_len=4096 * 4,
            theta=10000.0,
            backend="complex",
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
        layers=layers,
    )


def _build_1b_family(*, n_layers: int) -> DeepSeekEngramModel.Config:
    dim = 1024
    vocab_size = 102400
    rope_dim = 64

    base_layers = dsv3._build_dsv3_layers(
        n_layers=n_layers,
        n_dense_layers=1,
        dim=dim,
        n_heads=16,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=0.70,
        dense_hidden_dim=4096,
        moe_hidden_dim=512,
        num_experts=32,
        num_shared_experts=2,
        router_top_k=4,
        router_score_func="softmax",
        score_before_experts=False,
    )
    engram_layer_ids = [2, 6] if n_layers <= 12 else [2, 7]
    layers = _convert_layers_with_engram(
        base_layers=base_layers,
        dim=dim,
        vocab_size=vocab_size,
        engram_layer_ids=engram_layer_ids,
        engram_dim=256,
        ngram_orders=[2, 3],
        table_sizes=[32768, 131072],
        num_hash_heads=8,
        kernel_size=3,
    )

    return DeepSeekEngramModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=dsv3._EMBEDDING_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=dsv3._NORM_INIT),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=dsv3._output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=rope_dim,
            max_seq_len=4096 * 4,
            theta=10000.0,
            backend="complex",
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
        layers=layers,
    )


def _1b() -> DeepSeekEngramModel.Config:
    return _build_1b_family(n_layers=12)


def _1b_plus1() -> DeepSeekEngramModel.Config:
    return _build_1b_family(n_layers=13)


def _1b_plus2() -> DeepSeekEngramModel.Config:
    return _build_1b_family(n_layers=14)


def _3b() -> DeepSeekEngramModel.Config:
    dim = 1536
    n_layers = 12
    vocab_size = 102400
    rope_dim = 64

    base_layers = dsv3._build_dsv3_layers(
        n_layers=n_layers,
        n_dense_layers=1,
        dim=dim,
        n_heads=24,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=0.70,
        dense_hidden_dim=6144,
        moe_hidden_dim=768,
        num_experts=64,
        num_shared_experts=2,
        router_top_k=4,
        router_score_func="softmax",
        score_before_experts=False,
    )
    layers = _convert_layers_with_engram(
        base_layers=base_layers,
        dim=dim,
        vocab_size=vocab_size,
        engram_layer_ids=[2, 6],
        engram_dim=512,
        ngram_orders=[2, 3],
        table_sizes=[131072, 524288],
        num_hash_heads=8,
        kernel_size=3,
    )

    return DeepSeekEngramModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=dsv3._EMBEDDING_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=dsv3._NORM_INIT),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=dsv3._output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=rope_dim,
            max_seq_len=4096 * 4,
            theta=10000.0,
            backend="complex",
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
        layers=layers,
    )


deepseek_engram_configs = {
    "debugmodel": _debugmodel,
    "debugmodel_flex_attn": _debugmodel_flex_attn,
    "1B": _1b,
    "1B_plus1": _1b_plus1,
    "1B_plus2": _1b_plus2,
    "3B": _3b,
}


def model_registry(flavor: str) -> ModelSpec:
    config = deepseek_engram_configs[flavor]()
    return ModelSpec(
        name="deepseek_engram",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_deepseek_engram,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=None,
    )
