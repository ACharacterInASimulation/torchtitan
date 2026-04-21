import json
from dataclasses import dataclass

import torch
from torch import nn

from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.deepseek_v3.model import (
    Attention,
    DeepSeekV3Model,
    DeepSeekV3TransformerBlock,
)
from torchtitan.tools.logging import logger

from .engram import build_compressed_token_map, Engram


class DeepSeekEngramTransformerBlock(DeepSeekV3TransformerBlock):
    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV3TransformerBlock.Config):
        engram: Engram.Config | None = None

    def __init__(self, config: Config):
        super().__init__(config)
        self.engram = config.engram.build() if config.engram is not None else None

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
        engram_input_ids: torch.Tensor | None = None,
    ):
        if self.engram is not None:
            engram_input = self.attention_norm(x)
            x = x + self.engram(engram_input, engram_input_ids)

        attn_input = self.attention_norm(x)
        x = x + self.attention(attn_input, freqs_cis, attention_masks, positions)
        if self.moe_enabled:
            x = x + self.moe(self.ffn_norm(x))
        else:
            x = x + self.feed_forward(self.ffn_norm(x))
        return x


class DeepSeekEngramModel(DeepSeekV3Model):
    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV3Model.Config):
        def to_dict(self) -> dict:
            config_dict = DeepSeekV3Model.Config.to_dict(self)

            unique_layers: dict[str, dict] = {}
            ordered_keys: list[str] = []
            for layer_id, layer_cfg in enumerate(self.layers):
                layer_dict = layer_cfg.to_dict()
                layer_key = json.dumps(layer_dict, sort_keys=True, ensure_ascii=False)
                if layer_key not in unique_layers:
                    unique_layers[layer_key] = {
                        "layer_ids": [],
                        "layer_config": layer_dict,
                    }
                    ordered_keys.append(layer_key)
                unique_layers[layer_key]["layer_ids"].append(layer_id)

            config_dict["num_layers"] = len(self.layers)
            config_dict["layers"] = [unique_layers[key] for key in ordered_keys]
            return config_dict

        def update_from_config(
            self,
            *,
            trainer_config,
            **kwargs,
        ) -> None:
            DeepSeekV3Model.Config.update_from_config(
                self, trainer_config=trainer_config, **kwargs
            )

            compressed_token_map = None
            for layer_cfg in self.layers:
                if (
                    isinstance(layer_cfg, DeepSeekEngramTransformerBlock.Config)
                    and layer_cfg.engram is not None
                ):
                    layer_cfg.engram.vocab_size = self.vocab_size
                    if layer_cfg.engram.enable_tokenizer_compression:
                        if compressed_token_map is None:
                            try:
                                compressed_token_map = build_compressed_token_map(
                                    tokenizer_path=trainer_config.hf_assets_path,
                                    vocab_size=self.vocab_size,
                                )
                            except Exception as exc:
                                logger.warning(
                                    "Falling back to identity Engram token map because tokenizer compression failed: %s",
                                    exc,
                                )
                                compressed_token_map = list(range(self.vocab_size))
                        layer_cfg.engram.compressed_token_map = compressed_token_map

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            assert isinstance(self.layers[0].attention, Attention.Config)

            nparams_embedding = 0
            nparams_dense = 0
            nparams_moe_router = 0
            nparams_shared_experts = 0
            nparams_experts = 0
            nparams_engram_sparse = 0

            for name, param in model.named_parameters():
                if "tok_embeddings" in name:
                    nparams_embedding += param.numel()
                    nparams_dense += param.numel()
                elif "output" in name:
                    nparams_dense += param.numel()
                elif ".engram.tables." in name:
                    nparams_engram_sparse += param.numel()
                elif ".engram." in name:
                    nparams_dense += param.numel()
                elif "moe.shared_experts" in name:
                    nparams_shared_experts += param.numel()
                elif "moe.router" in name:
                    nparams_moe_router += param.numel()
                elif "moe.experts" in name:
                    nparams_experts += param.numel()
                else:
                    nparams_dense += param.numel()

            engram_configs = [
                layer_cfg.engram
                for layer_cfg in self.layers
                if isinstance(layer_cfg, DeepSeekEngramTransformerBlock.Config)
                and layer_cfg.engram is not None
            ]
            moe_config = next((l.moe for l in self.layers if l.moe is not None), None)
            moe_active = 0
            if moe_config is not None:
                moe_active = (
                    nparams_moe_router
                    + nparams_shared_experts
                    + nparams_experts * moe_config.router.top_k // moe_config.num_experts
                )
            engram_active = sum(
                len(engram_cfg.ngram_orders) * engram_cfg.engram_dim
                for engram_cfg in engram_configs
            )

            nparams_sparse = (
                nparams_moe_router
                + nparams_shared_experts
                + nparams_experts
                + nparams_engram_sparse
            )
            nparams_total = nparams_dense + nparams_sparse
            nparams_active = nparams_dense + moe_active + engram_active

            logger.info(
                "Total parameter count: dense %s, sparse_moe %s, sparse_engram %s, active_moe %s, active_engram %s, active_total %s",
                f"{nparams_dense:,}",
                f'{nparams_moe_router + nparams_shared_experts + nparams_experts:,}',
                f"{nparams_engram_sparse:,}",
                f"{moe_active:,}",
                f"{engram_active:,}",
                f"{nparams_active:,}",
            )

            # Engram hashing and table lookups are index/integer operations rather
            # than dense floating-point GEMMs, so they are excluded from the FLOP
            # estimate. The projection/conv weights already live in nparams_dense.
            engram_non_param_flops_per_token = sum(
                3 * engram_cfg.dim for engram_cfg in engram_configs
            )
            num_flops_per_token = (
                6 * (nparams_dense - nparams_embedding + moe_active)
                + 6
                * len(self.layers)
                * self.layers[0].attention.n_heads
                * (
                    self.layers[0].attention.qk_nope_head_dim
                    + self.layers[0].attention.qk_rope_head_dim
                    + self.layers[0].attention.v_head_dim
                )
                * seq_len
                + engram_non_param_flops_per_token
            )
            return nparams_total, num_flops_per_token

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
        engram_input_ids: torch.Tensor | None = None,
    ):
        hidden_states = (
            self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        )
        if engram_input_ids is None and not torch.is_floating_point(tokens):
            engram_input_ids = tokens

        for layer in self.layers.values():
            if isinstance(layer, DeepSeekEngramTransformerBlock):
                hidden_states = layer(
                    hidden_states,
                    self.freqs_cis,
                    attention_masks,
                    positions,
                    engram_input_ids=engram_input_ids,
                )
            else:
                hidden_states = layer(
                    hidden_states, self.freqs_cis, attention_masks, positions
                )

        hidden_states = (
            self.norm(hidden_states) if self.norm is not None else hidden_states
        )
        return self.output(hidden_states) if self.output is not None else hidden_states
