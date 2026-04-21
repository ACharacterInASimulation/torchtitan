# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, cast

import torch

from torchtitan.distributed.context_parallel import cp_shard
from torchtitan.models.common.decoder import Decoder
from torchtitan.trainer import Trainer


class DeepSeekEngramTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        pass

    def post_dataloading_process(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        inputs = input_dict["input"]
        extra_inputs = {k: v for k, v in input_dict.items() if k != "input"}
        extra_kwargs: dict[str, Any] = {"engram_input_ids": inputs}

        if isinstance(self.model_config, Decoder.Config):
            layer = self.model_config.layers[0]
            attn_config = layer.attention
        else:
            attn_config = None
        mask_type = getattr(attn_config, "mask_type", "causal")

        positions = extra_inputs.pop("positions", None)
        if mask_type == "block_causal":
            extra_kwargs["positions"] = positions
        elif self.parallel_dims.cp_enabled:
            extra_kwargs["positions"] = torch.arange(
                0, inputs.shape[1], dtype=torch.int32, device=self.device
            ).expand(inputs.shape)

        inner_attention = getattr(attn_config, "inner_attention", None)
        if inner_attention is not None:
            from torchtitan.models.common.attention import (
                FlexAttention,
                VarlenAttention,
            )

            if isinstance(
                inner_attention, (FlexAttention.Config, VarlenAttention.Config)
            ):
                assert (
                    self.tokenizer is not None
                ), "tokenizer is required for flex/varlen attention"
                model = cast(Decoder, self.model_parts[0])
                extra_kwargs["attention_masks"] = model.get_attention_masks(
                    input_batch=inputs,
                    tokenizer=self.tokenizer,
                    extra_inputs=extra_inputs,
                )

        if self.parallel_dims.cp_enabled:
            attention_masks = extra_kwargs.get("attention_masks")
            positions = extra_kwargs["positions"]
            engram_input_ids = extra_kwargs["engram_input_ids"]
            (inputs, labels, positions, engram_input_ids), attention_masks = cp_shard(
                self.parallel_dims.get_mesh("cp"),
                (inputs, labels, positions, engram_input_ids),
                attention_masks,
                self.config.parallelism.context_parallel_load_balancer,
            )
            extra_kwargs["positions"] = positions
            extra_kwargs["engram_input_ids"] = engram_input_ids
            if attention_masks is not None:
                extra_kwargs["attention_masks"] = attention_masks

        return inputs, labels, extra_inputs, extra_kwargs
