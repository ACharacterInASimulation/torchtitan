from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module, ModuleDict, ModuleList
from torchtitan.tools.logging import logger


def normalize_vocab_token(token: str) -> str:
    """Normalize tokenizer entries into a compact, deterministic form."""
    if token.startswith("<") and token.endswith(">"):
        return token

    normalized = token
    for prefix in ("Ġ", "▁"):
        normalized = normalized.replace(prefix, " ")
    normalized = normalized.replace("Ċ", "\n").strip().lower()
    return normalized or "<empty>"


def build_compressed_token_map(*, tokenizer_path: str, vocab_size: int) -> list[int]:
    """Build a paper-inspired token compression map from tokenizer vocabulary."""
    tokenizer = HuggingFaceTokenizer.Config().build(tokenizer_path=tokenizer_path)
    vocab = tokenizer.tokenizer.get_vocab()

    tokens_by_id = [str(i) for i in range(vocab_size)]
    for token, token_id in vocab.items():
        if 0 <= token_id < vocab_size:
            tokens_by_id[token_id] = token

    canonical_to_compressed: dict[str, int] = {}
    compressed_token_map = [0] * vocab_size
    for token_id, token in enumerate(tokens_by_id):
        canonical = normalize_vocab_token(token)
        compressed_token_map[token_id] = canonical_to_compressed.setdefault(
            canonical, len(canonical_to_compressed)
        )

    logger.info(
        "Built Engram token compression map: vocab_size=%s compressed_vocab_size=%s",
        vocab_size,
        len(canonical_to_compressed),
    )
    return compressed_token_map


class DepthwiseCausalConv1d(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        channels: int
        kernel_size: int = 3
        bias: bool = True

    def __init__(self, config: Config):
        super().__init__()
        self.channels = config.channels
        self.kernel_size = config.kernel_size
        self.weight = nn.Parameter(
            torch.empty(config.channels, 1, config.kernel_size)
        )
        self.bias = (
            nn.Parameter(torch.empty(config.channels)) if config.bias else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = F.conv1d(x, self.weight, self.bias, groups=self.channels)
        return x.transpose(1, 2)


class Engram(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        vocab_size: int
        engram_dim: int
        ngram_orders: list[int]
        table_sizes: list[int]
        num_hash_heads: int = 8
        kernel_size: int = 3
        enable_tokenizer_compression: bool = True
        compressed_token_map: list[int] | None = None
        embedding_param_init: dict | None = None
        value_proj: Linear.Config
        gate_proj: Linear.Config
        out_proj: Linear.Config
        conv: DepthwiseCausalConv1d.Config

        def to_dict(self) -> dict:
            config_dict = Module.Config.to_dict(self)
            if self.compressed_token_map is not None:
                config_dict["compressed_token_map"] = (
                    f"<elided token map with {len(self.compressed_token_map)} entries>"
                )
            return config_dict

    def __init__(self, config: Config):
        super().__init__()
        assert len(config.ngram_orders) == len(config.table_sizes), (
            "ngram_orders and table_sizes must have the same length"
        )
        assert config.engram_dim % config.num_hash_heads == 0, (
            "engram_dim must be divisible by num_hash_heads"
        )

        self.dim = config.dim
        self.vocab_size = config.vocab_size
        self.engram_dim = config.engram_dim
        self.ngram_orders = tuple(config.ngram_orders)
        self.table_sizes = tuple(config.table_sizes)
        self.num_hash_heads = config.num_hash_heads
        self.head_dim = config.engram_dim // config.num_hash_heads

        token_map = config.compressed_token_map
        if token_map is None:
            token_map = list(range(config.vocab_size))
        self._compressed_token_map_values = token_map
        self.register_buffer(
            "compressed_token_map",
            torch.empty(0, dtype=torch.long),
            persistent=False,
        )

        self.tables = ModuleDict()
        self._hash_coeffs_values: dict[int, list[list[int]]] = {}
        self._hash_offsets_values: dict[int, list[int]] = {}
        for order, table_size in zip(self.ngram_orders, self.table_sizes, strict=True):
            head_tables = ModuleList()
            for _ in range(self.num_hash_heads):
                head_tables.append(
                    Embedding.Config(
                        num_embeddings=table_size,
                        embedding_dim=self.head_dim,
                        param_init=config.embedding_param_init,
                    ).build()
                )
            self.tables[str(order)] = head_tables

            coeffs = []
            for head in range(self.num_hash_heads):
                coeffs.append(
                    [
                        (1315423911 + 2654435761 * (head + 1) * (idx + 1))
                        % 2147483647
                        for idx in range(order)
                    ]
                )
            offsets = [
                (104729 * (head + 1) + 1009 * order) % max(1, table_size)
                for head in range(self.num_hash_heads)
            ]
            self._hash_coeffs_values[order] = coeffs
            self._hash_offsets_values[order] = offsets
            self.register_buffer(
                f"hash_coeffs_{order}",
                torch.empty(0, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                f"hash_offsets_{order}",
                torch.empty(0, dtype=torch.long),
                persistent=False,
            )

        self.value_proj = config.value_proj.build()
        self.gate_proj = config.gate_proj.build()
        self.out_proj = config.out_proj.build()
        self.conv = config.conv.build()

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        assert buffer_device is None or buffer_device.type != "meta", (
            f"buffer_device must not be meta, got {buffer_device}. "
            "Engram buffers should be materialized on a real device."
        )
        if buffer_device is None:
            buffer_device = self.value_proj.weight.device

        self.compressed_token_map = torch.tensor(
            self._compressed_token_map_values,
            dtype=torch.long,
            device=buffer_device,
        )
        for order in self.ngram_orders:
            setattr(
                self,
                f"hash_coeffs_{order}",
                torch.tensor(
                    self._hash_coeffs_values[order],
                    dtype=torch.long,
                    device=buffer_device,
                ),
            )
            setattr(
                self,
                f"hash_offsets_{order}",
                torch.tensor(
                    self._hash_offsets_values[order],
                    dtype=torch.long,
                    device=buffer_device,
                ),
            )

    def _compress_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.compressed_token_map[input_ids]

    def _hash_ngrams(
        self,
        compressed_input_ids: torch.Tensor,
        *,
        order: int,
        table_size: int,
    ) -> torch.Tensor:
        batch_size, seq_len = compressed_input_ids.shape
        padded = F.pad(compressed_input_ids, (order - 1, 0), value=0)
        gram_slices = [padded[:, idx : idx + seq_len] for idx in range(order)]
        grams = torch.stack(gram_slices, dim=-1).long()

        coeffs = getattr(self, f"hash_coeffs_{order}")
        offsets = getattr(self, f"hash_offsets_{order}")
        hashed = (grams.unsqueeze(-2) * coeffs.view(1, 1, self.num_hash_heads, order)).sum(
            dim=-1
        )
        hashed = (hashed + offsets.view(1, 1, self.num_hash_heads)) % table_size
        return hashed.view(batch_size, seq_len, self.num_hash_heads)

    def _lookup_order(
        self, order: int, order_hashes: torch.Tensor, table_size: int
    ) -> torch.Tensor:
        del table_size  # already encoded in the embedding modules
        embeddings = []
        head_tables = self.tables[str(order)]
        for head_idx, table in enumerate(head_tables):
            embeddings.append(table(order_hashes[..., head_idx]))
        return torch.cat(embeddings, dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if input_ids is None:
            return torch.zeros_like(hidden_states)

        compressed_input_ids = self._compress_tokens(input_ids.long())

        order_embeddings = []
        for order, table_size in zip(self.ngram_orders, self.table_sizes, strict=True):
            order_hashes = self._hash_ngrams(
                compressed_input_ids, order=order, table_size=table_size
            )
            order_embeddings.append(
                self._lookup_order(order, order_hashes, table_size)
            )

        retrieved_memory = torch.cat(order_embeddings, dim=-1)
        retrieved_memory = self.value_proj(retrieved_memory)
        gate = torch.sigmoid(self.gate_proj(hidden_states))
        fused = retrieved_memory * gate
        fused = fused + self.conv(fused)
        return self.out_proj(fused)
