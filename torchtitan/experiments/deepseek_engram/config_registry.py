from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import ActivationCheckpointConfig, CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader

from . import model_registry
from .trainer import DeepSeekEngramTrainer


def deepseek_engram_debugmodel() -> DeepSeekEngramTrainer.Config:
    return DeepSeekEngramTrainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("debugmodel"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=30,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
            expert_tensor_parallel_degree=1,
            enable_sequence_parallel=False,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
    )


def deepseek_engram_debugmodel_flex_attn() -> DeepSeekEngramTrainer.Config:
    config = deepseek_engram_debugmodel()
    config.model_spec = model_registry("debugmodel_flex_attn")
    return config


def deepseek_engram_perf_debug() -> DeepSeekEngramTrainer.Config:
    config = deepseek_engram_debugmodel_flex_attn()
    config.activation_checkpoint = ActivationCheckpointConfig(mode="none")
    config.compile = CompileConfig(enable=True, components=["loss"])
    return config


def deepseek_engram_hsdp_debug() -> DeepSeekEngramTrainer.Config:
    config = deepseek_engram_debugmodel()
    config.activation_checkpoint = ActivationCheckpointConfig(mode="none")
    config.compile = CompileConfig(enable=True, components=["loss"])
    return config


def deepseek_engram_3b() -> DeepSeekEngramTrainer.Config:
    return DeepSeekEngramTrainer.Config(
        hf_assets_path="./assets/hf/deepseek-moe-16b-base",
        model_spec=model_registry("3B"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
        optimizer=OptimizersContainer.Config(lr=2.2e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=4096,
            steps=1000,
        ),
        parallelism=ParallelismConfig(
            pipeline_parallel_schedule="Interleaved1F1B",
            expert_parallel_degree=1,
            expert_tensor_parallel_degree=1,
            enable_sequence_parallel=False,
        ),
        checkpoint=CheckpointManager.Config(interval=10),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
        compile=CompileConfig(enable=True, components=["loss"]),
    )


def deepseek_engram_1b() -> DeepSeekEngramTrainer.Config:
    return DeepSeekEngramTrainer.Config(
        hf_assets_path="./assets/hf/deepseek-moe-16b-base",
        model_spec=model_registry("1B"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
        optimizer=OptimizersContainer.Config(lr=2.8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=2,
            global_batch_size=16,
            seq_len=4096,
            steps=1000,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
            expert_tensor_parallel_degree=1,
            enable_sequence_parallel=False,
        ),
        checkpoint=CheckpointManager.Config(interval=100),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
        compile=CompileConfig(enable=True, components=["loss"]),
    )


def deepseek_engram_1b_single_gpu() -> DeepSeekEngramTrainer.Config:
    config = deepseek_engram_1b()
    config.training = TrainingConfig(
        local_batch_size=4,
        global_batch_size=32,
        seq_len=2048,
        steps=10,

        dtype="bfloat16",
    )
    return config


def deepseek_engram_3b_single_gpu() -> DeepSeekEngramTrainer.Config:
    config = deepseek_engram_3b()
    config.training = TrainingConfig(
        local_batch_size=1,
        global_batch_size=1,
        seq_len=2048,
        steps=10,
        dtype="bfloat16",
    )
    return config

def deepseek_engram_3b_fsdp_4_gpu() -> DeepSeekEngramTrainer.Config:
    config = deepseek_engram_3b()
    config.training = TrainingConfig(
        local_batch_size=1,
        global_batch_size=1,
        seq_len=2048,
        steps=10,

        dtype="bfloat16",
    )
    return config

