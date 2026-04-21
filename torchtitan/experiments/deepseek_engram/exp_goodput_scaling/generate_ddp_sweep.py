from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from typing import Literal
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_LOG_ROOT = REPO_ROOT / "experiment_logs" / "deepseek_engram_1b_goodput_scaling"


@dataclass(frozen=True)
class RunSpec:
    world_size: int
    local_batch_size: int
    grad_accum_steps: int
    seq_len: int
    steps: int
    dtype: str
    optimizer_lr: float | None = None

    @property
    def global_batch_size(self) -> int:
        return self.world_size * self.local_batch_size * self.grad_accum_steps

    @property
    def run_name(self) -> str:
        return f"ddp_{self.world_size}gpu_gbs{self.global_batch_size}"


def parse_visible_devices(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    devices = [device.strip() for device in raw.split(",") if device.strip()]
    if not devices:
        raise ValueError("--visible-devices was provided but no device ids were found")
    return devices


def get_scaled_lr(
    *,
    base_lr: float,
    world_size: int,
    scaling_mode: Literal["none", "sqrt", "linear"],
) -> float | None:
    if scaling_mode == "none":
        return None
    if scaling_mode == "sqrt":
        return base_lr * math.sqrt(world_size)
    if scaling_mode == "linear":
        return base_lr * world_size
    raise ValueError(f"Unsupported lr scaling mode: {scaling_mode}")


def build_command(
    *,
    spec: RunSpec,
    module_name: str,
    config_name: str,
    python_bin: str,
    log_path: Path,
    visible_devices: list[str] | None,
    activation_checkpoint_mode: str | None,
    warmup_steps: int | None,
) -> str:
    command_parts = []

    if visible_devices is not None:
        if len(visible_devices) < spec.world_size:
            raise ValueError(
                f"Need at least {spec.world_size} visible devices, got {len(visible_devices)}"
            )
        command_parts.append(
            "CUDA_VISIBLE_DEVICES=" + ",".join(visible_devices[: spec.world_size])
        )

    command_parts.extend(
        [
            python_bin,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={spec.world_size}",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=localhost:0",
            "-m",
            "torchtitan.train",
            "--module",
            module_name,
            "--config",
            config_name,
            "--training.local_batch_size",
            str(spec.local_batch_size),
            "--training.global_batch_size",
            str(spec.global_batch_size),
            "--training.seq_len",
            str(spec.seq_len),
            "--training.steps",
            str(spec.steps),
            "--training.dtype",
            spec.dtype,
            "--metrics.log_freq",
            "1",
            "--parallelism.data_parallel_replicate_degree",
            str(spec.world_size),
            "--parallelism.data_parallel_shard_degree",
            "1",
            "--parallelism.pipeline_parallel_degree",
            "1",
            "--parallelism.tensor_parallel_degree",
            "1",
            "--parallelism.context_parallel_degree",
            "1",
            "--parallelism.expert_parallel_degree",
            "1",
            "--parallelism.expert_tensor_parallel_degree",
            "1",
        ]
    )

    if activation_checkpoint_mode is not None:
        command_parts.extend(
            ["--activation_checkpoint.mode", activation_checkpoint_mode]
        )
    if spec.optimizer_lr is not None:
        command_parts.extend(["--optimizer.lr", f"{spec.optimizer_lr:.8g}"])
    if warmup_steps is not None:
        command_parts.extend(["--lr_scheduler.warmup_steps", str(warmup_steps)])

    return " \\\n  ".join(command_parts) + f" \\\n  2>&1 | tee {log_path}"


def build_run_specs(
    *,
    min_gpus: int,
    max_gpus: int,
    local_batch_size: int,
    grad_accum_steps: int,
    seq_len: int,
    steps: int,
    dtype: str,
    base_lr: float,
    lr_scaling: Literal["none", "sqrt", "linear"],
) -> list[RunSpec]:
    return [
        RunSpec(
            world_size=world_size,
            local_batch_size=local_batch_size,
            grad_accum_steps=grad_accum_steps,
            seq_len=seq_len,
            steps=steps,
            dtype=dtype,
            optimizer_lr=get_scaled_lr(
                base_lr=base_lr,
                world_size=world_size,
                scaling_mode=lr_scaling,
            ),
        )
        for world_size in range(min_gpus, max_gpus + 1)
    ]


def write_plan_csv(plan_path: Path, specs: list[RunSpec], log_root: Path) -> None:
    with plan_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "run_name",
                "world_size",
                "local_batch_size",
                "grad_accum_steps",
                "global_batch_size",
                "seq_len",
                "steps",
                "dtype",
                "optimizer_lr",
                "log_path",
            ],
        )
        writer.writeheader()
        for spec in specs:
            writer.writerow(
                {
                    "run_name": spec.run_name,
                    "world_size": spec.world_size,
                    "local_batch_size": spec.local_batch_size,
                    "grad_accum_steps": spec.grad_accum_steps,
                    "global_batch_size": spec.global_batch_size,
                    "seq_len": spec.seq_len,
                    "steps": spec.steps,
                    "dtype": spec.dtype,
                    "optimizer_lr": spec.optimizer_lr,
                    "log_path": log_root / f"{spec.run_name}.log",
                }
            )


def write_shell_script(
    *,
    script_path: Path,
    specs: list[RunSpec],
    log_root: Path,
    repo_root: Path,
    module_name: str,
    config_name: str,
    python_bin: str,
    visible_devices: list[str] | None,
    activation_checkpoint_mode: str | None,
    lr_scaling: Literal["none", "sqrt", "linear"],
    warmup_steps: int | None,
) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"cd {repo_root}",
        f"mkdir -p {log_root}",
        "",
        "# Fixed study settings",
        f"# config={config_name}",
        f"# seq_len={specs[0].seq_len}",
        f"# local_batch_size={specs[0].local_batch_size}",
        f"# grad_accum_steps={specs[0].grad_accum_steps}",
        f"# dtype={specs[0].dtype}",
        f"# lr_scaling={lr_scaling}",
        "",
    ]

    for spec in specs:
        log_path = log_root / f"{spec.run_name}.log"
        lines.append(
            f'echo "Running {spec.run_name} (world_size={spec.world_size}, global_batch_size={spec.global_batch_size}, optimizer_lr={spec.optimizer_lr if spec.optimizer_lr is not None else "config_default"})"'
        )
        lines.append(
            build_command(
                spec=spec,
                module_name=module_name,
                config_name=config_name,
                python_bin=python_bin,
                log_path=log_path,
                visible_devices=visible_devices,
                activation_checkpoint_mode=activation_checkpoint_mode,
                warmup_steps=warmup_steps,
            )
        )
        lines.append("")

    script_path.write_text("\n".join(lines))
    script_path.chmod(0o755)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate DeepSeek Engram 1B DDP goodput scaling commands."
    )
    parser.add_argument("--min-gpus", type=int, default=1)
    parser.add_argument("--max-gpus", type=int, default=8)
    parser.add_argument("--local-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--python-bin", default="python")
    parser.add_argument("--module-name", default="deepseek_engram")
    parser.add_argument("--config-name", default="deepseek_engram_1b")
    parser.add_argument("--base-lr", type=float, default=2.8e-4)
    parser.add_argument(
        "--lr-scaling",
        choices=["none", "sqrt", "linear"],
        default="none",
    )
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--visible-devices", default=None)
    parser.add_argument("--activation-checkpoint-mode", default=None)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    args = parser.parse_args()

    if args.min_gpus < 1:
        raise ValueError("--min-gpus must be >= 1")
    if args.max_gpus < args.min_gpus:
        raise ValueError("--max-gpus must be >= --min-gpus")

    visible_devices = parse_visible_devices(args.visible_devices)

    specs = build_run_specs(
        min_gpus=args.min_gpus,
        max_gpus=args.max_gpus,
        local_batch_size=args.local_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        seq_len=args.seq_len,
        steps=args.steps,
        dtype=args.dtype,
        base_lr=args.base_lr,
        lr_scaling=args.lr_scaling,
    )

    args.log_root.mkdir(parents=True, exist_ok=True)
    plan_path = args.log_root / "ddp_goodput_plan.csv"
    script_path = args.log_root / "run_ddp_goodput_scaling.sh"

    write_plan_csv(plan_path, specs, args.log_root)
    write_shell_script(
        script_path=script_path,
        specs=specs,
        log_root=args.log_root,
        repo_root=REPO_ROOT,
        module_name=args.module_name,
        config_name=args.config_name,
        python_bin=args.python_bin,
        visible_devices=visible_devices,
        activation_checkpoint_mode=args.activation_checkpoint_mode,
        lr_scaling=args.lr_scaling,
        warmup_steps=args.warmup_steps,
    )

    print(f"Wrote plan CSV: {plan_path}")
    print(f"Wrote shell script: {script_path}")


if __name__ == "__main__":
    main()
