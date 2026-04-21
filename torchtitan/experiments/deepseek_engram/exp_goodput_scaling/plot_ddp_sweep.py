from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean


try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "matplotlib is required for plotting. Install it in the active environment."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_LOG_ROOT = REPO_ROOT / "experiment_logs" / "deepseek_engram_1b_goodput_scaling"
PLAN_FILENAME = "ddp_goodput_plan.csv"

STEP_PATTERN = re.compile(
    r"step:\s*(?P<step>\d+).*?"
    r"loss:\s*(?P<loss>[-+0-9.eE]+).*?"
    r"tps:\s*(?P<tps>[0-9,]+).*?"
    r"stat_eff:\s*(?P<stat_eff>(?:[-+0-9.eE]+|N/A)).*?"
    r"goodput:\s*(?P<goodput>(?:[-+0-9.eE,]+|N/A))"
)


@dataclass(frozen=True)
class StepRecord:
    step: int
    loss: float
    tps_per_device: float
    logged_stat_efficiency: float | None
    logged_goodput: float | None


@dataclass(frozen=True)
class RunSummary:
    run_name: str
    world_size: int
    global_batch_size: int
    log_path: Path
    num_points_used: int
    avg_loss: float
    avg_tps_per_device: float
    avg_total_tps: float
    avg_logged_stat_efficiency: float | None
    avg_total_stat_efficiency: float | None
    avg_logged_goodput: float | None
    avg_total_goodput: float | None


def parse_optional_float(raw: str) -> float | None:
    text = raw.replace(",", "")
    if text == "N/A":
        return None
    return float(text)


def parse_log(log_path: Path) -> list[StepRecord]:
    records: list[StepRecord] = []
    for line in log_path.read_text().splitlines():
        match = STEP_PATTERN.search(line)
        if match is None:
            continue
        records.append(
            StepRecord(
                step=int(match.group("step")),
                loss=float(match.group("loss")),
                tps_per_device=float(match.group("tps").replace(",", "")),
                logged_stat_efficiency=parse_optional_float(match.group("stat_eff")),
                logged_goodput=parse_optional_float(match.group("goodput")),
            )
        )
    return records


def collapse_rank_duplicates(records: list[StepRecord]) -> list[StepRecord]:
    grouped: dict[int, list[StepRecord]] = {}
    for record in records:
        grouped.setdefault(record.step, []).append(record)

    collapsed_records: list[StepRecord] = []
    for step in sorted(grouped):
        step_records = grouped[step]
        stat_values = [
            record.logged_stat_efficiency
            for record in step_records
            if record.logged_stat_efficiency is not None
        ]
        goodput_values = [
            record.logged_goodput
            for record in step_records
            if record.logged_goodput is not None
        ]
        collapsed_records.append(
            StepRecord(
                step=step,
                loss=fmean(record.loss for record in step_records),
                tps_per_device=fmean(
                    record.tps_per_device for record in step_records
                ),
                logged_stat_efficiency=fmean(stat_values) if stat_values else None,
                logged_goodput=fmean(goodput_values) if goodput_values else None,
            )
        )
    return collapsed_records


def load_plan(log_root: Path) -> list[dict[str, str]]:
    plan_path = log_root / PLAN_FILENAME
    if not plan_path.exists():
        raise FileNotFoundError(
            f"Could not find {plan_path}. Run generate_ddp_sweep.py first."
        )
    with plan_path.open() as file:
        return list(csv.DictReader(file))


def summarize_run(
    *,
    run_name: str,
    world_size: int,
    global_batch_size: int,
    log_path: Path,
    min_step: int,
) -> RunSummary | None:
    all_records = collapse_rank_duplicates(parse_log(log_path))
    filtered_records = [record for record in all_records if record.step >= min_step]
    if not filtered_records:
        return None

    total_tps_values = [
        record.tps_per_device * world_size for record in filtered_records
    ]

    logged_stat_values = [
        record.logged_stat_efficiency
        for record in filtered_records
        if record.logged_stat_efficiency is not None
    ]
    total_stat_values = [
        stat_value / world_size for stat_value in logged_stat_values
    ]

    logged_goodput_values = [
        record.logged_goodput
        for record in filtered_records
        if record.logged_goodput is not None
    ]

    recomputed_goodput_values = []
    for record in filtered_records:
        if record.logged_stat_efficiency is None:
            continue
        total_tps = record.tps_per_device * world_size
        total_stat_efficiency = record.logged_stat_efficiency / world_size
        recomputed_goodput_values.append(total_tps * total_stat_efficiency)

    return RunSummary(
        run_name=run_name,
        world_size=world_size,
        global_batch_size=global_batch_size,
        log_path=log_path,
        num_points_used=len(filtered_records),
        avg_loss=fmean(record.loss for record in filtered_records),
        avg_tps_per_device=fmean(record.tps_per_device for record in filtered_records),
        avg_total_tps=fmean(total_tps_values),
        avg_logged_stat_efficiency=fmean(logged_stat_values)
        if logged_stat_values
        else None,
        avg_total_stat_efficiency=fmean(total_stat_values) if total_stat_values else None,
        avg_logged_goodput=fmean(logged_goodput_values)
        if logged_goodput_values
        else None,
        avg_total_goodput=fmean(recomputed_goodput_values)
        if recomputed_goodput_values
        else None,
    )


def write_summary_csv(output_path: Path, summaries: list[RunSummary]) -> None:
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "run_name",
                "world_size",
                "global_batch_size",
                "log_path",
                "num_points_used",
                "avg_loss",
                "avg_tps_per_device",
                "avg_total_tps",
                "avg_logged_stat_efficiency",
                "avg_total_stat_efficiency",
                "avg_logged_goodput",
                "avg_total_goodput",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "run_name": summary.run_name,
                    "world_size": summary.world_size,
                    "global_batch_size": summary.global_batch_size,
                    "log_path": summary.log_path,
                    "num_points_used": summary.num_points_used,
                    "avg_loss": summary.avg_loss,
                    "avg_tps_per_device": summary.avg_tps_per_device,
                    "avg_total_tps": summary.avg_total_tps,
                    "avg_logged_stat_efficiency": summary.avg_logged_stat_efficiency,
                    "avg_total_stat_efficiency": summary.avg_total_stat_efficiency,
                    "avg_logged_goodput": summary.avg_logged_goodput,
                    "avg_total_goodput": summary.avg_total_goodput,
                }
            )


def make_xtick_labels(summaries: list[RunSummary]) -> list[str]:
    return [
        f"{summary.world_size} GPU\nGBS={summary.global_batch_size}"
        for summary in summaries
    ]


def plot_scaling_summary(output_path: Path, summaries: list[RunSummary]) -> None:
    x_values = [summary.world_size for summary in summaries]
    xtick_labels = make_xtick_labels(summaries)

    figure, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    plots = [
        (
            axes[0],
            [summary.avg_total_tps for summary in summaries],
            "Average Total Throughput",
            "Tokens / sec",
            "#1f77b4",
        ),
        (
            axes[1],
            [
                summary.avg_total_stat_efficiency
                if summary.avg_total_stat_efficiency is not None
                else math.nan
                for summary in summaries
            ],
            "Average Statistical Efficiency",
            "Loss change / total token",
            "#ff7f0e",
        ),
        (
            axes[2],
            [
                summary.avg_total_goodput
                if summary.avg_total_goodput is not None
                else math.nan
                for summary in summaries
            ],
            "Average Goodput",
            "Throughput x stat_eff",
            "#2ca02c",
        ),
    ]

    for axis, y_values, title, ylabel, color in plots:
        axis.plot(x_values, y_values, marker="o", linewidth=2.0, color=color)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)

    axes[-1].set_xticks(x_values)
    axes[-1].set_xticklabels(xtick_labels)
    axes[-1].set_xlabel("DDP world size")

    figure.suptitle("DeepSeek Engram 1B DDP Scaling Summary", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_combined_scaling(output_path: Path, summaries: list[RunSummary]) -> None:
    x_values = [summary.world_size for summary in summaries]
    xtick_labels = make_xtick_labels(summaries)

    throughput_values = [summary.avg_total_tps for summary in summaries]
    stat_eff_values = [
        summary.avg_total_stat_efficiency
        if summary.avg_total_stat_efficiency is not None
        else math.nan
        for summary in summaries
    ]
    goodput_values = [
        summary.avg_total_goodput
        if summary.avg_total_goodput is not None
        else math.nan
        for summary in summaries
    ]

    figure, axis_throughput = plt.subplots(figsize=(11, 6))
    axis_stat_eff = axis_throughput.twinx()
    axis_goodput = axis_throughput.twinx()
    axis_goodput.spines["right"].set_position(("outward", 70))

    line_throughput = axis_throughput.plot(
        x_values,
        throughput_values,
        marker="o",
        linewidth=2.2,
        color="#1f77b4",
        label="Total throughput",
    )[0]
    line_stat_eff = axis_stat_eff.plot(
        x_values,
        stat_eff_values,
        marker="s",
        linewidth=2.2,
        color="#ff7f0e",
        label="Stat efficiency",
    )[0]
    line_goodput = axis_goodput.plot(
        x_values,
        goodput_values,
        marker="^",
        linewidth=2.2,
        color="#2ca02c",
        label="Goodput",
    )[0]

    axis_throughput.set_title(
        "DeepSeek Engram 1B DDP Scaling: Throughput, Stat Efficiency, and Goodput"
    )
    axis_throughput.set_xlabel("DDP world size")
    axis_throughput.set_ylabel("Total throughput (tokens / sec)", color="#1f77b4")
    axis_stat_eff.set_ylabel("Stat efficiency", color="#ff7f0e")
    axis_goodput.set_ylabel("Goodput", color="#2ca02c")

    axis_throughput.set_xticks(x_values)
    axis_throughput.set_xticklabels(xtick_labels)
    axis_throughput.grid(True, alpha=0.3)

    axis_throughput.tick_params(axis="y", colors="#1f77b4")
    axis_stat_eff.tick_params(axis="y", colors="#ff7f0e")
    axis_goodput.tick_params(axis="y", colors="#2ca02c")

    figure.legend(
        [line_throughput, line_stat_eff, line_goodput],
        ["Total throughput", "Stat efficiency", "Goodput"],
        loc="upper left",
        bbox_to_anchor=(0.10, 0.93),
        frameon=False,
    )

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def normalize_to_baseline(values: list[float]) -> list[float]:
    baseline = values[0]
    if baseline == 0:
        return [math.nan for _ in values]
    return [value / baseline for value in values]


def plot_normalized_combined_scaling(
    output_path: Path, summaries: list[RunSummary]
) -> None:
    x_values = [summary.world_size for summary in summaries]
    xtick_labels = make_xtick_labels(summaries)

    throughput_values = normalize_to_baseline(
        [summary.avg_total_tps for summary in summaries]
    )
    stat_eff_values = normalize_to_baseline(
        [
            summary.avg_total_stat_efficiency
            if summary.avg_total_stat_efficiency is not None
            else math.nan
            for summary in summaries
        ]
    )
    goodput_values = normalize_to_baseline(
        [
            summary.avg_total_goodput
            if summary.avg_total_goodput is not None
            else math.nan
            for summary in summaries
        ]
    )

    figure, axis = plt.subplots(figsize=(11, 6))
    axis.plot(
        x_values,
        throughput_values,
        marker="o",
        linewidth=2.2,
        color="#1f77b4",
        label="Normalized throughput",
    )
    axis.plot(
        x_values,
        stat_eff_values,
        marker="s",
        linewidth=2.2,
        color="#ff7f0e",
        label="Normalized stat efficiency",
    )
    axis.plot(
        x_values,
        goodput_values,
        marker="^",
        linewidth=2.2,
        color="#2ca02c",
        label="Normalized goodput",
    )

    axis.axhline(1.0, color="#888888", linestyle="--", linewidth=1.0, alpha=0.8)
    axis.set_title("DeepSeek Engram 1B DDP Scaling: Normalized Metrics")
    axis.set_xlabel("DDP world size")
    axis.set_ylabel("Normalized to 1 GPU baseline")
    axis.set_xticks(x_values)
    axis.set_xticklabels(xtick_labels)
    axis.grid(True, alpha=0.3)
    axis.legend(frameon=False)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_throughput_per_device(output_path: Path, summaries: list[RunSummary]) -> None:
    x_values = list(range(len(summaries)))
    xtick_labels = make_xtick_labels(summaries)
    y_values = [summary.avg_tps_per_device for summary in summaries]

    figure, axis = plt.subplots(figsize=(10, 5))
    bars = axis.bar(x_values, y_values, color="#4e79a7")
    axis.set_title("DeepSeek Engram 1B Throughput Per Device Under DDP Scaling")
    axis.set_ylabel("Tokens / sec / device")
    axis.set_xlabel("DDP world size")
    axis.set_xticks(x_values)
    axis.set_xticklabels(xtick_labels)
    axis.grid(True, axis="y", alpha=0.3)

    for bar, y_value in zip(bars, y_values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            y_value,
            f"{y_value:,.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot DeepSeek Engram 1B DDP goodput scaling results."
    )
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--min-step", type=int, default=10)
    args = parser.parse_args()

    plan_rows = load_plan(args.log_root)

    summaries: list[RunSummary] = []
    missing_logs: list[Path] = []
    for row in plan_rows:
        log_path = Path(row["log_path"])
        if not log_path.exists():
            missing_logs.append(log_path)
            continue
        summary = summarize_run(
            run_name=row["run_name"],
            world_size=int(row["world_size"]),
            global_batch_size=int(row["global_batch_size"]),
            log_path=log_path,
            min_step=args.min_step,
        )
        if summary is not None:
            summaries.append(summary)

    if not summaries:
        missing_text = "\n".join(str(path) for path in missing_logs)
        raise SystemExit(
            "No usable log files were found for plotting.\n"
            f"Checked under: {args.log_root}\n"
            f"Missing logs:\n{missing_text}"
        )

    summaries.sort(key=lambda summary: summary.world_size)

    summary_csv = args.log_root / "ddp_scaling_summary.csv"
    summary_plot = args.log_root / "ddp_scaling_summary.png"
    combined_plot = args.log_root / "ddp_scaling_combined.png"
    normalized_combined_plot = args.log_root / "ddp_scaling_combined_normalized.png"
    throughput_plot = args.log_root / "ddp_throughput_per_device.png"

    write_summary_csv(summary_csv, summaries)
    plot_scaling_summary(summary_plot, summaries)
    plot_combined_scaling(combined_plot, summaries)
    plot_normalized_combined_scaling(normalized_combined_plot, summaries)
    plot_throughput_per_device(throughput_plot, summaries)

    print(f"Wrote summary CSV: {summary_csv}")
    print(f"Wrote plot: {summary_plot}")
    print(f"Wrote plot: {combined_plot}")
    print(f"Wrote plot: {normalized_combined_plot}")
    print(f"Wrote plot: {throughput_plot}")

    if missing_logs:
        print("Skipped missing logs:")
        for log_path in missing_logs:
            print(f"  {log_path}")


if __name__ == "__main__":
    main()
