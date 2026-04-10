"""Plot training progress from training_log.jsonl."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_log(path: Path) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def main() -> None:
    log_path = Path("checkpoints/training_log.jsonl")
    if not log_path.exists():
        print(f"No log file found at {log_path}")
        return

    entries = load_log(log_path)
    if not entries:
        print("Log file is empty")
        return

    iterations = [e["iteration"] for e in entries]

    # ── Winrate chart ────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    depths = sorted({d for e in entries for d in e.get("eval", {})})
    for depth in depths:
        winrates = []
        for e in entries:
            r = e.get("eval", {}).get(str(depth))
            winrates.append(r["winrate"] if r else None)
        axes[0].plot(iterations, winrates, marker="o", markersize=3, label=f"vs minimax d{depth}")

    axes[0].set_ylabel("Win rate (%)")
    axes[0].set_ylim(-5, 105)
    axes[0].axhline(y=50, color="gray", linestyle="--", alpha=0.5)
    axes[0].legend()
    axes[0].set_title("Eval Win Rate by Minimax Depth")
    axes[0].grid(True, alpha=0.3)

    # ── Loss chart ───────────────────────────────────────────────
    p_losses = [e["policy_loss"] for e in entries if e.get("policy_loss") is not None]
    v_losses = [e["value_loss"] for e in entries if e.get("value_loss") is not None]
    loss_iters = [e["iteration"] for e in entries if e.get("policy_loss") is not None]

    ax_loss = axes[1]
    ax_loss.plot(loss_iters, p_losses, marker="o", markersize=3, label="Policy loss", color="tab:blue")
    ax_loss.set_ylabel("Policy loss", color="tab:blue")
    ax_loss.tick_params(axis="y", labelcolor="tab:blue")
    ax_loss.set_xlabel("Iteration")
    ax_loss.grid(True, alpha=0.3)

    ax_val = ax_loss.twinx()
    ax_val.plot(loss_iters, v_losses, marker="s", markersize=3, label="Value loss", color="tab:orange")
    ax_val.set_ylabel("Value loss", color="tab:orange")
    ax_val.tick_params(axis="y", labelcolor="tab:orange")

    lines_1, labels_1 = ax_loss.get_legend_handles_labels()
    lines_2, labels_2 = ax_val.get_legend_handles_labels()
    ax_loss.legend(lines_1 + lines_2, labels_1 + labels_2)

    axes[1].set_title("Training Loss")

    plt.tight_layout()
    plt.savefig("checkpoints/training_progress.png", dpi=150)
    print("Saved checkpoints/training_progress.png")
    plt.show()


if __name__ == "__main__":
    main()
