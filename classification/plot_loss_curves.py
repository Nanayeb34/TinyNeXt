"""
Plot per-epoch training/val loss and top-1/top-5 accuracy
for TinyNeXt-T trained on 50k / 100k / 150k / 200k subsets.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

SIZES   = ["50k", "100k", "150k", "200k"]
COLORS  = ["#EF4444", "#F97316", "#3B82F6", "#10B981"]
LOG_DIR = Path("classification/logs/tinynext_t")

data = {}
for size in SIZES:
    with open(LOG_DIR / size / "stats.json") as f:
        data[size] = json.load(f)

# ── Figure: 2×2 grid ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 9))
gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.28)

axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]
titles = ["Training Loss", "Validation Loss", "Top-1 Accuracy (%)", "Top-5 Accuracy (%)"]
keys   = ["train_loss",   "val_loss",         "top1",               "top5"]

for ax, title, key in zip(axes, titles, keys):
    for size, color in zip(SIZES, COLORS):
        d = data[size]
        ax.plot(d["epoch"], d[key], color=color, linewidth=1.6, label=size)

        # Mark final value
        ax.annotate(
            f"{d[key][-1]:.2f}{'%' if 'top' in key else ''}",
            xy=(d["epoch"][-1], d[key][-1]),
            xytext=(3, 0), textcoords="offset points",
            fontsize=7, color=color, va="center",
        )

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, max(data["50k"]["epoch"]))

# LR on top of train loss (secondary axis)
ax_lr = axes[0].twinx()
for size, color in zip(SIZES, COLORS):
    ax_lr.plot(data[size]["epoch"], data[size]["lr"],
               color=color, linewidth=0.8, linestyle=":", alpha=0.5)
ax_lr.set_ylabel("LR", fontsize=7, color="gray")
ax_lr.tick_params(axis="y", labelsize=6, colors="gray")

# Shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="Train size", loc="upper center",
           ncol=4, fontsize=9, bbox_to_anchor=(0.5, 1.01), framealpha=0.9)

fig.suptitle("TinyNeXt-T  ·  50-epoch training curves", fontsize=13, y=1.04)

out = "classification/loss_curves.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")

# ── Print final numbers ───────────────────────────────────────────────────────
print(f"\n{'Size':<8} {'Train Loss':>11} {'Val Loss':>10} {'Top-1':>7} {'Top-5':>7}")
print("-" * 48)
for size in SIZES:
    d = data[size]
    print(f"{size:<8} {d['train_loss'][-1]:>11.4f} {d['val_loss'][-1]:>10.4f}"
          f" {d['top1'][-1]:>6.2f}% {d['top5'][-1]:>6.2f}%")
