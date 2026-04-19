"""
Scaling curve: TinyNeXt-T Top-1 / Top-5 vs training set size.
Uses a saturating (Michaelis-Menten) fit + bootstrap confidence bands.
Extrapolates to full ImageNet (1.28M).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.optimize import curve_fit

# ── Data ──────────────────────────────────────────────────────────────────────
sizes = np.array([50_000, 100_000, 150_000, 200_000], dtype=float)
top1  = np.array([28.67,  40.52,   46.31,   51.81])
top5  = np.array([52.27,  65.13,   70.76,   75.71])

FULL_IMAGENET = 1_281_167.0

# ── Model: y = L * x / (K + x)  (saturates at L) ─────────────────────────────
def mm_fit(x, L, K):
    return L * x / (K + x)

def fit_and_ci(x, y, x_plot, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    # Central fit
    p0 = [y.max() * 1.5, x.mean()]
    bounds = ([y.max(), 0], [200, 1e9])
    popt, _ = curve_fit(mm_fit, x, y, p0=p0, bounds=bounds, maxfev=20_000)
    y_fit = mm_fit(x_plot, *popt)

    # Bootstrap CI
    boot_curves = []
    for _ in range(n_boot):
        idx = rng.choice(len(x), len(x), replace=True)
        try:
            pb, _ = curve_fit(mm_fit, x[idx], y[idx], p0=popt,
                              bounds=bounds, maxfev=5_000)
            boot_curves.append(mm_fit(x_plot, *pb))
        except Exception:
            pass
    boot_curves = np.array(boot_curves)
    lo = np.percentile(boot_curves, 5,  axis=0)
    hi = np.percentile(boot_curves, 95, axis=0)

    extrap = mm_fit(FULL_IMAGENET, *popt)
    return y_fit, lo, hi, popt, extrap

x_plot = np.logspace(np.log10(40_000), np.log10(FULL_IMAGENET * 1.1), 500)

y1_fit, lo1, hi1, p1, extrap1 = fit_and_ci(sizes, top1, x_plot)
y5_fit, lo5, hi5, p5, extrap5 = fit_and_ci(sizes, top5, x_plot)

print(f"Saturating fit  (L = asymptote, K = half-saturation point)")
print(f"  Top-1: L={p1[0]:.1f}%  K={p1[1]/1e3:.0f}k   →  extrap @ 1.28M = {extrap1:.1f}%")
print(f"  Top-5: L={p5[0]:.1f}%  K={p5[1]/1e3:.0f}k   →  extrap @ 1.28M = {extrap5:.1f}%")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))

BLUE, GREEN = "#2563EB", "#16A34A"

# CI bands
ax.fill_between(x_plot, lo1, hi1, alpha=0.15, color=BLUE)
ax.fill_between(x_plot, lo5, hi5, alpha=0.15, color=GREEN)

# Fitted curves (solid up to 200k, dashed beyond)
cutoff = np.searchsorted(x_plot, 200_000)
ax.plot(x_plot[:cutoff], y1_fit[:cutoff], color=BLUE,  linewidth=2)
ax.plot(x_plot[:cutoff], y5_fit[:cutoff], color=GREEN, linewidth=2)
ax.plot(x_plot[cutoff:], y1_fit[cutoff:], color=BLUE,  linewidth=2,
        linestyle="--", label="Top-1 extrapolation")
ax.plot(x_plot[cutoff:], y5_fit[cutoff:], color=GREEN, linewidth=2,
        linestyle="--", label="Top-5 extrapolation")

# Measured data points
ax.scatter(sizes, top1, color=BLUE,  s=75, zorder=5, label="Top-1 measured")
ax.scatter(sizes, top5, color=GREEN, s=75, zorder=5, label="Top-5 measured")

# Extrapolated stars
ax.scatter([FULL_IMAGENET], [extrap1], color=BLUE,  s=160, marker="*",
           zorder=6, label=f"Top-1 est. {extrap1:.1f}% @ 1.28M")
ax.scatter([FULL_IMAGENET], [extrap5], color=GREEN, s=160, marker="*",
           zorder=6, label=f"Top-5 est. {extrap5:.1f}% @ 1.28M")

# Vertical line at full ImageNet
ax.axvline(FULL_IMAGENET, color="gray", linestyle=":", linewidth=1.2, alpha=0.8)
ax.text(FULL_IMAGENET * 0.92, 12, "Full ImageNet\n(1.28M)",
        ha="right", va="bottom", fontsize=8, color="gray")

# Annotations for extrapolated values
for val, color, dy in [(extrap1, BLUE, -9), (extrap5, GREEN, 5)]:
    ax.annotate(f"{val:.1f}%",
                xy=(FULL_IMAGENET, val),
                xytext=(-60, dy), textcoords="offset points",
                fontsize=9, color=color, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=color, alpha=0.5))

ax.set_xscale("log")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: (f"{int(x/1000)}k" if x < 1_000_000 else f"{x/1e6:.2f}M")
))
ax.set_xlabel("Training set size", fontsize=11)
ax.set_ylabel("ImageNet Accuracy (%)", fontsize=11)
ax.set_title(
    "TinyNeXt-T Scaling Curve  ·  50 epochs  ·  Saturating fit + 90% CI",
    fontsize=11,
)
ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
ax.set_ylim(10, 95)
ax.set_xlim(38_000, FULL_IMAGENET * 1.3)
ax.grid(True, which="both", alpha=0.25)

note = ("Dashed lines = extrapolation beyond training range (50-epoch runs).\n"
        "Shaded bands = 90% bootstrap CI.  Full training uses 100 epochs.")
ax.text(0.01, 0.01, note, transform=ax.transAxes,
        fontsize=7.5, color="gray", va="bottom")

plt.tight_layout()
out = "classification/scaling_curve.png"
plt.savefig(out, dpi=150)
print(f"\nSaved → {out}")
