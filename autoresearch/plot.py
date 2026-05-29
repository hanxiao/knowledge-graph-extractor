#!/usr/bin/env python3
"""
Render the autoresearch progress figure, in the style of karpathy/autoresearch's
overnight plot (metric vs experiment, with a best-so-far envelope). Here the
metric is decode tok/s (higher is better, so the envelope climbs).

  uv run --with matplotlib plot.py     # -> progress.png
"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))

rows = []
with open(os.path.join(HERE, "experiments.jsonl")) as f:
    for line in f:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
rows = [r for r in rows if "metrics" in r and r["metrics"].get("decode_tps")]

baseline = json.load(open(os.path.join(HERE, "baseline.json")))
base_tps = baseline["metrics"]["decode_tps"]
conf = json.load(open(os.path.join(HERE, "confirmation.json")))

COLOR = {"baseline": "#888888", "KEEP": "#2e7d32",
         "QUALITY_OK_NO_SPEEDUP": "#1565c0", "REJECT": "#c62828", "ERROR": "#c62828"}

xs = list(range(1, len(rows) + 1))
ys = [r["metrics"]["decode_tps"] for r in rows]
verd = [r.get("verdict", "?") for r in rows]
ids = [r["id"] for r in rows]

# best-so-far among quality-passing experiments (not REJECT/ERROR)
best, cur = [], 0.0
for r in rows:
    ok = r.get("verdict") not in ("REJECT", "ERROR")
    if ok:
        cur = max(cur, r["metrics"]["decode_tps"])
    best.append(cur if cur > 0 else None)

plt.style.use("seaborn-v0_8-whitegrid")
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                              gridspec_kw={"width_ratios": [2.4, 1]})

# ---- left: trajectory ----
ax.axhline(base_tps, ls="--", lw=1.4, color="#444", zorder=1,
           label=f"baseline {base_tps:.1f} t/s")
ax.step(xs, best, where="post", color="#2e7d32", lw=2.2, zorder=2,
        label="best so far (quality-passing)")
for x, y, v in zip(xs, ys, verd):
    ax.scatter(x, y, s=70, color=COLOR.get(v, "#555"), edgecolor="white",
               linewidth=0.8, zorder=3)
# annotate the run-best point
bi = max(range(len(rows)), key=lambda i: ys[i])
ax.annotate(f"{ids[bi]}\n{ys[bi]:.1f} t/s", (xs[bi], ys[bi]),
            textcoords="offset points", xytext=(6, 10), fontsize=9, color="#2e7d32")
ax.set_xlabel("experiment #")
ax.set_ylabel("decode throughput (tok/s)")
ax.set_title("autoresearch: Qwen3.6-35B-A3B KI-extraction decode on L4\n"
             "(karpathy-style progress; higher is better)", fontsize=11)
ax.set_ylim(min(ys) - 3, max(ys) + 4)
handles = [Line2D([0], [0], marker="o", ls="", color=COLOR[k],
                  label=k.replace("QUALITY_OK_NO_SPEEDUP", "quality-ok, no speedup").lower())
           for k in ["KEEP", "QUALITY_OK_NO_SPEEDUP", "REJECT", "baseline"]]
handles += [Line2D([0], [0], ls="--", color="#444", label=f"baseline {base_tps:.1f}"),
            Line2D([0], [0], color="#2e7d32", lw=2.2, label="best so far")]
ax.legend(handles=handles, fontsize=8, loc="lower right", framealpha=0.9)

# ---- right: 5-repeat confirmation (the rigorous result) ----
names = [c["id"] for c in conf]
means = [c["decode_mean"] for c in conf]
stds = [c["decode_std"] for c in conf]
cols = ["#888888" if n == "baseline" else "#2e7d32" for n in names]
bars = ax2.bar(range(len(conf)), means, yerr=stds, capsize=5, color=cols,
               edgecolor="white", width=0.62)
for i, c in enumerate(conf):
    delta = (c["decode_mean"] / means[0] - 1) * 100
    lbl = f"{c['decode_mean']:.2f}\n({delta:+.1f}%)" if i else f"{c['decode_mean']:.2f}"
    ax2.text(i, c["decode_mean"] + stds[i] + 0.15, lbl, ha="center", fontsize=9)
ax2.set_xticks(range(len(conf)))
ax2.set_xticklabels([n.replace("mtp_", "") for n in names], rotation=20, ha="right", fontsize=9)
ax2.set_ylabel("decode tok/s (mean ± std, 5 repeats)")
ax2.set_title("confirmed result\n(coverage 1.0 = zero quality loss)", fontsize=11)
ax2.set_ylim(min(means) - 2, max(m + s for m, s in zip(means, stds)) + 2)

fig.tight_layout()
out = os.path.join(HERE, "progress.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
