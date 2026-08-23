"""Generate PR/F1/confidence-reliability charts (run in full environment)."""
import json, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.loads(pathlib.Path("results/_week6_charts.json").read_text("utf-8"))
out = pathlib.Path("results/v3_analysis")

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(d["epochs"], d["prec"], label="Precision", color="#4472C4")
ax.plot(d["epochs"], d["rec"], label="Recall", color="#ED7D31")
ax.plot(d["epochs"], d["f1"], label="F1", color="#70AD47")
ax.set_xlabel("Epoch"); ax.set_ylabel("Score"); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(out/"PR_curve.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(d["epochs"], d["m50"], label="mAP50", color="#4472C4")
ax.plot(d["epochs"], d["m95"], label="mAP50-95", color="#ED7D31")
ax.set_xlabel("Epoch"); ax.set_ylabel("mAP"); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(out/"F1_curve.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(9,5))
ax.bar(d["bins"], d["acc"], width=0.07, color="#4472C4")
ax.set_xlabel("Confidence threshold"); ax.set_ylabel("Accuracy")
ax.set_title("Confidence Reliability"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(out/"confidence_reliability.png", dpi=150); plt.close(fig)
print("charts generated")
