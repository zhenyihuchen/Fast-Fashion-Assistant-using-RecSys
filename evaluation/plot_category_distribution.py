"""Bar chart of product distribution across categories in the Zara catalog."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PARQUET = ROOT / "offline" / "data" / "women_data.parquet"

df = pd.read_parquet(PARQUET)
counts = df["product_category"].value_counts().sort_values(ascending=True)

# Highlight categories above median count
median_count = counts.median()
colors = ["#4A90D9" if v > median_count else "#B0B0B0" for v in counts.values]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(counts.index, counts.values, color=colors, edgecolor="white")

ax.bar_label(bars, padding=4, fontsize=9)
ax.set_xlabel("Number of Items")
ax.set_title(f"Distribution of Products Across Categories (N = {len(df):,})")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out = ROOT / "evaluation" / "category_distribution.png"
plt.savefig(out, dpi=150)
print(f"Saved to {out}")
