# chart.py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from PIL import Image

# ------------------
# 1) Generate realistic synthetic data
# ------------------
np.random.seed(42)

n_customers = 2000
segments = [
    ("Low Value", 0.8, 50, 200),
    ("Mid Value", 1.2, 120, 500),
    ("High Value", 1.8, 400, 1500),
    ("VIP", 2.2, 1200, 6000)
]

rows = []
for label, shape, scale, median in segments:
    size = int(n_customers * 0.25)
    vals = np.random.gamma(shape=shape, scale=scale, size=size)
    outliers = np.random.choice([0, 1], size=size, p=[0.97, 0.03])
    vals = vals + outliers * np.random.uniform(median * 1.5, median * 6, size=size)
    for v in vals:
        rows.append((label, float(v)))

df = pd.DataFrame(rows, columns=["segment", "purchase_amount"])

df["purchase_amount"] = df["purchase_amount"].clip(lower=5.0)

# ------------------
# 2) Seaborn styling and plot creation
# ------------------
sns.set_style("whitegrid")
sns.set_context("talk", font_scale=1.0)

palette = sns.color_palette("Set2")

plt.figure(figsize=(8, 8))  # 8 inches * 64 dpi = 512px
ax = sns.boxplot(
    x="segment",
    y="purchase_amount",
    data=df,
    palette=palette,
    showfliers=True,
    linewidth=1.2
)

ax.set_title("Purchase Amount Distribution by Customer Segment",
             fontsize=18, weight='bold', pad=18)
ax.set_xlabel("Customer Segment", fontsize=14)
ax.set_ylabel("Purchase Amount (₹)", fontsize=14)

def currency_formatter(x, pos):
    if x >= 1000:
        return f"₹{int(x):,}"
    return f"₹{int(x)}"

ax.yaxis.set_major_formatter(FuncFormatter(currency_formatter))

plt.xticks(rotation=0)
plt.tight_layout()

# Annotate medians
medians = df.groupby('segment')['purchase_amount'].median()
for pos, seg in enumerate(df['segment'].unique()):
    med = medians[seg]
    ax.text(pos, med * 1.05, f"₹{int(med):,}",
            ha='center', fontsize=11, weight='semibold')

# ------------------
# 3) Save chart (initial save)
# ------------------
plt.savefig("chart.png", dpi=64, bbox_inches='tight')
plt.close()

# ------------------
# 4) FORCE RESIZE → Guaranteed 512×512
# ------------------
img = Image.open("chart.png")
img = img.resize((512, 512), Image.LANCZOS)
img.save("chart.png")
