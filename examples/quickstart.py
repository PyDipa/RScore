"""
examples/quickstart.py
======================
Minimal working example of rscore on cross-section data.
For a full walkthrough (diagnostics, panel, geographic) see extended_example.py.

Run from project root:
    python examples/quickstart.py

Prerequisites:
    pip install -e ".[dev]"
"""
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from rscore import RScore

try:
    DATA_DIR = Path(__file__).resolve().parent.parent / "data"
except NameError:
    DATA_DIR = Path.cwd() / "data"

# -- Load & prepare ----------------------------------------------------
df = pd.read_csv(DATA_DIR / "auto_cross_section.csv")

y       = "price"
predictors = ["mpg", "trunk", "weight", "length", "displacement"]
controls = ["gear_ratio", "headroom"]
cat_cols = ["foreign", "rep78"]

cols_need = [y] + predictors + controls + cat_cols
df = df[cols_need].dropna().copy()

# Global Z-score standardization (continuous variables only)
continuous = [y] + predictors + controls
sc = StandardScaler()
df[continuous] = sc.fit_transform(df[continuous])

# Categorical dummies
df["rep78"] = df["rep78"].astype(int).astype("category")
X_cat = pd.get_dummies(df[cat_cols], drop_first=True).astype(float)

# -- Fit & bootstrap ---------------------------------------------------
model = RScore(mode="cross_section")
model.fit(df, y_col=y, predictors=predictors, controls=controls, X_cat=X_cat)
model.bootstrap(B=500, seed=42)

# -- Inspect -----------------------------------------------------------
model.summary()

# Distribution of all RS; significant only (Wald-gated, α=0.05)
model.plot_rs()
model.plot_rs(sig_only=True, wald_gate=False, alpha=0.05)