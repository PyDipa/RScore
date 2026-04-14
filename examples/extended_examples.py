"""
examples/extended_examples.py
======================
Demonstrates the rscore package on both cross-section and panel data.

ANALYTICAL WORKFLOW
-------------------
RScore estimates how strongly each unit (e.g. a firm, province, or country)
responds to a given predictor j.  The recommended reading order is:

  1. R²  — does the model for predictor j fit the data reasonably well?
  2. δ_0 — the average responsiveness to j across all units (the baseline
            effect, worth noting even if not significant).
  3. Wald test — the global gate: are the interaction terms jointly significant?
                 If NO, all scores collapse to the same constant (δ_0); there
                 is no detectable heterogeneity and unit-level scores carry no
                 reliable signal.  Stop here for that predictor.
  4. δ_int — *which* contextual variables modulate responsiveness to j?
              These are the mechanism: they explain *why* some units respond
              differently from others (conditional on the Wald being significant).
  5. Bootstrap CIs — unit-level uncertainty: which individual scores are
                     statistically distinguishable from zero?
  6. RS distribution / maps — read and narrate the geographic or cross-unit
                               heterogeneity.

Run from the project root:
    python examples/extended_examples.py

Prerequisites:
    pip install -e ".[dev]"

"""
##
# import sys
from pathlib import Path

import numpy as np
# import os
# import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Works both as a script and from any working directory


try:
    DATA_DIR = Path(__file__).resolve().parent.parent / "data"
except NameError:
    # Interactive console (REPL, IPython, PyCharm console)
    DATA_DIR = Path.cwd() / "data"

from rscore import RScore
from rscore.statistics import *


## ======================================================================
# CROSS-SECTION EXAMPLE  (Cerulli 2017, Application 1 — auto.dta)
# ======================================================================

print("=" * 60)
print("CROSS-SECTION EXAMPLE")
print("=" * 60)

# -- Load and prepare data ---------------------------------------------
df_raw = pd.read_csv(DATA_DIR/ "auto_cross_section.csv")

y = "price"
predictors = ["mpg", "trunk", "weight", "length", "displacement"]
controls = ["gear_ratio", "headroom"]
cat_cols = ["foreign", "rep78"]

cols_need = [y] + predictors + controls + cat_cols
df_cs = df_raw[cols_need].dropna().copy()


## Standardise continuous variables
continuous = [y] + predictors + controls

# OPTIMAL STANDARDIDATION
standard_scaler = False
if standard_scaler: #gaussian Z-score
    sc = StandardScaler()
    df_cs[continuous] = sc.fit_transform(df_cs[continuous])

# plot data distributions
max_vars = min(9, len(continuous))

fig, ax = plt.subplots(figsize=(10,7),nrows=3,ncols=3)
ax = ax.ravel()
for i,x in enumerate(continuous):
    print(i)
    res = test_standardization(df_cs[x].values)
    recommendation = res['recommendation']
    print(f" *** best dist: { recommendation} ***")
    ax[i].hist(df_cs[x].values, bins=20,label= recommendation,color='grey',alpha =0.7)
    ax[i].set_title(f"{x}, { recommendation}")
    out = standardize_data(df_cs[x],res,plot=False)
    if not standard_scaler:
        df_cs[x] = out['z_scores']

to_del = max_vars - i
for i in range(to_del):
    ax[-i-1].set_axis_off()
plt.tight_layout()

#Z-score plots
fig, ax = plt.subplots(figsize=(10,7),nrows=3,ncols=3)
ax = ax.ravel()
for i,x in enumerate(continuous):
    ax[i].hist(df_cs[x].values, bins=20,label= recommendation,color='grey',alpha =0.7)
    ax[i].set_title(f"{x}, Z-scores")
to_del = max_vars - i
for i in range(to_del):
    ax[-i-1].set_axis_off()
plt.tight_layout()


# sc = StandardScaler()
# df_cs[continuous] = sc.fit_transform(df_cs[continuous])

##
# Build dummies with DROP_FRIST == TRUE
df_cs["rep78"] = df_cs["rep78"].astype(int).astype("category")
X_cat = pd.get_dummies(df_cs[cat_cols], drop_first=True).astype(float)

## -- Fit ----------------------------------------------------------------
model_cs = RScore(mode="cross_section")
model_cs.fit(df_cs, y_col=y, predictors=predictors, controls=controls, X_cat=X_cat)
model_cs.bootstrap(B=1000, seed=42)
# NB
# se wald test fallisce: Tutti gli score collassano attorno alla stessa costante — l'effetto medio di j.
# Non c'è eterogeneità rilevabile: ogni unità risponde a j più o meno allo stesso modo.
# Guardare la distribuzione degli score in quel caso è leggere rumore.
# In pratica: quel fattore j ha un effetto, ma è omogeneo tra le unità — non differenziato tra unità.
# Informativamente utile come stima media, inutile come analisi di eterogeneità.

##
# ======================================================================
# RESULTS
# ======================================================================
# -- Inspect results ----------------------------------------------------
model_cs.summary()

# ----------------------------------------------------------------------
# 1. R² — overall model fit per predictor
print("\n--- 1. Model fit (R²) ---")
for j, r2 in model_cs.R2.items():
    print(f"  {j:>15s}  R² = {r2:.4f}")
print('R² — fit solido e omogeneo (~0.65–0.71). Il modello è stabile tra i fattori, nessun outlier preoccupante.')

# ----------------------------------------------------------------------
# 2. δ_0 — average responsiveness (baseline effect)
print("\n--- 2. Average responsiveness (δ_0) ---")
for j in predictors:
    delta0 = model_cs.models[j].params["Xj"]
    pval   = model_cs.models[j].pvalues["Xj"]
    tag    = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.10 else ""))
    print(f"  {j:>15s}  δ_0 = {delta0:.4f}  p = {pval:.4f} {tag}")
print('δ_0 — solo trunk e weight hanno EFFETTO MEDIO significativo ma questo non esclude la possibilità che gli interaction con'
      'altre variabili sia signifivo. Ne prendiamo atto.')

# ----------------------------------------------------------------------
# 3. Wald test — global gate for heterogeneity
print("\n--- 3. Wald test (joint significance of interaction terms) ---")
interpretable_predictors = []
for j, w in model_cs.wald.items():
    tag     = "***" if w["p_value"] < 0.01 else ("**" if w["p_value"] < 0.05 else ("*" if w["p_value"] < 0.10 else ""))
    verdict = "→ heterogeneity detected" if w["p_value"] < 0.05 else "→ WARNING: scores unreliable, stop here"
    print(f"  {j:>15s}  F = {w['F_stat']:.3f}  p = {w['p_value']:.4f} {tag}  {verdict}")
    if w["p_value"] < 0.05:
        interpretable_predictors.append(j)
print("WALD - solo weight e displacement passano il gate. NB: displacement ha Wald significativo (p=0.02)"
      " ma δ_0 non significativo (p=0.54) — l'effetto medio è nullo, ma la dispersione degli effetti unitari esiste")

# ----------------------------------------------------------------------
# 4. δ_int — which contextual variables drive heterogeneity?
print("\n--- 4. interaction terms (mechanism of heterogeneity) ---")
for j in interpretable_predictors:
    print(f"\n  predictor: {j}")
    print(model_cs.interaction[j].to_string())
print("interaction : significativi per weight e displacement; "
      "displacement è il modulatore principale di weight (p < alpha), "
      "weight è il modulatore principale di displacement (p < alpha),"
      "Sono co-determinati. "
      "length è negativo in entrambi (marginale per weight, significativo per displacement, p=0.003)."
      " mpg e trunk rumore.")


# 5. Bootstrap CIs — which unit-level scores are significant?
print("\n--- 5. Bootstrap inference on unit-level scores ---")
for j in interpretable_predictors:
    boot = model_cs.RS_boot[j]
    n_sig = (boot["p_boot_two_sided"] < 0.05).sum()
    n_tot = len(boot)
    print(f"  {j:>15s}  significant units: {n_sig}/{n_tot} ({100*n_sig/n_tot:.1f}%)")

print("weight: 62/69 unità significative (90%) — eterogeneità solida, score affidabili,"
      " displacement: 0/69 unità significative ( 0%)— paradosso: il Wald rileva eterogeneità aggregata"
      " ma il campione non basta per localizzarla")

# 6. RS distribution — only for predictors that passed the Wald gate
print("\n--- 6. RS distribution (interpretable predictors only) ---")
if interpretable_predictors:
    model_cs.plot_rs(predictors=interpretable_predictors)
    model_cs.plot_rs(predictors=interpretable_predictors, sig_only=True, alpha=0.05)
else:
    print("  No predictors passed the Wald gate — nothing to plot.")

print('Il problema: se il Wald non rileva struttura sistematica negli interaction, '
      'i b_ij non variano in modo spiegabile — sono essenzialmente δ_0 più rumore. '
      'Quei 29% significativi al bootstrap non riflettono eterogeneità reale: '
      'riflettono varianza campionaria dei coefficienti propagata attraverso covariati che non hanno potere modulante. Sono falsi positivi strutturali')

# All scores
model_cs.plot_rs()

# Only WALD significant + significant scores
model_cs.plot_rs(sig_only=True,wald_gate = True, alpha=0.05)


## ======================================================================
# PANEL EXAMPLE  (Cerulli 2017, Application 2 — World Bank)
# ======================================================================

print("\n" + "=" * 60)
print("PANEL EXAMPLE")
print("=" * 60)

# -- Load and prepare data ---------------------------------------------
df_raw_p = pd.read_csv(DATA_DIR / "worldbank_panel.csv")

unit_col = "countrycode"
time_col = "year"
y_col_p = "Y"
predictors_p = ["B", "G", "C", "I", "E", "M"]

cols_need_p = [unit_col, time_col, y_col_p] + predictors_p
df_panel = df_raw_p[cols_need_p].dropna().copy()

focus_countries = ["GBR", "FRA", "ITA", "ESP", "DEU"]

df_panel = df_panel[df_panel[unit_col].isin(focus_countries)].copy()

## Standardise continuous variables
cont_p = [y_col_p] + predictors_p
#--- con questo metodo pool Cross-sectional variation e time variation vengono compresse nella stessa scala

# OPTIMAL STANDARDIDATION
standard_scaler = False
if standard_scaler: #gaussian Z-score
    sc = StandardScaler()
    # ------ pooled (column-wise)) -------------------
    # df_panel[cont_p] = sc.fit_transform(df_panel[cont_p])
    # ------ within-time , unit pooled --------------
    df_panel[cont_p] = df_panel.groupby("year")[cont_p].transform(
    lambda x: (x - x.mean()) / x.std())


# ======================================================================
# OPTIMAL STANDARDIZATION — panel, within-year
# ======================================================================
unit_groups = df_panel[unit_col].values
colors_map = {"FRA": "#2166ac", "ITA": "#d6604d", "ESP": "#4dac26",
              "DEU": "#9970ab", "GBR": "#e08214"}


max_vars = len(cont_p)
ncols = 3
nrows = (max_vars + ncols - 1) // ncols

# --- Collect raw values BEFORE standardization ---
raw_by_var = {}
for x in cont_p:
    raw_by_var[x] = {}
    for c in focus_countries:
        mask = df_panel[unit_col] == c
        raw_by_var[x][c] = df_panel.loc[mask, x].values

# --- Standardization + collect z-scores ---
z_by_var = {}
for x in cont_p:
    res = test_standardization(df_panel[x].values, groups=unit_groups, plot=False)
    recommendation = max(
        set(v['recommendation'] for v in res.values()),
        key=list(v['recommendation'] for v in res.values()).count
    )
    print(f"{x:>5s}  best dist (most common): {recommendation}")

    # HERE df_panel is standardiezed by groups
    if not standard_scaler:
        out = standardize_data(df_panel[x], res, groups=unit_groups, plot=False)
        df_panel[x] = out['z_scores']

    z_by_var[x] = {}
    for c in focus_countries:
        mask = df_panel[unit_col] == c
        z_by_var[x][c] = df_panel.loc[mask, x].values

# --- Plot 1: raw values ---
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4 * nrows))
axes = axes.ravel()
for i, x in enumerate(cont_p):
    ax = axes[i]
    for c in focus_countries:
        ax.hist(raw_by_var[x][c], bins=10, alpha=0.5,
                label=c, color=colors_map[c], density=True)
    ax.set_title(f"{x} — raw values")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)
for j in range(i + 1, len(axes)):
    axes[j].set_axis_off()
plt.suptitle("Raw values by country", fontsize=13)
plt.tight_layout()
plt.show()

# --- Plot 2: z-scores ---
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4 * nrows))
axes = axes.ravel()
for i, x in enumerate(cont_p):
    ax = axes[i]
    for c in focus_countries:
        ax.hist(z_by_var[x][c], bins=10, alpha=0.5,
                label=c, color=colors_map[c], density=True)
    ax.axvline(0, color="black", linestyle=":", linewidth=1)
    ax.set_title(f"{x} — within-unit Z-scores")
    ax.set_xlabel("Z-score")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)
for j in range(i + 1, len(axes)):
    axes[j].set_axis_off()
plt.suptitle("Within-unit standardization by country", fontsize=13)
plt.tight_layout()
plt.show()



## -- Fit ----------------------------------------------------------------
model_p = RScore(mode="panel", unit_col=unit_col)
model_p.fit(df_panel, y_col=y_col_p, predictors=predictors_p)
model_p.bootstrap(B=500, seed=42)

##-- Inspect results ----------------------------------------------------
model_p.summary()
# ======================================================================
# RESULTS — follow the analytical hierarchy
# ======================================================================


# ----------------------------------------------------------------------
# 1. R² — overall model fit per predictor
print("\n--- 1. Model fit (R²) ---")
for j, r2 in model_p.R2.items():
    print(f"  {j:>5s}  R² = {r2:.4f}")
print('R² — fit solido e omogeneo (~0.82-0.83). Il modello è stabile tra i fattori, nessun outlier preoccupante.')

# ----------------------------------------------------------------------
# 2. δ_0 — average responsiveness (baseline effect)
print("\n--- 2. Average responsiveness (δ_0) ---")
for j in predictors_p:
    delta0 = model_p.models[j].params["Xj"]
    pval   = model_p.models[j].pvalues["Xj"]
    tag    = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.10 else ""))
    print(f"  {j:>5s}  δ_0 = {delta0:.4f}  p = {pval:.4f} {tag}")
print('δ_0 — tutti hanno EFFETTO MEDIO significativo')

# ----------------------------------------------------------------------
# 3. Wald test — global gate for heterogeneity
print("\n--- 3. Wald test (joint significance of interaction terms) ---")
interpretable_predictors_p = []
for j, w in model_p.wald.items():
    tag     = "***" if w["p_value"] < 0.01 else ("**" if w["p_value"] < 0.05 else ("*" if w["p_value"] < 0.10 else ""))
    verdict = "→ heterogeneity detected" if w["p_value"] < 0.05 else "→ WARNING: scores unreliable, stop here"
    print(f"  {j:>5s}  F = {w['F_stat']:.3f}  p = {w['p_value']:.4f} {tag}  {verdict}")
    if w["p_value"] < 0.05:
        interpretable_predictors_p.append(j)
print("WALD - solo I ed E passano il gate. Entrambi hanno Wald significativo (p<alpha)"
      " l'effetto medio NON è nullo, E la dispersione degli effetti unitari esiste")

# ----------------------------------------------------------------------
# 4. δ_int — which contextual variables drive heterogeneity?
print("\n--- 4. interaction terms (mechanism of heterogeneity) ---")
for j in interpretable_predictors_p:
    print(f"\n  predictor: {j}")
    print(model_p.interaction[j].to_string())
print("interaction : significativi per I ed E; "
      "E il modulatore principale di I (p < alpha), "
      "I è il modulatore principale di E (p < alpha),"
      "I ed E sono co-determinati. "
      "Altre variabili non significtive, sono solo rumore.")

# ----------------------------------------------------------------------
# 5. Bootstrap CIs — which unit-level scores are significant?
print("\n--- 5. Bootstrap inference on unit-level scores ---")
for j in interpretable_predictors_p:
    boot  = model_p.RS_boot[j]
    # filter to focus countries (one row per country-year, take any year)
    units = model_p._df_aug[model_p.unit_col].values
    mask_focus = pd.Series(units).isin(focus_countries).values
    n_sig = (boot.loc[mask_focus, "p_boot_two_sided"] < 0.05).sum()
    n_tot = mask_focus.sum()
    print(f"  {j:>5s}  significant obs (focus countries): {n_sig}/{n_tot} ({100*n_sig/n_tot:.1f}%)")
print("I ed E - terogeneità solida, score affidabili")



# 6. RS distribution — only for predictors that passed the Wald gate
print("\n--- 6. RS distribution & panel diagnostics (focus countries only) ---")
if interpretable_predictors_p:
    # split per paese
    model_p.plot_rs(predictors=interpretable_predictors_p,
                    group=df_panel[unit_col].values,
                    group_name="country",
                    wald_gate=True, sig_only=True)

    # split per anno
    model_p.plot_rs(predictors=interpretable_predictors_p,
                    group=df_panel[time_col].values,
                    group_name="year",
                    wald_gate=True, sig_only=True)
    model_p.plot_decompose(predictors=interpretable_predictors_p, wald_gate=True, sig_only=True)
    for j in interpretable_predictors_p:
        model_p.plot_timeseries(predictor=j, units=focus_countries, time_col=time_col)
else:
    print("  No predictors passed the Wald gate — nothing to plot.")



## ======================================================================
# PANEL - GEOGRAFIC - DATASET  EXAMPLE  (Di Paola et al 2023)
# ======================================================================

print("\n" + "=" * 60)
print("Geografic-PANEL EXAMPLE")
print("=" * 60)

# PRE-PROCESSING
#  -- Load data ---------------------------------------------------------
yields_raw = pd.read_csv(DATA_DIR / "Clean_dataset_olive_yields_107prov.csv")
drivers = pd.read_csv(DATA_DIR / "Climate_Variables.csv")
assert drivers.shape[0] == 990, "Expected 66 × 15 = 990 rows in drivers"

# -- Filter yields to the 66 provinces with a complete time series 2006-2020 -------------
YEAR_START, YEAR_END = 2006, 2020

subset = yields_raw[yields_raw["is_complete_ts"] == 1].copy()
provinces = subset["province"].unique()
print(f"Provinces with is_complete_ts==1: {len(provinces)}")  # must be 66

# Select only year columns in [2006, 2020] — discard 2021-2024 before melt
year_cols = [c for c in subset.columns
             if str(c).isdigit() and YEAR_START <= int(c) <= YEAR_END]

y_long = (
    subset[["province"] + year_cols]
    .melt(id_vars="province", var_name="year", value_name="yield")
)
y_long["year"] = y_long["year"].astype(int)

print(f"y_long shape : {y_long.shape}")  # should be ≤990

# -- Prepare drivers ---------------------------------------------------
col_predictors = ["Tmax_b1", "Tave_b2", "Tmin_b3", "Tmax_b5", "GDD_b6"]

drivers_sel = drivers[['province','year'] + col_predictors]
# -- Merge on (prov, year) and drop NaNs --------------------------------
df_geo =  drivers_sel.copy()
df_geo.insert(2,"yield",y_long["yield"])

print(f"df_geo shape : {df_geo.shape}")
print(f"Provinces in model : {df_geo['province'].nunique()}")
print(f"Years in model     : {df_geo['year'].unique()}")

# -- Standardize (within-unit, per province) ---------------------------
# Rationale: RS measures sensitivity relative to each province's own
# historical baseline. Global pooling would conflate between-province

# yield: within-province standardization (distribution typically non-Gaussian)
# col_predictors: already standardized as Z-scores in the original dataset — do not re-scale

standard_scaler = False
unit_groups  = df_geo["province"].values
if standard_scaler:
    sc = StandardScaler()
    df_geo['yield'] = sc.fit_transform(df_geo['yield'])
else:

    res = test_standardization(df_geo['yield'].values, groups=unit_groups, plot=False)
    recommendation = max(
        set(v["recommendation"] for v in res.values()),
        key=list(v["recommendation"] for v in res.values()).count,
    )
    print([v["recommendation"] for v in res.values()])
    out = standardize_data(df_geo['yield'], res, groups=unit_groups, plot=False)
    df_geo["y (z_score)"] = out["z_scores"]

# -- Initialize Rscore and Fit ---------------------------------------------------------------
model_geo = RScore(mode="panel", unit_col="province")
model_geo.fit(df_geo, y_col="y (z_score)", predictors=col_predictors)
# No controls, no categoricals.
model_geo.bootstrap(B=500, seed=42)

# ======================================================================
# RESULTS — analytical hierarchy
# ======================================================================

model_geo.summary()

# 1. R² -------------------------------------------------------
print("\n--- 1. Model fit (R²) ---")
for j, r2 in model_geo.R2.items():
    print(f"  {j:>12s}  R² = {r2:.4f}")
# R² values (~0.10) reflect within-province temporal variation after removing
# fixed effects; selected climate variables alone explain ~10% of yield variability,
# likely due to the coarse quality of yield data

# 2. δ_0 -------------------------------------------------------
print("\n--- 2. Average responsiveness (δ_0) ---")
for j in col_predictors:
    delta0 = model_geo.models[j].params["Xj"]
    pval = model_geo.models[j].pvalues["Xj"]
    tag = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.10 else "ns"))
    print(f"  {j:>12s}  δ_0 = {delta0:+.4f}  p = {pval:.4f} {tag}")
# The sign pattern of δ_0 is phenologically consistent with the results in Di Paola et al 2023 (fig.4)
#  negative effects results from Tmax_b1,Tave_b2 and GDD_b6; positive for Tmin_b3 and Tmax_b5

# 3. Wald test — gate -------------------------------------------------------
print("\n--- 3. Wald test (joint significance of interaction terms) ---")

interpretable_geo = []
for j, w in model_geo.wald.items():
    tag = "***" if w["p_value"] < 0.01 else ("**" if w["p_value"] < 0.05 else ("*" if w["p_value"] < 0.10 else "ns"))
    verdict = "→ heterogeneity detected" if w["p_value"] < 0.05 else "→ scores collapse to δ_0 — stop here"
    print(f"  {j:>12s}  F = {w['F_stat']:.3f}  p = {w['p_value']:.4f} {tag}  {verdict}")
    if w["p_value"] < 0.05:
        interpretable_geo.append(j)
# Temperature sensitivity in the first bi-months is homogeneous across provinces
# — a shared structural vulnerability of Italian olive production.
# Heterogeneity emerges only in the harvest phase (GDD_b6),
# suggesting that local context modulates heat stress tolerance specifically
# during fruit maturation and collection.
# ===============================================================================
# SIGNIFICANT WARD ARE EXPETECTE TO SHOW HIGHEST STD IN RScore AMONG PROVINCES
# ===============================================================================

# 4. interaction
print("\n--- 4. interaction terms (mechanism of heterogeneity) ---")
for j in interpretable_geo:
    print(f"\n  predictor: {j}")
    print(model_geo.interaction[j].to_string())
# The responsiveness to harvest-phase heat accumulation (GDD_b6) is amplified
# in provinces experiencing warmer winters (Tmax_b1): a compound climate stress
# hypothesized and discussed in Di Paola et al 2023.
# pathway where insufficient winter chilling increases vulnerability to
# end-of-season heat. This cascade effect represents the primary mechanism
# of spatial heterogeneity in olive yield sensitivity.

# 5. Bootstrap
print("\n--- 5. Bootstrap inference on unit-level scores ---")
for j in interpretable_geo:
    boot = model_geo.RS_boot[j]
    n_sig = (boot["p_boot_two_sided"] < 0.05).sum()
    n_tot = len(boot)
    print(f"  {j:>12s}  significant obs: {n_sig}/{n_tot} ({100 * n_sig / n_tot:.1f}%)")

# 6. Plots
# ---------------------------------------------------------
# import shapefile:
import geopandas as gpd
S = gpd.read_file(DATA_DIR / "shapefile_IT/ProvCM01012022_WGS84.shp")

df_rs = df_geo[["province", "year"]].copy()
rs_cols = []
for j in col_predictors:
    mask_ns = model_geo.RS_boot[j]["p_boot_two_sided"].values >= 0.05
    # print(len(mask_ns[mask_ns==False]))
    rs_val = model_geo.RS[j].values.copy()
    rs_val[mask_ns] = np.nan
    df_rs[f"rs_{j}"] = rs_val
    rs_cols.append(f"rs_{j}")


# -- Template 107 province (una volta sola) ----------------------------
predictor ="rs_GDD_b6"# "rs_Tmax_b1"
year = 2014

gdf = S[["DEN_UTS", "geometry"]].copy()


rs_mean = df_rs.groupby("province")[rs_cols].mean()
rs_std = df_rs.groupby("province")[rs_cols].std()


gdf["mean_f"] = gdf["DEN_UTS"].map(rs_mean[predictor])
gdf["sd_f"] = gdf["DEN_UTS"].map(rs_std[predictor])

rs_year = df_rs[df_rs["year"] == year].set_index("province")[predictor]
gdf["year_i"] = gdf["DEN_UTS"].map(rs_year)



def plot_map(gdf, col, title="", figsize=(8, 8), cmap_div="RdBu_r", cmap_seq="YlOrRd"):
    """
    Auto-selects colormap and norm:
    - diverging (TwoSlopeNorm, RdBu_r) if values span both signs or col contains 'mean' or 'rs_'
    - sequential (YlOrRd)              if values are all non-negative (e.g. std)
    """
    import matplotlib.colors as mcolors
    vals = gdf[col].dropna().values
    diverging = (vals.min() < 0) or any(s in col for s in ("mean", "rs_"))

    if diverging:
        vmax = np.nanmax(np.abs(vals))
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        cmap = cmap_div
    else:
        norm = mcolors.Normalize(vmin=0, vmax=vals.max())
        cmap = cmap_seq

    fig, ax = plt.subplots(figsize=figsize)
    gdf.plot(
        column=col, ax=ax,
        cmap=cmap, norm=norm,
        legend=True,
        legend_kwds={"label": col, "orientation": "horizontal"},
        edgecolor="black", linewidth=0.5,
        missing_kwds=dict(color="lightgrey"),
    )
    ax.set_title(title or col)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    return fig, ax

plot_map(gdf, "mean_f", title=f"Mean RS · {predictor} · sig only")
plot_map(gdf, "sd_f",   title=f"Std RS  · {predictor}· temporal variability")
plot_map(gdf, "year_i",   title=f"RS · {predictor} · {year}")



