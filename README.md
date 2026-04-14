# rscore

**Responsiveness Scores for heterogeneous unit-level slope estimation.**

A Python implementation of the methodology from
[Cerulli (2017)](https://doi.org/10.1177/1536867X1701700208), extended with:

- **Bootstrap inference** on unit-specific scores (propagating parametric
  uncertainty from interaction coefficients to individual RS)
- **Wald test** for joint significance of interaction terms
- **Variance decomposition** (between vs within) for panel data

## The idea

Standard regression gives one slope per predictor — the same for everyone.
RScore gives **one slope per unit**, by adding interaction terms between
each focal predictor and the others. The result is a matrix
(observations × predictors) of local, unit-specific slopes that reveal
*who* responds strongly and *why*.

For a worked example, the full model derivation, and interpretation
guidelines, see [`docs/methodology.md`](docs/methodology.md).

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/rscore.git
cd rscore
pip install -e ".[dev]"
```

## Quick start

### Cross-section

```python
from rscore import RScore
from sklearn.preprocessing import StandardScaler
import pandas as pd

df = pd.read_csv("data.csv").dropna()

# Standardise continuous variables (required before fitting)
scaler = StandardScaler()
df[["price", "mpg", "weight"]] = scaler.fit_transform(
    df[["price", "mpg", "weight"]]
)

model = RScore(mode="cross_section")
model.fit(df, y_col="price",
          predictors=["mpg", "weight"],
          controls=["headroom"])

# Check Wald test first — if it fails, scores are noise
model.summary()

model.bootstrap(B=1000, seed=42)

# Distribution of all scores
model.plot_rs()

# Only significant scores (bootstrap p < 0.05)
model.plot_rs(sig_only=True, alpha=0.05)
```

### Panel

```python
model = RScore(mode="panel", unit_col="province")
model.fit(df, y_col="drought_index",
          predictors=["precip", "temp", "ndvi"])

model.summary()
model.bootstrap(B=500, seed=42)

# Variance decomposition: between-unit vs within-unit
fig, axes, decomp_table = model.plot_decompose()

# Time series for selected units
model.plot_timeseries(predictor="precip",
                      units=["TE", "AQ", "RM"],
                      time_col="year")
```

## API reference

### Core workflow

| Method | Description |
|--------|-------------|
| `fit(df, y_col, predictors, controls, X_cat)` | Estimate RS for each predictor. One regression per predictor, each with its own interaction terms and R². |
| `bootstrap(B, seed, alpha)` | Bootstrap inference on unit scores. Cross-section: iid resampling. Panel: cluster resampling by unit. |
| `summary()` | Print R² per predictor, Wald tests, mean predictor responses, significant interaction counts. |

### Plotting

| Method | Description |
|--------|-------------|
| `plot_rs(predictors, group, sig_only, wald_gate, alpha)` | KDE of score distributions. Optionally split by a grouping variable (e.g. country, year). Use `sig_only=True` to show only bootstrap-significant scores, `wald_gate=True` to suppress predictors that failed the Wald test. |
| `plot_decompose(predictors, sig_only, wald_gate, alpha)` | Between-unit vs within-unit variance decomposition (panel only). Shows absolute variance and percentage composition side by side. Returns the decomposition table as a DataFrame. |
| `plot_timeseries(predictor, units, time_col, sig_only, alpha)` | RS time series for selected units (panel only). One line per unit, with optional masking of non-significant observations. |

### Key attributes (after `fit`)

| Attribute | Type | Description |
|-----------|------|-------------|
| `RS` | DataFrame (n × p) | Responsiveness scores for all units and predictors |
| `R2` | dict | R² for each predictor model |
| `wald` | dict | Wald test results (F-statistic, p-value, df) per predictor |
| `interaction` | dict | Interaction coefficient table (β, se, t, p) per predictor |
| `MFR` | Series | Mean Predictor Response — column means of RS |
| `MUR` | Series | Mean Unit Response — row means of RS |
| `TUR` | Series | Total Unit Response — row sums of RS |

### Key attributes (after `bootstrap`)

| Attribute | Type | Description |
|-----------|------|-------------|
| `RS_boot` | dict | Per-predictor DataFrames with bootstrap mean, CI, p-value, sign stability |
| `beta_boot` | dict | Per-predictor bootstrap draws of the interaction coefficients |

## Analytical workflow

The recommended reading order for results is:

1. **R²** — does the model fit the data?
2. **δ₀** (baseline slope) — what is the average effect of predictor j?
3. **Wald test** — the gate: are interaction terms jointly significant?
   If not, scores collapse to δ₀; stop here for that predictor.
4. **Interaction coefficients** — *which* other predictors modulate the
   slope? This is the mechanism behind heterogeneity.
5. **Bootstrap CIs** — which individual units have scores statistically
   different from zero?
6. **Score distributions / maps** — explore the spatial or cross-unit
   heterogeneity.

## Standardisation

RScore expects standardised input — this is **not** handled internally.
The user is responsible for standardising the DataFrame before calling
`fit()`. This is deliberate: the standardisation choice affects results
and should be an explicit, documented decision.

The companion module `rscore.statistics` provides utility functions to
help with this step:

```python
from rscore.statistics import test_standardization, standardize_data

# Compare Gaussian, Gamma, Pearson III, KDE fits
result = test_standardization(df["precipitation"].values, plot=True)

# Apply the recommended transform (Probability Integral Transform → z-scores)
output = standardize_data(df["precipitation"], result)
df["precipitation"] = output["z_scores"]
```

These are **standalone utilities**, not part of the RScore estimation
pipeline. They work independently and can be used for any standardisation
task.

## Terminology note

Cerulli (2017) uses "predictors" where standard statistical usage would say
"predictors" or "independent variables". This documentation uses traditional
terminology throughout. The API parameter `predictors=` (called `predictors=`
in Cerulli's Stata command) retains the concept under a more standard name.

## Documentation

Full methodological reference — model derivation, worked example,
interpretation guide, standardisation options, limitations, and known
scope boundaries — is in [`docs/methodology.md`](docs/methodology.md).

## License

MIT