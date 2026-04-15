# Methodology

## Contents

1. [The rationale](#the rationale)
2. [A worked example](#a-worked-example)
3. [The model](#the-model)
4. [Cross-section estimation](#cross-section-estimation)
5. [Panel estimation (fixed effects)](#panel-estimation-fixed-effects)
6. [Aggregate measures](#aggregate-measures)
7. [Bootstrap inference on unit scores](#bootstrap-inference-on-unit-scores)
8. [Wald test for interaction significance](#wald-test-for-interaction-significance)
9. [Variance decomposition (panel)](#variance-decomposition-panel)
10. [Standardisation choices](#standardisation-choices)
11. [Practical workflow](#practical-workflow)
12. [Limitations and scope](#limitations-and-scope)
13. [References](#references)

---

## The rationale

In ordinary regression, every unit shares the same slope — one number
summarises the effect of predictor $x_j$ on outcome $y$ for *everyone*.
But **what if each unit has its own slope due to compound effects from other variables?**

Instead of estimating a single coefficient $\beta_j$,
RScore lets $\beta_j$ vary across units as a linear function of each unit's
observed values on the *other* predictors. Concretely, the method multiplies the
focal predictor $x_j$ by other predictors $x_k$ selected to interact with it,
adding these products — called **interaction terms** — to the regression. The
estimated coefficients on those interaction terms, combined with each
unit's actual $x_k$ values, give every unit its own local slope.



```
Standard regression              RScore
──────────────────               ──────
One slope for all        →        One locally evaluated slope per unit
β_j                      →        RS_ij = β_j + context-dependent adjustments
```

### What it produces

For each (unit, predictor) pair, a **Responsiveness Score** (RS): the
slope of $y$ with respect to predictor $x_j$, evaluated at that
unit's specific covariate values. Collected into a matrix of dimensions
(n_observations × p_predictors), these scores reveal who responds
strongly, who weakly, and — via the interaction coefficients — *why*.

### What it does NOT do

RScore detects heterogeneity only insofar as it is *linearly mediated*
by the predictors included in the model. If a unit responds differently
because of an unobserved feature $Z$, RScore will not detect it. A
non-significant Wald test means the *included* predictors do not explain
slope variation — not that slope variation is absent.

---

## A worked example

Suppose we study how **winter temperature** ($x_1$) affects **crop yield**
($y$) across three provinces, controlling for **temperature at harvest** ($x_2$).
All variables are standardised (mean 0, sd 1).

**Standard regression** estimates a single slope:

$$y_i = \alpha + {\beta} \cdot x_{i1} + {\gamma}\cdot x_{i2}$$

where ${\beta}$ and $\gamma$ are the slope of$x_1$ and $x_2$, respectively, namely the direct effect of winter temperature and temperature at harvest on yield.

[//]: # (&#40;the latter  matters for the model fit but not for RS, because RS is the partial derivative of $y$ with respect to )

[//]: # (winter temperature only.)

Every province gets the same answer: "${\beta}$ more sd of winter temperature → +$ {\beta}$ sd
of yield." Flat, uniform, uninformative about local differences.

**RScore** adds the interaction $x_1 \times x_2$ to the regression and
estimates:


$$y_i = \alpha + {\beta_0} \cdot x_{i1}
      + \gamma \cdot x_{i2}
      + {\beta_1} \cdot (x_{i1} \cdot x_{i2})
      + \varepsilon_i$$

Now, supposing ${\beta_0}=0.40$ and ${\beta_1}=0.25$, the slope of winter temperature *for unit $i$* is:

$$RS_{i,\text{win_temp}} = \beta_0 + \beta_1 \cdot x_{i2}
                        = 0.40 + 0.25 \cdot x_{i,\text{temp}}$$

| Province | Temperature at harvest($x_{i2}$) | RS                              | Interpretation   |
|----------|----------------------------------|---------------------------------|------------------|
| A        | +1.2 (hot)                       | 0.40 + 0.25 × 1.2 = **0.70**    | Strong response  |
| B        | 0.0 (average)                    | 0.40 + 0.25 × 0.0 = **0.40**    | Average response |
| C        | −0.8 (cool)                      | 0.40 + 0.25 × (−0.8) = **0.20** | Weak response    |

Province A benefits more from warm winters due to a positive interaction with harvest temperature. 
Province C,being cooler, barely responds, despite the interaction exist and being positve. The *global* coefficients ($\beta_0 = 0.40$,
$\beta_1 = 0.25$) are the same for everyone — what differs is each
province's *temperature at harvest*, and that is what makes the slope local.

This procedure is repeated for each predictor — one regression per
predictor, each with its own set of interaction terms and its own R².

---

## The model

### Starting point: standard regression

$$y_i = \alpha + \beta_j \cdot x_{ij} + \sum_{k \neq j} \beta_k \cdot x_{ik} + \varepsilon_i$$

where:

- $y_i$ — outcome for unit $i$
- $x_{ij}$ — value of the focal predictor $j$ for unit $i$
- $\beta_j$ — slope of $x_j$: **one number, the same for all units**
- $x_{ik}$ — value of each other predictor $k$ for unit $i$
- $\varepsilon_i$ — error term

The limitation: $\beta_j$ cannot capture that different units may respond
to the same predictor in different ways.

### Heterogeneous-slope model

We allow the slope of $x_j$ to depend on the other predictors by adding
**interaction terms**: the product of the focal predictor $x_j$ with each
of the other predictors $x_k$.

$$y_i = \alpha
      + \beta_j \cdot x_{ij}
      + \sum_{k \neq j} \gamma_k \cdot x_{ik}
      + \sum_{k \neq j} \beta_k \cdot \underbrace{(x_{ij} \cdot x_{ik})}_{\text{interaction}}
      + \varepsilon_i$$

where:

- $\beta_j$ — **baseline slope** of $x_j$ on $y$, shared by all units
  (analogous to $\beta_j$ in standard regression)
- $\gamma_k$ — direct effect of each other predictor $x_k$ on $y$
- $\beta_k$ — **interaction coefficient**: how much predictor $x_k$
  modifies the slope of $x_j$. Estimated once on all data.
- $(x_{ij} \cdot x_{ik})$ — interaction term: a new variable built by
  multiplying the focal predictor by each other predictor

### The Responsiveness Score

The unit-specific slope of $x_j$ is obtained by taking the partial
derivative of $y$ with respect to $x_j$:

$$RS_{ij} = \frac{\partial y}{\partial x_j}\Bigg|_{\text{unit } i}
          = \beta_0 + \sum_{k \neq j} \beta_k \cdot x_{ik}$$

- $\beta_0$ and $\beta_k$ are **global** — estimated once, identical
  for every unit
- $x_{ik}$ is the **observed value** of predictor $k$ for unit $i$ —
  this is what makes $RS_{ij}$ unit-specific

A province with high $x_{ik}$ gets a different score than one with low
$x_{ik}$, even though they share the same estimated coefficients.

---

## Cross-section estimation

For cross-section data, one OLS regression is estimated per focal
predictor $j$:

$$y_i = \beta_j
      + \sum_{k \neq j} \gamma_k \cdot x_{ik}
      + \beta_0 \cdot x_{ij}
      + \sum_{k \neq j} \beta_k \cdot (x_{ij} \cdot x_{ik})
      + z_i' \zeta
      + \eta_i$$

where $z_i$ are optional controls and/or categorical dummies, and $\eta_i$
is heteroskedastic by construction (the interaction terms create non-constant
variance). Standard errors are computed with HC3 robust covariance.

After estimation:

$$RS_{ij} = \hat{\beta}_j + \sum_{k \neq j} \hat{\beta}_k \cdot x_{ik}$$

There are **p separate regressions** — one per focal predictor — each with
the same $y$ but a different interaction structure. Each regression has its
own R².

---

## Panel estimation (fixed effects)

When the data have a unit × time structure, unobserved time-invariant
heterogeneity can be absorbed via fixed effects:

$$y_{it} = \gamma_0
         + \sum_{k \neq j} \gamma_k \cdot x_{ikt}
         + \beta_0 \cdot x_{ijt}
         + \sum_{k \neq j} \beta_k \cdot (x_{ijt} \cdot x_{ikt})
         + \alpha_i + \eta_{it}$$

The within-transformation (demeaning by unit) eliminates $\alpha_i$.

### Order of operations matters

The interaction terms must be computed **before** demeaning:

1. Compute interaction terms on original values: $\text{interaction}_{jk} = x_{jt} \cdot x_{kt}$
2. Demean $y$, all predictors, controls, **and** interaction terms together

This is critical because:

$$\widetilde{x_j \cdot x_k} \neq \tilde{x}_j \cdot \tilde{x}_k$$

(where tilde denotes the demeaned variable). The difference equals the
product of unit means, which is nonzero in panel data. In Cerulli's
original Stata implementation this order is guaranteed by design
(`xtreg, fe` demeans everything together). In Python it must be enforced
explicitly — and the package does so.

### Scores use original values

After estimating $\beta$ on the demeaned data, the scores are computed on
the **original standardised values**, not the demeaned ones:

$$RS_{ijt} = \hat{\beta}_0 + \sum_{k \neq j} \hat{\beta}_k \cdot x_{ikt}$$

The demeaning is only a device to identify the $\beta$ coefficients without
bias from unit fixed effects. The scores measure each unit's responsiveness
in its absolute context.

### No constant in panel mode

The within-transformation removes all unit means, making a constant
collinear. The package omits it automatically when `mode="panel"`.

---

## Aggregate measures

From the (n × p) matrix of scores **RS**, three summary statistics are
defined:

**Mean Predictor Response (MPR)** — column means of RS:

$$MPR_j = \frac{1}{n} \sum_i RS_{ij}$$

*On average, how strongly does $y$ react to predictor $j$?*

**Mean Unit Response (MUR)** — row means of RS:

$$MUR_i = \frac{1}{p} \sum_j RS_{ij}$$

*On average across all predictors, how responsive is unit $i$?*

**Total Unit Response (TUR)** — row sums of RS:

$$TUR_i = \sum_j RS_{ij}$$

*What is the total cumulative sensitivity of unit $i$ across all
predictors?* Requires standardised variables to be meaningful (otherwise
it sums apples and oranges).

---

## Bootstrap inference on unit scores

### Motivation

The scores $RS_{ij}$ are deterministic functions of estimated coefficients
and observed covariates. To assess whether a given score is statistically
different from zero, we need confidence intervals.

### Procedure

1. Resample the data (iid bootstrap for cross-section, cluster bootstrap
   by unit for panel — following Cameron & Miller, 2015).
2. Re-estimate the regression on each bootstrap sample to get new
   $\hat{\beta}^{(b)}$ coefficients.
3. Compute scores on the **original** covariate values using
   $\hat{\beta}^{(b)}$: this propagates parametric uncertainty while
   holding the "context" of each unit fixed.
4. Repeat B times. The distribution of $RS_{ij}^{(b)}$ across bootstrap
   replications yields confidence intervals and p-values for each unit.

### Bootstrap output (per unit, per predictor)

| Column | Meaning |
|--------|---------|
| `mean` | Bootstrap mean of the score |
| `median` | Bootstrap median |
| `ci_lo_95`, `ci_hi_95` | 95% percentile confidence interval |
| `p_boot_two_sided` | 2 × min(P(RS ≤ 0), P(RS ≥ 0)) — analogous to a two-sided p-value |
| `sign_stability` | max(fraction positive, fraction negative) — how consistently the score keeps its sign |
| `mu_positive_share` | Fraction of bootstrap draws where RS > 0 |

### What the bootstrap does NOT capture

The bootstrap propagates uncertainty in $\beta$ (the interaction
coefficients) to the scores. It does **not** account for:

- Sampling variability of the covariates $x_{ik}$ themselves
- Model misspecification (the linearity of the interaction structure)
- Uncertainty in the choice of which variables are predictors vs controls

This is by design: we hold $x$ fixed and ask how precisely we can estimate
the slope *at that point*.

---

## Wald test for interaction significance

Before looking at individual scores, a joint hypothesis test determines
whether there is any detectable heterogeneity at all:

$$H_0: \beta_1 = \beta_2 = \ldots = \beta_{p-1} = 0
\quad \text{(all interaction coefficients are zero)}$$
$$H_1: \text{at least one } \beta_k \neq 0$$

If $H_0$ is not rejected, all scores collapse to $\beta_0$ plus noise —
there is no evidence of heterogeneous slopes, and unit-level scores carry
no reliable signal.

The package computes this as an F-test (with HC3 covariance) for each
predictor model.

### How to read the results

**Wald rejects + many significant individual scores** — genuine
heterogeneity. The score distribution is informative; proceed with
analysis.

**Wald rejects + few significant individual scores** — aggregate
heterogeneity exists but the sample is too small to localise it at the
unit level. This is not a contradiction: the joint test has more power
than n individual tests.

**Wald does not reject** — no evidence of heterogeneous slopes from the
included predictors. Unit scores are uninformative. A standard regression
with a single slope may be sufficient.

**Wald does not reject + visible spatial variation in RS maps** — the
observed variation reflects noise propagated from non-significant
$\beta_k$, not genuine heterogeneity. The map carries no interpretable
signal for that predictor.

---

## Variance decomposition (panel)

In panel data, each unit has multiple scores over time. The total variance
of scores can be split:

$$\text{Var}(RS_j) = \text{Var}_{\text{between}} + \text{Var}_{\text{within}}$$

where:

- **Var\_between** = variance of unit temporal means across units.
  Measures structural differences between units.
- **Var\_within** = mean of within-unit temporal variances.
  Measures how much each unit's responsiveness fluctuates over time.

| Dominance | Meaning | Implication |
|-----------|---------|-------------|
| Between dominates | Units are structurally different but individually stable over time | A cross-section on temporal means captures the main signal |
| Within dominates | Each unit changes over time; between-unit differences are smaller | The panel dimension is essential |

For climate data, within-dominance is typical: a province's drought
responsiveness varies with land use changes, irrigation policy, vegetation
cover. That temporal instability is often the signal of interest for
adaptation analysis.

The package provides this decomposition via `plot_decompose()`, which shows
both absolute variance and percentage composition for each predictor.

---

## Standardisation choices

Cerulli (2017) uses **global z-score standardisation**: subtract the
overall mean and divide by the overall standard deviation.

$$z_{it} = \frac{x_{it} - \bar{x}}{\sigma_x}$$

This makes scores comparable across predictors and units in absolute
scale — a prerequisite for computing TUR (which sums scores across
predictors).

An alternative for panel data is **within-unit standardisation**:

$$z_{it} = \frac{x_{it} - \bar{x}_i}{\sigma_{x,i}}$$

This measures sensitivity relative to each unit's own historical norm.
It isolates within-unit dynamics but sacrifices absolute cross-unit
comparability.

| Approach | Pros | Cons |
|----------|------|------|
| Global | Cross-unit comparability; TUR is meaningful | Mixes between- and within-unit variation |
| Within-unit | Isolates temporal dynamics | Loses absolute scale; TUR not directly comparable |

The package does **not** standardise internally — the user is responsible
for standardising the input DataFrame before calling `fit()`. This is
deliberate: standardisation is a modelling choice that should be explicit
and documented, not hidden inside a black box.

The companion module `rscore.statistics` provides utility functions
(`test_standardization`, `standardize_data`) to help choose and apply
the appropriate standardisation, including non-Gaussian alternatives via
the Probability Integral Transform (PIT). These are convenience tools,
not part of the RScore estimation pipeline.

---

## Practical workflow

```python
from rscore import RScore
from sklearn.preprocessing import StandardScaler
import pandas as pd

# 1. Load and clean
df = pd.read_csv("data.csv").dropna()

# 2. Standardise continuous variables
#    (global z-score shown here; within-unit is also valid — see above)
scaler = StandardScaler()
df[continuous_cols] = scaler.fit_transform(df[continuous_cols])

# 3. Build categorical dummies (if any)
X_cat = pd.get_dummies(df[cat_cols], drop_first=True).astype(float)

# 4. Fit
model = RScore(mode="panel", unit_col="province")
model.fit(df, y_col="drought",
          predictors=["precip", "temp", "ndvi"],
          controls=["elevation"],
          X_cat=X_cat)

# 5. Check Wald test FIRST — this is the gate
model.summary()
# If Wald rejects → proceed.  Otherwise → reconsider specification.

# 6. Bootstrap inference
model.bootstrap(B=500, seed=42)

# 7. Explore distributions and decomposition
model.plot_rs(sig_only=True)              # significant scores only
fig, ax, table = model.plot_decompose()   # between vs within variance

# 8. Time series for specific units
model.plot_timeseries(predictor="precip",
                      units=["TE", "AQ", "RM"],
                      time_col="year")
```

### Decision points

**Which variables should be predictors vs controls?**

Predictors are variables whose *heterogeneous interaction with each other*
you want to study. Controls enter the model linearly (no interaction
terms) — they affect $y$ but their slope heterogeneity is not of interest.
This choice should be theory-driven, not p-value-driven.

**When should I remove a non-significant interaction term?**

If the Wald test rejects but individual $\beta_k$ are non-significant,
do not automatically drop them. Moving a variable from predictors to
controls changes the entire interaction structure. Justify any such
change with domain knowledge, not statistical fishing.

**What if R² is reasonable but most individual scores are noisy?**

This is expected, not a contradiction. R² measures aggregate model fit;
individual scores partition that fit across units with inevitably less
precision. The bootstrap p-values tell you which units have reliably
heterogeneous responses.

---

## Limitations and scope

**Linearity**: the interaction structure is additive and linear. If the
true heterogeneity is nonlinear, RS captures only the linear projection.

**No causal identification**: RS measures association, not causation.
The interaction terms have no causal interpretation without further
identification assumptions.

**Curse of dimensionality**: with p predictors, each model has p−1
interaction terms. Many predictors lead to multicollinearity (the package
warns when the condition number exceeds 30).

**Standardisation sensitivity**: results depend on the standardisation
choice, which should be documented and justified.

**Bootstrap scope**: the bootstrap propagates parametric uncertainty only.
It does not protect against model misspecification or covariate measurement
error.

**Interaction-mediated heterogeneity only**: RScore detects heterogeneity
only if it is linearly mediated by the included predictors. A
non-significant Wald test does not mean the effect of $x_j$ is everywhere
the same — it means the *included* predictors $x_k$ do not systematically
modulate it. Heterogeneity driven by omitted covariates (elevation, land
use, institutional context, etc.) remains invisible to the Wald test.

---

## References

1. Cerulli, G. (2017). "Estimating responsiveness scores using rscore".
   *The Stata Journal*, 17(2), 422–441.
   [DOI: 10.1177/1536867X1701700208](https://doi.org/10.1177/1536867X1701700208)

2. Cameron, A.C. & Miller, D.L. (2015). "A Practitioner's Guide to
   Cluster-Robust Inference". *Journal of Human Resources*, 50(2), 317–372.

3. Wooldridge, J.M. (2010). *Econometric Analysis of Cross Section and
   Panel Data*. 2nd ed. MIT Press.