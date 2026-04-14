"""
rscore.core
===========
Object-oriented implementation of the Responsiveness Score methodology
(Cerulli, 2017, The Stata Journal).

This module provides the :class:`RScore` class, which estimates unit-specific
responsiveness scores via iterated random-coefficient regression, with
optional cluster-aware bootstrap inference and dedicated plotting methods.

References
----------
- Cerulli, G. (2017). "Estimating responsiveness scores using rscore".
  The Stata Journal, 17(2), 422-441.
- Cameron, A.C. & Miller, D.L. (2015). "A Practitioner's Guide to
  Cluster-Robust Inference". J. Human Resources, 50(2), 317-372.
"""

from __future__ import annotations

import warnings
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------
_CAT_PREFIX = "_cat__"
_interaction_PREFIX = "interaction_"
_COND_NUM_THRESHOLD = 30


# ===================================================================
# RScore class
# ===================================================================

class RScore:
    """Responsiveness Score estimator for cross-section and panel data.

    Parameters
    ----------
    mode : {"cross_section", "panel"}
        Data structure.
    unit_col : str or None
        Column identifying panel units.  Required when ``mode="panel"``.

    Examples
    --------
    model = RScore(mode="panel", unit_col="province")
    model.fit(df, y_col="y", predictors=["x1", "x2"], controls=["z1"])
    model.bootstrap(B=500, seed=42)
    model.plot_between(predictors=["x1", "x2"])
    """

    # ------------------------------------------------------------------ init
    def __init__(self, mode: str = "cross_section",
                 unit_col: Optional[str] = None):
        if mode not in ("cross_section", "panel"):
            raise ValueError("mode must be 'cross_section' or 'panel'")
        if mode == "panel" and unit_col is None:
            raise ValueError("unit_col is required for mode='panel'")

        self.mode = mode
        self.unit_col = unit_col

        # Will be populated by fit()
        self._df_aug: Optional[pd.DataFrame] = None
        self._y_col: Optional[str] = None
        self._predictors: Optional[List[str]] = None
        self._controls: List[str] = []
        self._cat_cols: List[str] = []
        self._add_constant: bool = True

        # Results (populated by fit / bootstrap)
        self.RS: Optional[pd.DataFrame] = None
        self.R2: Optional[Dict[str, float]] = None
        self.interaction: Optional[Dict[str, pd.DataFrame]] = None
        self.wald: Optional[Dict[str, dict]] = None
        self.models: Optional[Dict[str, sm.OLS]] = None
        self.MFR: Optional[pd.Series] = None
        self.MUR: Optional[pd.Series] = None
        self.TUR: Optional[pd.Series] = None

        self.RS_boot: Optional[Dict[str, pd.DataFrame]] = None
        self.beta_boot: Optional[Dict[str, pd.DataFrame]] = None

        self._fitted = False
        self._bootstrapped = False

    # ================================================================
    # PUBLIC API — fit
    # ================================================================

    def fit(self, df: pd.DataFrame, y_col: str, predictors: List[str],
            controls: Optional[List[str]] = None,
            X_cat: Optional[pd.DataFrame] = None) -> "RScore":
        """Estimate responsiveness scores for every predictor.

        Parameters
        ----------
        df : DataFrame
            Standardised continuous variables.  Must **not** contain the
            dummy columns passed via *X_cat*.
        y_col : str
            Name of the response variable.
        predictors : list of str
            Variables whose heterogeneous slopes are of interest.
        controls : list of str, optional
            Variables entering the model without interaction terms.
        X_cat : DataFrame, optional
            Pre-built dummies (e.g. from ``pd.get_dummies(…, drop_first=True)``).

        Returns
        -------
        self
            Fitted estimator (for method chaining).

        Notes
        -----
        When *y_col* is standardised (z-score), the resulting scores are in
        units of standard deviations of y.  To recover physical units,
        back-transform: ``RS_physical = RS_zscore * sd_y``.
        """
        controls = controls or []
        self._y_col = y_col
        self._predictors = list(predictors)
        self._controls = list(controls)

        # Embed categorical dummies into df (drop_first_true)
        self._df_aug, self._cat_cols = self._embed_cat(df, X_cat)
        # Add constant for OLS
        self._add_constant = (self.mode == "cross_section")

        RS = pd.DataFrame(index=df.index)
        R2: Dict[str, float] = {}
        interaction_dict: Dict[str, pd.DataFrame] = {}
        wald_dict: Dict[str, dict] = {}
        models_dict: Dict[str, sm.OLS] = {}

        # ===============================================================
        # Fit OLS, test the significance beta_int and compute RSCORE:
        for j in self._predictors:
            model, tab, rs_ols, wald_result = self._ols_interaction(j)
            RS[j] = rs_ols
            R2[j] = model.rsquared
            interaction_dict[j] = tab
            wald_dict[j] = wald_result
            models_dict[j] = model
        # ===============================================================

        self.RS = RS
        self.R2 = R2
        self.interaction = interaction_dict
        self.wald = wald_dict
        self.models = models_dict
        self.MFR = RS.mean(axis=0)
        self.MUR = RS.mean(axis=1)
        self.TUR = RS.sum(axis=1)
        self._fitted = True
        return self

    # ================================================================
    # PUBLIC API — bootstrap
    # ================================================================

    def bootstrap(self, B: int = 500, seed: int = 42,
                  alpha: float = 0.05) -> "RScore":
        """
        Run bootstrap inference on unit-level scores.

        For cross-section data, rows are resampled (iid bootstrap).
        For panel data, entire units are resampled (cluster bootstrap,
        Cameron & Miller 2015).

        The bootstrap propagates **parametric uncertainty** from the
        delta coefficients through to the individual scores b_ij,
        evaluated on the *original* (full-sample) covariate values.
        This means the resulting confidence intervals reflect uncertainty
        in the estimated slopes, not sampling variability in the
        covariates themselves.

        Parameters
        ----------
        B : int
            Number of bootstrap replications.
        seed : int
            Base RNG seed.  Each predictor j uses ``seed + j_index`` to
            ensure independent streams across predictors when parallelising (if any)
        alpha : float
            Significance level for confidence intervals.

        Returns
        -------
        self
        """
        self._check_fitted("bootstrap")

        resample_fn = (self._resample_rows if self.mode == "cross_section"
                       else self._resample_clusters)

        RS_boot: Dict[str, pd.DataFrame] = {}
        beta_boot: Dict[str, pd.DataFrame] = {}

        for j_idx, j in enumerate(self._predictors):
            # FIX #2: per-predictor seed avoids identical RNG streams
            j_seed = seed + j_idx
            b_sum, rs_sum, _, _ = self._bootstrap_predictor(
                j, resample_fn, B=B, seed=j_seed, alpha=alpha
            )
            RS_boot[j] = rs_sum
            beta_boot[j] = b_sum

        self.RS_boot = RS_boot
        self.beta_boot = beta_boot
        self._bootstrapped = True
        return self

    # ================================================================
    # PUBLIC API — plotting methods
    # ================================================================

    def plot_rs(self,
                predictors: Optional[List[str]] = None,
                group: Optional[np.ndarray] = None,
                group_name: str = "group",
                sig_only: bool = False,
                alpha: float = 0.05,
                wald_gate: bool = False,
                figsize: Tuple = (10, 5)) -> Tuple[plt.Figure, plt.Axes]:
        """KDE of responsiveness scores, optionally split by a user-supplied group.

        Works for both cross-section and panel data.

        Parameters
        ----------
        predictors : list of str, optional
            Subset of predictors to plot.  Defaults to all.
        group : array-like of shape (n_obs,), optional
            Grouping vector (e.g. ``df[unit_col].values`` or ``df[time_col].values``).
            Must have the same length as ``self.RS``.  If None, all observations
            are pooled into a single KDE per predictor.
        group_name : str
            Label used in the legend title (e.g. "country", "year").
        sig_only : bool
            If True, non-significant scores (``p_boot >= alpha``) are set to NaN
            before plotting.  Requires bootstrap results.
        alpha : float
            Significance threshold.
        wald_gate : bool
            If True, only predictors that passed the Wald test (``p < alpha``) are
            plotted.  predictors that did not pass are silently dropped with a warning.
        figsize : tuple
            Figure size.

        Returns
        -------
        fig, ax
        """
        self._check_fitted("plot_rs")
        predictors = list(predictors or self._predictors)

        # --- Wald gate -------------------------------------------------------
        if wald_gate:
            predictors = [j for j in predictors if self.wald[j]["p_value"] < alpha]
            if not predictors:
                print("No predictors passed the Wald gate — nothing to plot.")
                return None, None
        else:
            failed = [j for j in predictors if self.wald[j]["p_value"] >= alpha]
            if failed:
                print(f"WARNING: {failed} did not pass the Wald gate — "
                      f"scores shown may reflect noise, not genuine heterogeneity. "
                      f"Use wald_gate=True to suppress them.")

        # --- Validate group --------------------------------------------------
        if group is not None:
            group = np.asarray(group)
            if len(group) != len(self.RS):
                raise ValueError(
                    f"group length ({len(group)}) must match RS length ({len(self.RS)})."
                )
            unique_groups = np.unique(group)
        else:
            unique_groups = None

        sig_tag = f" (p<{alpha})" if sig_only else ""

        # --- One subplot per predictor (only if grouped) ------------------------
        if unique_groups is not None:
            n = len(predictors)
            fig, axes = plt.subplots(1, n, figsize=(figsize[0] * n / max(n, 1),
                                                    figsize[1]), sharey=False)
            axes = [axes] if n == 1 else list(axes)
        else:
            fig, ax_single = plt.subplots(figsize=figsize)
            axes = [ax_single] * len(predictors)

        for ax, j in zip(axes, predictors):
            vals_full = self._get_score_series(j, sig_only, alpha)

            if unique_groups is None:
                xs, ys = self._kde(vals_full[np.isfinite(vals_full)])
                if xs is not None:
                    ax.plot(xs, ys, linewidth=2,
                            label=self._sig_label(j, sig_only, alpha))
            else:
                for g in unique_groups:
                    mask = group == g
                    vals_g = vals_full[mask]
                    xs, ys = self._kde(vals_g[np.isfinite(vals_g)])
                    if xs is not None:
                        ax.plot(xs, ys, linewidth=2, label=str(g))

            ax.axvline(0, color="dimgrey", linestyle=":", linewidth=1)
            if unique_groups is not None:
                ax.set_title(self._sig_label(j, sig_only, alpha))
            ax.set_xlabel("Responsiveness Score")
            ax.set_ylabel("Density")
            legend_title = group_name if unique_groups is not None else "predictor"
            ax.legend(title=legend_title, fontsize=8)

        title_tag = f" by {group_name}" if unique_groups is not None else ""
        fig.suptitle(f"RS distribution{title_tag}{sig_tag}", fontsize=13)
        plt.tight_layout()
        plt.show(block=False)
        return fig, axes if unique_groups is not None else ax_single

    def plot_decompose(self, predictors: Optional[List[str]] = None,
                       group_col: Optional[str] = None,
                       sig_only: bool = False, alpha: float = 0.05,
                       wald_gate: bool = False,
                       figsize: Tuple = (11, 5)) -> Tuple:
        """Variance decomposition: Var(RS) = Var_between + Var_within.

        Parameters
        ----------
        predictors : list of str, optional
            Subset of predictors.
        group_col : str, optional
            Column for group-wise faceting.
        sig_only : bool
            Use only significant scores.
        alpha : float
            Significance threshold.
        figsize : tuple
            Figure size.

        Returns
        -------
        fig, axes, summary : DataFrame with decomposition values.
        """
        self._check_fitted("plot_decompose")
        self._check_panel("plot_decompose")
        predictors = predictors or self._predictors

        if wald_gate:
            predictors = [j for j in predictors if self.wald[j]["p_value"] < alpha]
            if not predictors:
                print("No predictors passed the Wald gate — nothing to plot.")
                return None, None, None
        else:
            failed = [j for j in predictors if self.wald[j]["p_value"] >= alpha]
            if failed:
                print(f"WARNING: {failed} did not pass the Wald gate — "
                      f"scores shown may reflect noise, not genuine heterogeneity. "
                      f"Use wald_gate=True to suppress them.")

        RS_vals = self._filtered_rs(predictors, sig_only, alpha)
        data = self._merge_meta(RS_vals, predictors, group_col)
        sig_tag = f" (p<{alpha})" if sig_only else ""

        def _xtick(j):
            if not sig_only:
                return j
            n_sig, n_tot, pct = self._sig_counts_single(j, alpha)
            return f"{j}\n{n_sig}/{n_tot}\n({pct:.0f}%)"

        xticks = [_xtick(j) for j in predictors]

        def _decompose(sub):
            vb, vw = [], []
            for j in predictors:
                m = sub.groupby(self.unit_col)[j].mean()
                vb.append(m.var() if len(m) > 1 else 0)
                vw.append(sub.groupby(self.unit_col)[j].var().mean())
            return np.nan_to_num(vb), np.nan_to_num(vw)

        if group_col is None:
            vb, vw = _decompose(data)
            vt = vb + vw
            pb = np.where(vt > 0, vb / vt * 100, 0)
            pw = np.where(vt > 0, vw / vt * 100, 0)
            x = np.arange(len(predictors))

            fig, axes = plt.subplots(1, 2, figsize=figsize)
            for ax, (b, w, ylabel, ttl) in zip(axes, [
                (vb, vw, "Absolute variance", "Absolute values"),
                (pb, pw, "% total variance", "Composition %"),
            ]):
                ax.bar(x, b, 0.5, label="Between", color="steelblue")
                ax.bar(x, w, 0.5, bottom=b, label="Within", color="darkorange")
                ax.set_xticks(x)
                ax.set_xticklabels(xticks, fontsize=8)
                ax.set_ylabel(ylabel)
                ax.set_title(ttl)
                if ttl == "Composition %":
                    ax.set_ylim(0, 110)
                    for xi, p_w in enumerate(pw):
                        ax.text(xi, 101, f"W:{p_w:.0f}%", ha="center",
                                va="bottom", fontsize=7, color="dimgrey")
                ax.legend()
            fig.suptitle(f"RS variance decomposition{sig_tag}", fontsize=13)
            plt.tight_layout()
            plt.show(block=False)

            summary = pd.DataFrame({
                "predictor": predictors,
                "var_between": np.round(vb, 4),
                "var_within": np.round(vw, 4),
                "var_total": np.round(vt, 4),
                "pct_between": np.round(pb, 1),
                "pct_within": np.round(pw, 1),
            })
            if sig_only:
                for j in predictors:
                    n_sig, n_tot, _ = self._sig_counts_single(j, alpha)
                    summary.loc[summary["predictor"] == j, "n_sig"] = n_sig
                    summary.loc[summary["predictor"] == j, "n_tot"] = n_tot

            return fig, axes, summary

        else:
            categories = data[group_col].dropna().unique()
            n = len(categories)
            fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
            axes = [axes] if n == 1 else list(axes)
            rows = []
            for ax, cat in zip(axes, categories):
                vb, vw = _decompose(data[data[group_col] == cat])
                vt = vb + vw
                pb = np.where(vt > 0, vb / vt * 100, 0)
                pw = np.where(vt > 0, vw / vt * 100, 0)
                x = np.arange(len(predictors))
                ax.bar(x, pb, 0.5, label="Between %", color="steelblue")
                ax.bar(x, pw, 0.5, bottom=pb, label="Within %",
                       color="darkorange")
                ax.set_xticks(x)
                ax.set_xticklabels(xticks, fontsize=7)
                ax.set_title(str(cat))
                ax.set_ylim(0, 110)
                for xi, p_w in enumerate(pw):
                    ax.text(xi, 101, f"W:{p_w:.0f}%", ha="center",
                            va="bottom", fontsize=7, color="dimgrey")
                if ax is axes[0]:
                    ax.set_ylabel("% variance")
                    ax.legend()
                for fj, p_b, p_w in zip(predictors, pb, pw):
                    rows.append({"category": cat, "predictor": fj,
                                 "pct_between": round(p_b, 1),
                                 "pct_within": round(p_w, 1)})
            fig.suptitle(
                f"RS variance decomposition by {group_col}{sig_tag}",
                fontsize=13,
            )
            plt.tight_layout()
            plt.show(block=False)
            summary = pd.DataFrame(rows)
            return fig, axes, summary

    def plot_timeseries(self, predictor: str, units: List[str],
                        time_col: str,
                        sig_only: bool = False, alpha: float = 0.05,
                        figsize: Tuple = (10, 5)) -> Tuple[plt.Figure, plt.Axes]:
        """Time-series of RS for selected units.

        Parameters
        ----------
        predictor : str
            Single predictor to plot.
        units : list of str
            Unit identifiers to include.
        time_col : str
            Column with the time dimension.
        sig_only : bool
            Mask non-significant observations to NaN.
        alpha : float
            Significance threshold.
        figsize : tuple
            Figure size.

        Returns
        -------
        fig, ax
        """
        self._check_fitted("plot_timeseries")
        self._check_panel("plot_timeseries")

        df_ts = self._df_aug[[self.unit_col, time_col]].copy()
        vals = self._get_score_series(predictor, sig_only, alpha)
        df_ts["RS"] = vals

        fig, ax = plt.subplots(figsize=figsize)
        for u in units:
            d = df_ts[df_ts[self.unit_col] == u].sort_values(time_col)
            ax.plot(d[time_col], d["RS"], marker="o", linewidth=2, label=u)

        sig_tag = f" (p<{alpha})" if sig_only else ""
        ax.set_title(f"RS time series — {predictor}{sig_tag}")
        ax.set_xlabel(time_col)
        ax.set_ylabel(f"RS [{predictor}]")
        ax.legend(ncols=min(len(units), 5))
        plt.tight_layout()
        plt.show(block=False)
        return fig, ax

    # ================================================================
    # PUBLIC API — summary
    # ================================================================

    def summary(self) -> str:
        """Print a concise summary of the fitted model."""
        self._check_fitted("summary")
        lines = ["=" * 60, "RScore Summary", "=" * 60]
        lines.append(f"Mode: {self.mode}")
        if self.mode == "panel":
            n_units = self._df_aug[self.unit_col].nunique()
            lines.append(f"Units: {n_units}  |  Obs: {len(self._df_aug)}")
        else:
            lines.append(f"Obs: {len(self._df_aug)}")

        lines.append(f"\nR-squared per predictor:")
        for j in self._predictors:
            lines.append(f"  {j:>20s}  R2 = {self.R2[j]:.4f}")

        lines.append(f"\nWald test H0: all interaction coefficients = 0:")
        for j in self._predictors:
            w = self.wald[j]
            tag = "***" if w["p_value"] < 0.01 else (
                  "**" if w["p_value"] < 0.05 else (
                  "*" if w["p_value"] < 0.10 else ""))
            lines.append(
                f"  {j:>20s}  F = {w['F_stat']:.2f}  "
                f"p = {w['p_value']:.4f} {tag}"
            )

        lines.append(f"\nMean predictor Responsiveness (MFR):")
        for j in self._predictors:
            lines.append(f"  {j:>20s}  {self.MFR[j]:.4f}")

        lines.append(f"\nSignificant interaction terms (p < 0.05):")
        for j in self._predictors:
            tab = self.interaction[j]
            n_sig = (tab["pval"] < 0.05).sum()
            lines.append(f"  {j:>20s}  {n_sig}/{len(tab)}")

        if self._bootstrapped:
            lines.append(f"\nBootstrap: computed")
        else:
            lines.append(f"\nBootstrap: not yet computed (call .bootstrap())")

        out = "\n".join(lines)
        print(out)
        return out

    # ================================================================
    # PRIVATE — OLS estimation
    # ================================================================

    def _ols_interaction(self, j: str):
        """
        Run OLS for predictor j with interaction terms and Wald test.
        Returns (model, interaction_table, rs_ols, wald_result).
        """
        df_X, other, int_cols = self._prepare_fit_df(j)
        # build X as df_X minus j and y
        X, y = self._assemble_X(df_X, j, other, int_cols)

        # FIX #4: condition number warning
        cond = np.linalg.cond(X.values.astype(float))
        if cond > _COND_NUM_THRESHOLD:
            warnings.warn(
                f"[{j}] Design matrix condition number = {cond:.0f} "
                f"(>{_COND_NUM_THRESHOLD}). Potential multicollinearity "
                f"among interaction terms.",
                stacklevel=2,
            )

        model = sm.OLS(y.to_numpy(), X).fit(cov_type="HC3")

        # interaction significance table
        tab = pd.DataFrame({
            "beta": model.params[int_cols],
            "se": model.bse[int_cols],
            "t": model.tvalues[int_cols],
            "pval": model.pvalues[int_cols],
        })
        ci = model.conf_int().loc[int_cols]
        ci.columns = ["ci_lo_95", "ci_hi_95"]
        tab = tab.join(ci).sort_values("pval")
        tab.index = [c.replace(_interaction_PREFIX, "") for c in tab.index]

        # FIX #3: Wald test — H0: all interaction coefficients = 0
        wald_result = self._wald_test(model, int_cols)
        p = wald_result["p_value"]
        verdict = ("OK — heterogeneity detected, interaction terms jointly significant, scores are interpretable"
                   if p < 0.05 else
                   "WARNING — interaction terms jointly insignificant, scores unreliable")
        print(f"[Wald | {j}]  F={wald_result['F_stat']:.3f}  p={p:.4f}  →  {verdict}")

        # Scores on original (non-demeaned) data
        rs_ols = self._compute_rs(model.params, j)
        return model, tab, rs_ols, wald_result

    def _wald_test(self, model, int_cols):
        """Joint significance test:
            H0: all interaction coefficients coefficients are zero.

            The Wald test is the global gate to run BEFORE interpreting individual
            unit scores b_ij.  If H0 is not rejected, the interaction terms are
            jointly insignificant: there is no evidence of slope heterogeneity and
            unit-level scores carry no reliable signal.  In that case, stop here —
            do not proceed to individual score analysis.
            """

        if not int_cols:
            return {"F_stat": np.nan, "p_value": np.nan, "df": (0, 0)}

        from scipy.stats import f as f_dist

        param_names = list(model.params.index)
        k = len(int_cols)

        R = np.zeros((k, len(param_names)))
        for i, col in enumerate(int_cols):
            R[i, param_names.index(col)] = 1.0

        Rb = R @ model.params.values
        RVR = R @ model.cov_params().values @ R.T
        chi2 = Rb @ np.linalg.solve(RVR, Rb)

        f_stat = chi2 / k
        df_resid = int(model.df_resid)
        p_val = float(f_dist.sf(f_stat, k, df_resid))

        return {"F_stat": float(f_stat), "p_value": p_val, "df": (k, df_resid)}

    def _compute_rs(self, coeff: pd.Series, j: str) -> np.ndarray:
        """
        Compute b_ij = delta_0 + x_{i,-j} @ interaction coefficients.

        IMPORTANT: always evaluated on ``self._df_aug`` (the original
        standardised values, NOT the within-transformed data).  The
        demeaning is used only to identify the delta coefficients without
        bias from unit fixed effects.  The scores measure each unit's
        responsiveness in its absolute context.
        """
        other = [v for v in self._predictors if v != j]
        b0 = coeff["Xj"]
        b_int = coeff[[f"{_interaction_PREFIX}{v}" for v in other]].to_numpy()
        return b0 + self._df_aug[other].astype(float).to_numpy() @ b_int

    # ================================================================
    # PRIVATE — data preparation
    # ================================================================

    @staticmethod
    def _embed_cat(df: pd.DataFrame,
                   X_cat: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, List[str]]:
        """Embed dummy columns into df with a unique prefix.
        Return augmented DataFrame with dummy columns"""
        if X_cat is None:
            return df.reset_index(drop=True).copy(), []
        cat_cols = [f"{_CAT_PREFIX}{c}" for c in X_cat.columns]
        X_cat_r = X_cat.rename(
            columns={c: f"{_CAT_PREFIX}{c}" for c in X_cat.columns}
        )
        df_aug = pd.concat(
            [df.reset_index(drop=True), X_cat_r.reset_index(drop=True)],
            axis=1,
        )
        return df_aug, cat_cols

    def _prepare_fit_df(self, j: str):
        """Build the OLS-ready DataFrame for predictor j.

        Steps:
          1. Compute interaction terms on *original* values (always).
          2. If panel: within-transform y, predictors, controls, AND interaction
             together (the critical fix — demeaning the product, not
             multiplying the demeaned).
        """
        df = self._df_aug.reset_index(drop=True).copy()
        other = [v for v in self._predictors if v != j]

        # Step 1: interaction on original values
        int_cols = []
        for v in other:
            col = f"{_interaction_PREFIX}{v}"
            df[col] = df[j].astype(float).values * df[v].astype(float).values
            int_cols.append(col)

        # Step 2: within-transform (panel only)
        if self.mode == "panel":
            vars_to_demean = (
                [self._y_col] + self._predictors + self._controls + int_cols
            )
            df = self._within_transform(df, vars_to_demean)

        return df, other, int_cols

    def _assemble_X(self, df_X: pd.DataFrame, j: str,
                    other: List[str], int_cols: List[str]):
        """Assemble design matrix from prepared DataFrame:
        remove y and Xj from df_X """
        df = df_X.reset_index(drop=True)
        main_cols = other + self._controls + self._cat_cols
        X_main = df[main_cols].astype(float)
        Xj = df[j].astype(float).rename("Xj")
        X_int = df[int_cols].astype(float)

        X = pd.concat([X_main, Xj, X_int], axis=1)
        if self._add_constant:
            X = sm.add_constant(X)
        y = df[self._y_col].astype(float)
        return X, y

    def _within_transform(self, df: pd.DataFrame,
                          vars_list: List[str]) -> pd.DataFrame:
        """
        Within-unit de-meaning for fixed-effects estimation."""

        df_w = df.copy()
        means = df.groupby(self.unit_col)[vars_list].transform("mean")
        df_w[vars_list] = df[vars_list].values - means.values
        return df_w

    # ================================================================
    # PRIVATE — bootstrap
    # ================================================================

    def _bootstrap_predictor(self, j: str, resample_fn, B: int,
                          seed: int, alpha: float):
        """Bootstrap for a single predictor j."""
        rng = np.random.default_rng(seed)
        n_obs = len(self._df_aug)
        other = [v for v in self._predictors if v != j]
        int_cols = [f"{_interaction_PREFIX}{v}" for v in other]
        cols_keep = ["Xj"] + int_cols

        coeff_draws = np.full((B, len(cols_keep)), np.nan)
        rs_draws = np.full((B, n_obs), np.nan)

        for b in range(B):
            df_b = resample_fn(self._df_aug, rng)

            df_X_b, _, int_cols_b = self._prepare_fit_df_on(df_b, j)
            X_b, y_b = self._assemble_X(df_X_b, j, other, int_cols_b)

            try:
                model_b = sm.OLS(y_b.to_numpy(), X_b).fit()
            except Exception:
                continue

            coeff_draws[b, :] = model_b.params[cols_keep].to_numpy()
            # Propagate to original data (parametric uncertainty only)
            rs_draws[b, :] = self._compute_rs(model_b.params, j)

        n_failed = np.isnan(coeff_draws[:, 0]).sum()
        if n_failed > 0:
            warnings.warn(
                f"[{j}] {n_failed}/{B} bootstrap iterations failed "
                f"({100 * n_failed / B:.0f}%) — results based on {B - n_failed} draws.",
                stacklevel=2,
            )

        beta_df = pd.DataFrame(coeff_draws, columns=cols_keep)
        rs_df = pd.DataFrame(rs_draws, columns=np.arange(n_obs))

        return (
            self._build_summary(beta_df, alpha),
            self._build_summary(rs_df, alpha),
            beta_df,
            rs_df,
        )

    def _prepare_fit_df_on(self, df_b: pd.DataFrame, j: str):
        """
        Prepare OLS DataFrame on a bootstrap sample.

        Same logic as _prepare_fit_df but operates on the given df_b
        rather than self._df_aug.
        """
        df = df_b.reset_index(drop=True).copy()
        other = [v for v in self._predictors if v != j]

        int_cols = []
        for v in other:
            col = f"{_interaction_PREFIX}{v}"
            df[col] = df[j].astype(float).values * df[v].astype(float).values
            int_cols.append(col)

        if self.mode == "panel":
            vars_to_demean = (
                [self._y_col] + self._predictors + self._controls + int_cols
            )
            df = self._within_transform(df, vars_to_demean)

        return df, other, int_cols

    def _resample_rows(self, df_aug: pd.DataFrame,
                       rng: np.random.Generator) -> pd.DataFrame:
        """IID row bootstrap for cross-section data."""
        idx = rng.integers(0, len(df_aug), size=len(df_aug))
        return df_aug.iloc[idx].reset_index(drop=True)

    def _resample_clusters(self, df_aug: pd.DataFrame,
                           rng: np.random.Generator) -> pd.DataFrame:
        """Cluster bootstrap for panel data (Cameron & Miller 2015).

        FIX #4: uses merge instead of concat-in-loop for efficiency.
        Entire units are resampled with replacement.  The within-transform
        will be recomputed on the bootstrap sample inside _prepare_fit_df_on.
        """
        units = df_aug[self.unit_col].unique()
        sampled = rng.choice(units, size=len(units), replace=True)

        # Efficient merge: build a mapping DataFrame and merge
        boot_map = pd.DataFrame({
            self.unit_col: sampled,
            "_boot_id": np.arange(len(sampled)),
        })
        df_b = boot_map.merge(df_aug, on=self.unit_col, how="left")
        df_b = df_b.drop(columns="_boot_id").reset_index(drop=True)
        return df_b

    @staticmethod
    def _build_summary(draws_df: pd.DataFrame,
                       alpha: float = 0.05) -> pd.DataFrame:
        """Compute bootstrap summary statistics."""
        lo, hi = alpha / 2, 1 - alpha / 2
        ci_label = int((1 - alpha) * 100)
        summary = pd.DataFrame({
            "mean": draws_df.mean(),
            "median": draws_df.median(),
            f"ci_lo_{ci_label}": draws_df.quantile(lo),
            f"ci_hi_{ci_label}": draws_df.quantile(hi),
            "mu_positive_share": (draws_df > 0).mean(),
            "mu_negative_share": (draws_df < 0).mean(),
        })
        p_left = (draws_df <= 0).mean()
        p_right = (draws_df >= 0).mean()
        summary["p_boot_two_sided"] = 2 * np.minimum(p_left, p_right)
        summary["sign_stability"] = summary[
            ["mu_positive_share", "mu_negative_share"]
        ].max(axis=1)
        return summary

    # ================================================================
    # PRIVATE — plotting helpers
    # ================================================================

    @staticmethod
    def _kde(values, n=300):
        """Gaussian KDE, returning (xs, ys) or (None, None)."""
        v = values[np.isfinite(values)]
        if len(v) < 5 or v.std() == 0:
            return None, None
        lo, hi = np.percentile(v, [1, 99])
        xs = np.linspace(lo, hi, n)
        return xs, gaussian_kde(v)(xs)

    def _merge_meta(self, RS_vals: pd.DataFrame,
                    predictors: List[str],
                    group_col: Optional[str] = None) -> pd.DataFrame:
        """Merge RS values with unit (and optionally group) metadata."""
        cols = [self.unit_col]
        if group_col is not None:
            cols.append(group_col)
        return pd.concat([
            self._df_aug[cols].reset_index(drop=True),
            RS_vals[predictors].reset_index(drop=True),
        ], axis=1)

    def _filtered_rs(self, predictors: List[str],
                     sig_only: bool, alpha: float) -> pd.DataFrame:
        """Return RS values, NaN-ing non-significant scores if requested."""
        RS_filt = self.RS[predictors].copy().reset_index(drop=True)
        if sig_only:
            self._check_bootstrapped("sig_only filtering")
            for j in predictors:
                mask = self.RS_boot[j]["p_boot_two_sided"].values >= alpha
                RS_filt.loc[mask, j] = np.nan
        return RS_filt

    def _get_score_values(self, j: str, sig_only: bool,
                          alpha: float) -> np.ndarray:
        """Get score values for predictor j, optionally filtered."""
        vals = self.RS[j].values.copy()
        if sig_only:
            self._check_bootstrapped("sig_only filtering")
            mask = self.RS_boot[j]["p_boot_two_sided"].values >= alpha
            vals[mask] = np.nan
        return vals[np.isfinite(vals)]

    def _get_score_series(self, j: str, sig_only: bool,
                          alpha: float) -> np.ndarray:
        """Get full score series (with NaN for non-significant if requested)."""
        vals = self.RS[j].values.copy()
        if sig_only:
            self._check_bootstrapped("sig_only filtering")
            mask = self.RS_boot[j]["p_boot_two_sided"].values >= alpha
            vals[mask] = np.nan
        return vals

    def _sig_counts_single(self, j: str, alpha: float):
        """Return (n_sig, n_tot, pct) for predictor j."""
        n_tot = self.RS[j].notna().sum()
        if self._bootstrapped:
            n_sig = (self.RS_boot[j]["p_boot_two_sided"] < alpha).sum()
        else:
            n_sig = n_tot
        pct = 100 * n_sig / n_tot if n_tot > 0 else 0
        return n_sig, n_tot, pct

    def _sig_label(self, j: str, sig_only: bool, alpha: float) -> str:
        """Build legend label with significance counts."""
        if not sig_only:
            return j
        n_sig, n_tot, pct = self._sig_counts_single(j, alpha)
        return f"{j}  ({n_sig}/{n_tot}, {pct:.0f}%)"

    def _plot_grouped_kde(self, data, predictors, group_col,
                          sig_only, alpha, agg, figsize):
        """Faceted KDE plots grouped by a categorical column."""
        n = len(predictors)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
        axes = [axes] if n == 1 else list(axes)

        if agg == "between":
            plot_data = data.groupby(
                [self.unit_col, group_col]
            )[predictors].mean().reset_index()
        else:
            plot_data = data

        sig_tag = f" (p<{alpha})" if sig_only else ""
        for ax, j in zip(axes, predictors):
            for cat, grp in plot_data.groupby(group_col):
                xs, ys = self._kde(grp[j].dropna().values)
                if xs is not None:
                    ax.plot(xs, ys, linewidth=2, label=str(cat))
            ax.axvline(0, color="dimgrey", linestyle=":", linewidth=1)
            ax.set_title(self._sig_label(j, sig_only, alpha))
            ax.set_xlabel("RS" if agg == "between" else "RS deviation")
            ax.legend(title=group_col, fontsize=8)
        axes[0].set_ylabel("Density")
        label = "BETWEEN" if agg == "between" else "WITHIN"
        fig.suptitle(f"{label} by {group_col}{sig_tag}", fontsize=13)
        plt.tight_layout()
        plt.show(block=False)
        return fig, axes

    # ================================================================
    # PRIVATE — validation guards
    # ================================================================

    def _check_fitted(self, method: str):
        if not self._fitted:
            raise RuntimeError(
                f"Cannot call {method}() before fit(). "
                f"Run model.fit(...) first."
            )

    def _check_bootstrapped(self, context: str):
        if not self._bootstrapped:
            raise RuntimeError(
                f"{context} requires bootstrap results. "
                f"Run model.bootstrap(...) first."
            )

    def _check_panel(self, method: str):
        if self.mode != "panel":
            raise RuntimeError(
                f"{method}() is only available for mode='panel'."
            )
