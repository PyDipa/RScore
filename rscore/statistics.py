# ===================================================================
#  Standardization Test
# ===================================================================
# Four distribution families are always evaluated:
#     Gaussian  |  Gamma  |  Pearson type III  |  Gaussian KDE (non-parametric)
#
# Model selection criterion
# -------------------------
# - KS statistic D  : primary, comparable across ALL four families (lower = better).
# - AIC             : secondary, parametric families only (Gaussian / Gamma / Pearson III).
#                     KDE is excluded from AIC ranking because no canonical k is defined.

# KS p-value caveat
# -----------------
# When distribution parameters are estimated by MLE from the same sample, the
# classical KS p-values are anti-conservative (Lilliefors 1967; Stephens 1974).
# They are reported here as indicative; for formal inference use the bootstrap-
# corrected p-values via `n_bootstrap > 0` (≥ 999 recommended).
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import warnings
from typing import Optional

_ALL_DISTS = ("gaussian", "gamma", "pearson3", "kde")

# helpers
# PRIVATE CORE FITTER  (single distribution, single clean array)
def _fit_single_dist(dataset,dist=None):
    """
    Fit *one* distribution to a clean (finite, 1-D) array and return stats.

    Parameters
    ----------
    dataset : np.ndarray
        Finite-only values (caller must filter).
    dist : {"gaussian", "gamma", "pearson3", "kde"}

    Returns
    -------
    dict
        distribution, params, KS_statistic, KS_p_value, log_likelihood,
        AIC, k_params, error_percent, goodness_percent.

    """
    if dist is None:
        dist = 'gaussian'
    dist = dist.lower()

    # ── Gamma ──────────────────────────────────────────────────────────────
    if dist == "gamma":
        # SPI-canonical: fit solo su positivi, nessuno shift
        pos_mask = dataset > 0
        if not np.any(pos_mask):
            raise ValueError("Gamma fit: nessun valore positivo.")
        data_used = dataset[pos_mask]  # ← solo positivi
        shape, loc, scale = stats.gamma.fit(data_used, floc=0)
        # Rivaluta KS sull'intero dataset tramite zero-inflation CDF
        qq = np.sum(dataset == 0) / len(dataset)

        def _zinfl_cdf(x):
            return qq + (1 - qq) * stats.gamma.cdf(x, shape, loc=loc, scale=scale)

        D, p_ks = stats.kstest(dataset[pos_mask], "gamma", args=(shape, loc, scale))
        logpdf = stats.gamma.logpdf(data_used, shape, loc=loc, scale=scale)
        params = {"shape": shape, "loc": loc, "scale": scale, "qq": float(qq)}
        k = 3
    # ── Pearson type III ───────────────────────────────────────────────────
    elif dist == "pearson3":
        data_used = dataset
        shape, loc, scale = stats.pearson3.fit(data_used)
        D, p_ks = stats.kstest(data_used, "pearson3", args=(shape, loc, scale))
        logpdf = stats.pearson3.logpdf(data_used, shape, loc=loc, scale=scale)
        params = {"shape": shape, "loc": loc, "scale": scale}
        k = 3

    # ── Gaussian ────────────────────────────────────────────────────────────
    elif dist == "gaussian":
        data_used = dataset
        mu    = np.mean(data_used)
        sigma = np.std(data_used, ddof=0)
        if sigma == 0:
            raise ValueError("Gaussian fit requires non-zero variance.")
        D, p_ks = stats.kstest(data_used, "norm", args=(mu, sigma))
        logpdf = stats.norm.logpdf(data_used, loc=mu, scale=sigma)
        params = {"mu": mu, "sigma": sigma}
        k = 2

    # ── Gaussian KDE (non-parametric) ───────────────────────────────────────
    elif dist == "kde":
        data_used = dataset
        kde = stats.gaussian_kde(data_used, bw_method="silverman")

        # Vectorised CDF via trapezoidal integration on a fine grid
        std   = np.std(data_used)
        # bw    = kde.predictor * std
        bw = kde.factor * std
        x_min = data_used.min() - 4 * bw
        x_max = data_used.max() + 4 * bw
        x_grid   = np.linspace(x_min, x_max, 4096)
        pdf_grid = kde.evaluate(x_grid)
        cdf_grid = np.cumsum(pdf_grid) * (x_grid[1] - x_grid[0])
        cdf_grid /= cdf_grid[-1]                       # ensure [0, 1]

        def kde_cdf(x: np.ndarray) -> np.ndarray:
            return np.interp(x, x_grid, cdf_grid)

        D, p_ks = stats.kstest(data_used, kde_cdf)

        pdf_vals = kde.evaluate(data_used)
        pdf_vals = np.clip(pdf_vals, 1e-300, None)
        logpdf   = np.log(pdf_vals)

        # Store only serialisable primitives; expose CDF callable separately
        params = {
            "bw_method":  "silverman",
            "bw_factor": float(kde.factor),
            "n_fit":      int(len(data_used)),
            "_kde_cdf":   kde_cdf,      # callable, not JSON-serialisable
        }
        k = None  # AIC undefined for KDE

    else:
        raise ValueError(
            f"Unknown distribution '{dist}'. "
            "Choose from: 'gaussian', 'gamma', 'pearson3', 'kde'."
        )

    # ── Derived metrics ─────────────────────────────────────────────────────
    log_likelihood = float(np.sum(logpdf))
    aic = (2 * k - 2 * log_likelihood) if k is not None else np.nan

    return {
        "distribution":    dist,
        "params":          params,
        "KS_statistic":    float(D),
        "KS_p_value":      float(p_ks),
        "log_likelihood":  log_likelihood,
        "AIC":             float(aic) if not np.isnan(aic) else np.nan,
        "k_params":        k,
        "error_percent":   float(100.0 * D),
        "goodness_percent": float(100.0 * (1.0 - D)),
    }

#  BOOTSTRAP KS p-VALUE CORRECTION  (optional)
def _bootstrap_ks_pvalue(dataset:np.ndarray,fit_result: dict,
    n_bootstrap: int = 999,rng: Optional[np.random.Generator] = None,seed = None
) -> float:
    """
    Parametric bootstrap to correct KS p-values for the Lilliefors bias.

    Resample n_bootstrap synthetic datasets from the fitted distribution,
    refit, compute D* each time → empirical null distribution of D.
    Bootstrap p-value = fraction of D* ≥ D_observed.

    Parameters
    ----------
    dataset   : original data (filtered).
    fit_result: dict returned by _fit_single_dist.
    n_bootstrap: number of bootstrap replicates.
    rng       : optional numpy Generator for reproducibility.

    Returns
    -------
    float : bootstrap-corrected p-value.
    """
    if rng is None:
        rng = np.random.default_rng()

    dist   = fit_result["distribution"]
    params = fit_result["params"]
    n      = len(dataset)
    D_obs  = fit_result["KS_statistic"]
    D_boot = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        # Draw synthetic sample from fitted distribution
        if dist == "gaussian":
            sample = rng.normal(params["mu"], params["sigma"], size=n)
        elif dist == "gamma":
            sample = stats.gamma.rvs(
                params["shape"], loc=params["loc"], scale=params["scale"],
                size=n, random_state=rng.integers(1 << 31)
            )
            if params["shift_applied"]:
                sample -= 1.0          # undo shift for consistent comparison
        elif dist == "pearson3":
            sample = stats.pearson3.rvs(
                params["shape"], loc=params["loc"], scale=params["scale"],
                size=n, random_state=rng.integers(1 << 31)
            )
        else:  # kde — resample from original data (smoothed bootstrap)
            std = np.std(dataset)
            bw  = params["bw_predictor"] * std
            idx    = rng.integers(0, n, size=n)
            sample = dataset[idx] + rng.normal(0, bw, size=n)

        # Refit and compute D*
        try:
            res = _fit_single_dist(sample, dist, shift_for_gamma=(dist == "gamma"))
            D_boot[i] = res["KS_statistic"]
        except Exception:
            D_boot[i] = np.nan

    valid = D_boot[np.isfinite(D_boot)]
    return float(np.mean(valid >= D_obs))

# Fit all four families to `dataset` and return a comparison dict.
def _analyze_all(dataset,  n_bootstrap=0,seed=None):
    """
    Fit all four families to `dataset` and return a comparison dict.

    Selection rule
    --------------
    1. Best KS (all 4) → primary recommendation.
    2. Best AIC (parametric 3 only) → secondary note.
    """
    dataset = dataset[np.isfinite(dataset)]

    skewness = float(stats.skew(dataset))
    _, p_normal = stats.normaltest(dataset)

    fits: dict[str, dict] = {}
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    for d in _ALL_DISTS:
        try:
            res = _fit_single_dist(dataset, d)
            if n_bootstrap > 0:
                res["KS_p_value_bootstrap"] = _bootstrap_ks_pvalue(
                    dataset, res, n_bootstrap=n_bootstrap, rng=rng
                )
            fits[d] = res
        except Exception as exc:
            warnings.warn(f"Fitting '{d}' failed: {exc}", RuntimeWarning)
            fits[d] = None

    # ── Rank by KS statistic (lower = better), None last ──────────────────
    valid_fits = {d: v for d, v in fits.items() if v is not None}
    best_ks  = min(valid_fits, key=lambda d: valid_fits[d]["KS_statistic"])

    # ── Best parametric by AIC ─────────────────────────────────────────────
    parametric = {d: v for d, v in valid_fits.items()
                  if d != "kde" and not np.isnan(v["AIC"])}
    best_aic = min(parametric, key=lambda d: parametric[d]["AIC"]) if parametric else None

    return {
        "skewness":         skewness,
        "normality_p_value": float(p_normal),
        "fits":             fits,
        "best_by_KS":       best_ks,
        "best_by_AIC":      best_aic,
        "recommendation":   best_ks,       # primary
    }


# CDF COMPARISON PLOT
def _plot_cdf_comparison(dataset,analysis,title = ""):
    """
    Empirical CDF (ECDF) vs theoretical CDFs for all fitted families.

    This is the most direct visual diagnostic: how closely does each
    fitted distribution track the empirical CDF?
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax_cdf, ax_pdf = axes

    dataset = dataset[np.isfinite(dataset)]
    # ECDF
    x_ecdf = np.sort(dataset)
    y_ecdf = np.arange(1, len(x_ecdf) + 1) / len(x_ecdf)

    colors = {
        "gaussian": "#2166ac",
        "gamma":    "#d6604d",
        "pearson3": "#4dac26",
        "kde":      "#9970ab",
    }
    labels = {
        "gaussian": "Gaussian",
        "gamma":    "Gamma",
        "pearson3": "Pearson III",
        "kde":      "KDE (Silverman)",
    }

    x_plot = np.linspace(dataset.min(), dataset.max(), 1000)

    fits = analysis["fits"]
    best = analysis["best_by_KS"]

    for d, res in fits.items():
        if res is None:
            continue
        params = res["params"]
        lw  = 2.5 if d == best else 1.2
        ls  = "-"  if d == best else "--"
        col = colors[d]
        D   = res["KS_statistic"]
        lbl = f"{labels[d]}  (D={D:.3f})"
        if d == best:
            lbl += "  *"

        # CDF curve
        if d == "gaussian":
            cdf = stats.norm.cdf(x_plot, loc=params["mu"], scale=params["sigma"])
            pdf = stats.norm.pdf(x_plot, loc=params["mu"], scale=params["sigma"])
        elif d == "gamma":

            cdf = stats.gamma.cdf(
                x_plot, params["shape"],
                loc=params["loc"], scale=params["scale"]
            )
            pdf = stats.gamma.pdf(
                x_plot, params["shape"],
                loc=params["loc"], scale=params["scale"]
            )
        elif d == "pearson3":
            cdf = stats.pearson3.cdf(
                x_plot, params["shape"],
                loc=params["loc"], scale=params["scale"]
            )
            pdf = stats.pearson3.pdf(
                x_plot, params["shape"],
                loc=params["loc"], scale=params["scale"]
            )
        else:  # kde
            cdf = params["_kde_cdf"](x_plot)
            pdf = stats.gaussian_kde(dataset, bw_method="silverman").evaluate(x_plot)

        ax_cdf.plot(x_plot, cdf, color=col, lw=lw, ls=ls, label=lbl)
        ax_pdf.plot(x_plot, pdf, color=col, lw=lw, ls=ls, label=lbl)

    # Empirical CDF
    ax_cdf.step(x_ecdf, y_ecdf, where="post", color="black",
                lw=1.5, alpha=0.8, label="Empirical CDF")
    ax_pdf.hist(dataset, bins="auto", density=True,
                color="black", alpha=0.2, label="Empirical density")
    counts, edges = np.histogram(dataset, bins="auto", density=True)
    ax_pdf.set_ylim(0, counts.max() * 1.15)  # 15% buffer

    for ax, ylabel, ttl in zip(
        [ax_cdf, ax_pdf],
        ["Cumulative probability", "Density"],
        [f"CDF comparison — {title}".strip(" —"),
         f"PDF comparison — {title}".strip(" —")],
    ):
        ax.set_xlabel("Value")
        ax.set_ylabel(ylabel)
        ax.set_title(ttl)
        ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


# PUBLIC API

def test_standardization(data, groups=None,
    plot = True, n_bootstrap = 0, seed = None):
    """
    Fit all four distribution families and recommend the best one.

    Always evaluates: Gaussian | Gamma | Pearson III | Gaussian KDE.

    Primary selection criterion : lowest KS statistic D (scale-free, valid
                                  for all four families including KDE).
    Secondary criterion         : lowest AIC (parametric families only).

    Parameters
    ----------
    data : array-like
        Input data. Temporal aggregation (e.g., for SPI-3/6/12) must be
        performed BEFORE calling this function — results reflect the
        input scale as-is.
    groups : array-like or None
        Optional grouping vector (same length as `data`). Analysis is run
        independently per group.

    plot : bool, default True
        Generate empirical vs theoretical CDF/PDF comparison figures.
    n_bootstrap : int, default 0
        If > 0, compute Lilliefors-corrected KS p-values via parametric
        bootstrap. Recommended ≥ 999 for stable estimates.
    seed : int or None
        Random seed for bootstrap reproducibility.

    Returns
    -------
    dict
        If groups is None:
            {skewness, normality_p_value, fits, best_by_KS, best_by_AIC,
             recommendation}
        If groups is provided:
            {group_label: <same dict>}

    Notes
    -----
    KS p-values without bootstrap correction are anti-conservative when
    parameters are estimated from the same sample (Lilliefors 1967).
    The KS statistic D itself remains a valid distance metric regardless.
    """
    data   = np.asarray(data, dtype=float)


    if groups is None:
        result = _analyze_all(data, n_bootstrap)
        if plot:
            fig = _plot_cdf_comparison(data, result)
            plt.show()
        return result

    groups = np.asarray(groups)
    if len(groups) != len(data):
        raise ValueError("`groups` must have the same length as `data`.")

    results = {}
    for g in np.unique(groups):
        subset = data[groups == g]
        res    = _analyze_all(subset, n_bootstrap)
        if plot:
            fig = _plot_cdf_comparison(subset, res, title=str(g))
            plt.show()
        results[g] = res

    return results


def fit_distribution_stats(data, dist= "gamma", groups=None,
    shift_for_gamma= True, plot = True, n_bootstrap: int = 0,
    seed = None):
    """
    Fit a *single* specified distribution and return goodness-of-fit stats.

    Use `test_standardization` first to identify the best family, then
    call this function to obtain the full fit statistics for that family.

    Parameters
    ----------
    data : array-like
        Input dataset. Results reflect the input time scale as-is.
    dist : {"gaussian", "gamma", "pearson3", "kde"}, default "gamma"
    groups : array-like or None
    shift_for_gamma : bool, default True
    plot : bool, default True
        Show empirical vs theoretical CDF/PDF for the chosen distribution.
    n_bootstrap : int, default 0
        If > 0, compute Lilliefors-corrected KS p-value.
    seed : int or None

    Returns
    -------
    dict
        If groups is None:
            {distribution, skewness, normality_p_value, params,
             KS_statistic, KS_p_value [, KS_p_value_bootstrap],
             log_likelihood, AIC, k_params, error_percent, goodness_percent}
        If groups is provided:
            {group_label: <same dict>}
    """
    data  = np.asarray(data, dtype=float)
    dist  = dist.lower()
    rng   = np.random.default_rng(seed)

    def _single(dataset: np.ndarray) -> dict:
        dataset = dataset[np.isfinite(dataset)]
        skewness  = float(stats.skew(dataset))
        _, p_norm = stats.normaltest(dataset)
        res = _fit_single_dist(dataset, dist, shift_for_gamma=shift_for_gamma)
        if n_bootstrap > 0:
            res["KS_p_value_bootstrap"] = _bootstrap_ks_pvalue(
                dataset, res, n_bootstrap=n_bootstrap, rng=rng
            )
        res["skewness"]          = skewness
        res["normality_p_value"] = float(p_norm)
        return res

    if groups is None:
        dataset_clean = data[np.isfinite(data)]
        res = _single(dataset_clean)

        if plot:
            # Minimal plot: empirical ECDF + fitted CDF
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            ax_cdf, ax_pdf = axes

            x_ecdf = np.sort(dataset_clean)
            y_ecdf = np.arange(1, len(x_ecdf) + 1) / len(x_ecdf)
            x_plot = np.linspace(x_ecdf[0], x_ecdf[-1], 1000)
            params = res["params"]

            if dist == "gaussian":
                cdf = stats.norm.cdf(x_plot, loc=params["mu"], scale=params["sigma"])
                pdf = stats.norm.pdf(x_plot, loc=params["mu"], scale=params["sigma"])
                lbl = "Gaussian fit"
            elif dist == "gamma":
                shift = 1.0 if params["shift_applied"] else 0.0
                cdf = stats.gamma.cdf(
                    x_plot + shift, params["shape"],
                    loc=params["loc"], scale=params["scale"]
                )
                pdf = stats.gamma.pdf(
                    x_plot + shift, params["shape"],
                    loc=params["loc"], scale=params["scale"]
                )
                lbl = "Gamma fit"
            elif dist == "pearson3":
                cdf = stats.pearson3.cdf(
                    x_plot, params["shape"],
                    loc=params["loc"], scale=params["scale"]
                )
                pdf = stats.pearson3.pdf(
                    x_plot, params["shape"],
                    loc=params["loc"], scale=params["scale"]
                )
                lbl = "Pearson III fit"
            else:  # kde
                cdf = params["_kde_cdf"](x_plot)
                pdf = stats.gaussian_kde(dataset_clean, bw_method="silverman").evaluate(x_plot)
                lbl = "KDE (Silverman)"

            ax_cdf.step(x_ecdf, y_ecdf, where="post", color="black",
                        lw=1.5, alpha=0.8, label="Empirical CDF")
            ax_cdf.plot(x_plot, cdf, color="#d6604d", lw=2, label=lbl)
            ax_cdf.set(xlabel="Value", ylabel="Cumulative probability",
                       title=f"CDF — {lbl}  (D={res['KS_statistic']:.3f})")
            ax_cdf.legend()

            ax_pdf.hist(dataset_clean, bins="auto", density=True,
                        color="black", alpha=0.25, label="Empirical density")
            ax_pdf.plot(x_plot, pdf, color="#d6604d", lw=2, label=lbl)
            ax_pdf.set(xlabel="Value", ylabel="Density", title=f"PDF — {lbl}")
            ax_pdf.legend()

            fig.tight_layout()
            plt.show()

        return res

    groups = np.asarray(groups)
    if len(groups) != len(data):
        raise ValueError("`groups` must have the same length as `data`.")

    results = {}
    for g in np.unique(groups):
        results[g] = _single(data[groups == g])
    return results

def standardize_data(data,analysis_result,groups=None,plot = True):
    """
    Transform data to standard normal scores using the distribution recommended
    by `test_standardization()`.

    Method — Probability Integral Transform (PIT):
    -----------------------------------------------
        1.  Fit the recommended distribution to the data
            (parameters are re-estimated here, consistent with _fit_single_dist).
        2.  Map each observation x → p = F(x)  (CDF value ∈ (0, 1)).
        3.  Map p → z = Φ⁻¹(p)               (standard-normal quantile).

    The result is a zero-mean, unit-variance series that can be directly
    compared across sites, variables, and time scales — the same logic
    underlying SPI (McKee 1993) and SPEI (Vicente-Serrano 2010).

    Why re-estimate parameters instead of reusing stored params?
    ------------------------------------------------------------
    When `groups` are present, each group needs its own fit.  For the
    ungrouped case the re-estimation cost is negligible and guarantees
    that the standardized scores are always internally consistent with
    the data passed in (e.g. if the user subsets or filters before calling).

    Parameters
    ----------
    data : array-like
        Original (non-standardized) values.  Must be the same series, or
        a compatible subset, of what was passed to `test_standardization()`.
        Temporal aggregation (SPI-3, SPI-6, …) must be done beforehand.
    analysis_result : dict
        Output of `test_standardization()`.
        - If `groups` is None → flat dict with key "recommendation".
        - If `groups` is provided → nested dict {group: {..., "recommendation"}}.
    groups : array-like or None
        Grouping vector (e.g. month labels for seasonal standardization).
        Must have the same length as `data`.
        Each group is standardized independently using the recommendation
        found for that group in `analysis_result`.
    plot : bool, default True
        Scatter plot of original vs standardized values, plus histogram of
        z-scores with a N(0,1) reference curve.

        CDF values are clipped to [cdf_clip, 1 - cdf_clip] before the
        normal-quantile transform to avoid ±∞ at the tails.

    Returns
    -------
    dict with keys:
        "z_scores"       : np.ndarray, standardized values (NaN preserved).
        "distribution"   : str, distribution used.
        "params"         : dict (or {group: dict} when grouped), fitted params.
        "cdf_values"     : np.ndarray, intermediate CDF values F(x).
        "recommendation" : str (or {group: str} when grouped).

    Raises
    ------
    KeyError
        If `analysis_result` does not contain a "recommendation" key (wrong input).
    ValueError
        If `groups` is provided but not present as keys in `analysis_result`.

    Notes
    -----
    - NaN / ±Inf in `data` are preserved as NaN in `z_scores`.
    - For KDE the CDF is approximated via trapezoidal integration on a fine
      grid (4096 points) — same approach as in `_fit_single_dist`.
    - The Gaussian case degenerates to the classic z-score: z = (x − μ) / σ,
      but is routed through the PIT for uniformity.

    References
    ----------
    McKee T.B. et al. (1993).  J. Am. Meteorol. Soc.
    Vicente-Serrano S.M. et al. (2010).  J. Clim. 23, 1696–1718.
    """

    data   = np.asarray(data, dtype=float)
    valid_mask = np.isfinite(data)
    cdf_clip = 3.17e-5  # ≡ Φ(−4)

    # ── Internal PIT engine ────────────────────────────────────────────────
    def _pit(dataset_clean: np.ndarray, dist: str, params: dict) -> np.ndarray:
        x = dataset_clean
        qq = np.sum(x == 0) / len(x)  # zero-inflation factor

        if dist == "gaussian":
            p = stats.norm.cdf(x, loc=params["mu"], scale=params["sigma"])

        elif dist == "gamma":
            qq = params["qq"]   # already estimated on the whole dataset in _fit_single_dist
            Gx = stats.gamma.cdf(
                x, params["shape"],
                loc=params["loc"], scale=params["scale"],
            )
            p = qq + (1 - qq) * Gx

        elif dist == "pearson3":
            Gx = stats.pearson3.cdf(
                x, params["shape"],
                loc=params["loc"], scale=params["scale"],
            )
            p = qq + (1 - qq) * Gx

        elif dist == "kde":
            kde = stats.gaussian_kde(dataset_clean, bw_method="silverman")
            std = np.std(dataset_clean)
            bw = kde.factor * std
            x_min = dataset_clean.min() - 4 * bw
            x_max = dataset_clean.max() + 4 * bw
            grid = np.linspace(x_min, x_max, 4096)
            pdf_grid = kde.evaluate(grid)
            cdf_grid = np.cumsum(pdf_grid) * (grid[1] - grid[0])
            cdf_grid /= cdf_grid[-1]
            Gx = np.interp(x, grid, cdf_grid)
            p = qq + (1 - qq) * Gx

        return np.clip(p, cdf_clip, 1.0 - cdf_clip)

    # ── Helper: standardize one group ──────────────────────────────────────
    def _standardize_group(
        data_full: np.ndarray,         # full-length array (with NaN)
        mask: np.ndarray,              # boolean, finite positions
        group_result: dict,            # analysis dict for this group
    ) -> tuple[np.ndarray, dict, np.ndarray, str]:
        """
        Returns (z_scores_full, params, cdf_values_full, dist_used).
        NaN positions in data_full remain NaN in z_scores_full.
        """
        rec   = group_result["recommendation"]
        dist  = rec
        # Refit to get fresh params (guarantees consistency)
        fit   = _fit_single_dist(data_full[mask], dist)
        params = fit["params"]

        # PIT on clean data
        p_clean = _pit(data_full[mask], dist, params)
        z_clean = stats.norm.ppf(p_clean)

        # Reconstruct full-length arrays preserving NaN positions
        z_full   = np.full(len(data_full), np.nan)
        cdf_full = np.full(len(data_full), np.nan)
        z_full[mask]   = z_clean
        cdf_full[mask] = p_clean

        return z_full, params, cdf_full, dist

    # ══════════════════════════════════════════════════════════════════════
    # CASE 1: no groups
    # ══════════════════════════════════════════════════════════════════════
    if groups is None:
        if "recommendation" not in analysis_result:
            raise KeyError(
                "'recommendation' key not found in analysis_result. "
                "Pass the direct output of test_standardization() with groups=None."
            )
        z_full, params, cdf_full, dist_used = _standardize_group(
            data, valid_mask, analysis_result
        )

        if plot:
            _plot_standardization(data[valid_mask], z_full[valid_mask], dist_used)

        return {
            "z_scores":      z_full,
            "distribution":  dist_used,
            "params":        params,
            "cdf_values":    cdf_full,
            "recommendation": dist_used,
        }

    # ══════════════════════════════════════════════════════════════════════
    # CASE 2: grouped
    # ══════════════════════════════════════════════════════════════════════
    groups = np.asarray(groups)
    if len(groups) != len(data):
        raise ValueError("`groups` must have the same length as `data`.")

    unique_groups = np.unique(groups)
    missing = [g for g in unique_groups if g not in analysis_result]
    if missing:
        raise ValueError(
            f"Groups {missing} are present in `data` but not in `analysis_result`. "
            "Run test_standardization() with the same `groups` vector."
        )

    z_full   = np.full(len(data), np.nan)
    cdf_full = np.full(len(data), np.nan)
    params_by_group: dict  = {}
    dist_by_group:   dict  = {}

    for g in unique_groups:
        g_mask = (groups == g) & valid_mask
        group_result = analysis_result[g]

        z_g, params_g, cdf_g, dist_g = _standardize_group(
            data, g_mask, group_result
        )
        z_full[g_mask]   = z_g[g_mask]
        cdf_full[g_mask] = cdf_g[g_mask]
        params_by_group[g] = params_g
        dist_by_group[g]   = dist_g

    if plot:
        _plot_standardization(
            data[valid_mask], z_full[valid_mask],
            dist_label="grouped (" + ", ".join(
                f"{g}→{d}" for g, d in dist_by_group.items()
            ) + ")",
        )

    return {
        "z_scores":      z_full,
        "distribution":  dist_by_group,
        "params":        params_by_group,
        "cdf_values":    cdf_full,
        "recommendation": dist_by_group,
    }


def _plot_standardization(original,z_scores,dist_label= ""):
    """
    Two-panel diagnostic:
        Left  — scatter: original values vs z-scores (shows monotonic mapping).
        Right — histogram of z-scores vs N(0,1) reference.

    A well-standardized series should yield:
        - monotonically increasing scatter (left),
        - histogram hugging the N(0,1) bell curve (right).
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax_sc, ax_hz = axes

    # ── Scatter: original vs z ─────────────────────────────────────────────
    ax_sc.scatter(original, z_scores, s=10, alpha=0.5, color="#2166ac")
    ax_sc.axhline(0,  color="grey", lw=0.8, ls="--")
    ax_sc.axhline( 1.96, color="#d6604d", lw=0.8, ls=":", label="±1.96 σ")
    ax_sc.axhline(-1.96, color="#d6604d", lw=0.8, ls=":")
    ax_sc.set_xlabel("Original value")
    ax_sc.set_ylabel("z-score")
    ax_sc.set_title(f"Original → z-score  [{dist_label}]", fontsize=9)
    ax_sc.legend(fontsize=8)

    # ── Histogram of z-scores vs N(0,1) ───────────────────────────────────
    ax_hz.hist(z_scores, bins="auto", density=True,
               color="#2166ac", alpha=0.4, label="Standardized data")
    x_ref = np.linspace(-4, 4, 400)
    ax_hz.plot(x_ref, stats.norm.pdf(x_ref), color="black",
               lw=1.8, label="N(0, 1)")
    ax_hz.set_xlabel("z-score")
    ax_hz.set_ylabel("Density")
    ax_hz.set_title("Distribution of z-scores vs N(0,1)", fontsize=9)
    ax_hz.legend(fontsize=8)

    fig.tight_layout()
    plt.show()
    return fig


