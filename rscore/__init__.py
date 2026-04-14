"""
rscore — Responsiveness Scores for heterogeneous unit-level analysis.

Implements the methodology from Cerulli (2017, The Stata Journal) with
extensions for bootstrap inference on unit-specific scores and panel
fixed-effects estimation.

Quick start
-----------
>>> from rscore import RScore
>>> model = RScore(mode="panel", unit_col="province")
>>> model.fit(df, y_col="y", predictors=["x1", "x2"])
>>> model.bootstrap(B=500)
>>> model.summary()
>>> model.plot_between()
"""

from rscore.core import RScore

__all__ = ["RScore"]
__version__ = "0.1.0"
