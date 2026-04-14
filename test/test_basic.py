"""
tests/test_basic.py
===================
Basic tests for the rscore package using synthetic data.

Run with:  pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI

from rscore import RScore


# =====================================================================
# Fixtures — synthetic data
# =====================================================================

@pytest.fixture
def cross_section_data():
    """Synthetic cross-section: y = 2*x1 + heterogeneous slope on x2."""
    rng = np.random.default_rng(123)
    n = 200
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    # True model: slope of x2 depends on x1
    # y = 2*x1 + (1 + 0.5*x1)*x2 + 0.3*x3 + noise
    y = 2 * x1 + (1 + 0.5 * x1) * x2 + 0.3 * x3 + rng.standard_normal(n)

    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})
    return df


@pytest.fixture
def panel_data():
    """Synthetic balanced panel: 20 units x 10 periods."""
    rng = np.random.default_rng(456)
    n_units, T = 20, 10
    units = np.repeat(np.arange(n_units), T)
    times = np.tile(np.arange(T), n_units)

    # Unit fixed effects
    alpha_i = rng.standard_normal(n_units)

    x1 = rng.standard_normal(n_units * T)
    x2 = rng.standard_normal(n_units * T)
    # Slope of x1 depends on x2
    y = (alpha_i[units] + (1 + 0.3 * x2) * x1 + 0.5 * x2
         + rng.standard_normal(n_units * T) * 0.5)

    df = pd.DataFrame({
        "unit": [f"U{u:02d}" for u in units],
        "time": times,
        "y": y, "x1": x1, "x2": x2,
    })
    return df


# =====================================================================
# Tests — cross-section
# =====================================================================

class TestCrossSection:
    """Tests for cross-section mode."""

    def test_fit_returns_self(self, cross_section_data):
        model = RScore(mode="cross_section")
        result = model.fit(cross_section_data, y_col="y",
                           predictors=["x1", "x2"], controls=["x3"])
        assert result is model

    def test_rs_shape(self, cross_section_data):
        model = RScore(mode="cross_section")
        model.fit(cross_section_data, y_col="y",
                  predictors=["x1", "x2"], controls=["x3"])
        assert model.RS.shape == (200, 2)
        assert list(model.RS.columns) == ["x1", "x2"]

    def test_r2_positive(self, cross_section_data):
        model = RScore(mode="cross_section")
        model.fit(cross_section_data, y_col="y",
                  predictors=["x1", "x2"], controls=["x3"])
        for j, r2 in model.R2.items():
            assert 0 < r2 <= 1, f"R2 for {j} out of range: {r2}"

    def test_wald_present(self, cross_section_data):
        model = RScore(mode="cross_section")
        model.fit(cross_section_data, y_col="y",
                  predictors=["x1", "x2"], controls=["x3"])
        for j in ["x1", "x2"]:
            assert "F_stat" in model.wald[j]
            assert "p_value" in model.wald[j]

    def test_wald_detects_interaction(self, cross_section_data):
        """The x2 model has a true interaction with x1 -> Wald should reject."""
        model = RScore(mode="cross_section")
        model.fit(cross_section_data, y_col="y",
                  predictors=["x1", "x2"], controls=["x3"])
        # x2's slope depends on x1, so the x2 model should show
        # significant interaction
        assert model.wald["x2"]["p_value"] < 0.05

    def test_aggregates(self, cross_section_data):
        model = RScore(mode="cross_section")
        model.fit(cross_section_data, y_col="y",
                  predictors=["x1", "x2"], controls=["x3"])
        assert len(model.MFR) == 2
        assert len(model.MUR) == 200
        assert len(model.TUR) == 200

    def test_bootstrap(self, cross_section_data):
        model = RScore(mode="cross_section")
        model.fit(cross_section_data, y_col="y",
                  predictors=["x1", "x2"], controls=["x3"])
        model.bootstrap(B=50, seed=0)
        assert model._bootstrapped
        for j in ["x1", "x2"]:
            assert "p_boot_two_sided" in model.RS_boot[j].columns

    def test_plot_distribution(self, cross_section_data):
        model = RScore(mode="cross_section")
        model.fit(cross_section_data, y_col="y",
                  predictors=["x1", "x2"], controls=["x3"])
        fig, ax = model.plot_rs()
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_summary_string(self, cross_section_data):
        model = RScore(mode="cross_section")
        model.fit(cross_section_data, y_col="y",
                  predictors=["x1", "x2"], controls=["x3"])
        s = model.summary()
        assert "R-squared" in s
        assert "Wald" in s


# =====================================================================
# Tests — panel
# =====================================================================

class TestPanel:
    """Tests for panel mode."""

    def test_panel_requires_unit_col(self):
        with pytest.raises(ValueError, match="unit_col"):
            RScore(mode="panel")

    def test_fit_panel(self, panel_data):
        model = RScore(mode="panel", unit_col="unit")
        model.fit(panel_data, y_col="y", predictors=["x1", "x2"])
        assert model.RS.shape == (200, 2)

    def test_no_constant_in_panel(self, panel_data):
        """Panel with FE should not add a constant."""
        model = RScore(mode="panel", unit_col="unit")
        model.fit(panel_data, y_col="y", predictors=["x1", "x2"])
        assert not model._add_constant

    def test_bootstrap_cluster(self, panel_data):
        model = RScore(mode="panel", unit_col="unit")
        model.fit(panel_data, y_col="y", predictors=["x1", "x2"])
        model.bootstrap(B=30, seed=0)
        assert model._bootstrapped

    def test_plot_decompose(self, panel_data):
        model = RScore(mode="panel", unit_col="unit")
        model.fit(panel_data, y_col="y", predictors=["x1", "x2"])
        fig, ax,summary = model.plot_decompose()
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


    def test_plot_timeseries(self, panel_data):
        model = RScore(mode="panel", unit_col="unit")
        model.fit(panel_data, y_col="y", predictors=["x1", "x2"])
        fig, ax = model.plot_timeseries(
            predictor="x1", units=["U00", "U01"], time_col="time"
        )
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_panel_plot_fails_for_cross_section(self, cross_section_data):
        model = RScore(mode="cross_section")
        model.fit(cross_section_data, y_col="y",
                  predictors=["x1", "x2"], controls=["x3"])
        with pytest.raises(RuntimeError, match="only available for.*panel"):
            model.plot_decompose()


# =====================================================================
# Tests — error handling
# =====================================================================

class TestValidation:
    """Tests for validation guards."""

    def test_bootstrap_before_fit_raises(self):
        model = RScore(mode="cross_section")
        with pytest.raises(RuntimeError, match="before fit"):
            model.bootstrap()

    def test_plot_before_fit_raises(self):
        model = RScore(mode="cross_section")
        with pytest.raises(RuntimeError, match="before fit"):
            model.plot_rs()

    def test_sig_only_without_bootstrap_raises(self, cross_section_data):
        model = RScore(mode="cross_section")
        model.fit(cross_section_data, y_col="y",
                  predictors=["x1", "x2"], controls=["x3"])
        with pytest.raises(RuntimeError, match="bootstrap"):
            model.plot_rs(sig_only=True)

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode"):
            RScore(mode="invalid")
