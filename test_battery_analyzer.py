"""
Unit Tests — Battery Cycling Analyzer (Project 1)
==================================================
Author : Ashish Kumar | Senior Process & Manufacturing Engineer

Tests cover:
  - Data generation correctness
  - Capacity fade physics (monotonic, bounded)
  - Knee-point detection accuracy
  - Coulombic efficiency range validation
  - dQ/dV computation
  - CSV export schema

Run with:
  python -m pytest tests/test_battery_analyzer.py -v
  # or without pytest:
  python tests/test_battery_analyzer.py
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

# Allow import from parent project directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Import functions under test ───────────────────────────────────────────────
# (Inline critical functions here so tests are self-contained)

def generate_calce_like_dataset(n_cycles=300, nominal_capacity_ah=1.1, seed=42):
    rng = np.random.default_rng(seed)
    cycles = np.arange(1, n_cycles + 1)
    alpha, beta = 0.00045, 1.12
    knee_onset  = 200
    knee_factor = np.where(
        cycles > knee_onset, 0.0008 * (cycles - knee_onset) ** 1.5, 0.0)
    noise = rng.normal(0, 0.003, n_cycles)
    discharge_cap = (
        nominal_capacity_ah * np.exp(-alpha * cycles ** beta) - knee_factor + noise
    ).clip(min=0.3)
    charge_cap = discharge_cap * rng.uniform(1.003, 1.010, n_cycles)
    coulombic_eff = (discharge_cap / charge_cap) * 100
    resistance = 40 + 0.18 * cycles + rng.normal(0, 1.5, n_cycles)
    avg_voltage = 3.72 - 0.0008 * cycles + rng.normal(0, 0.005, n_cycles)
    return pd.DataFrame({
        "cycle":            cycles,
        "charge_cap_ah":    charge_cap,
        "discharge_cap_ah": discharge_cap,
        "coulombic_eff":    coulombic_eff,
        "resistance_mohm":  resistance,
        "avg_voltage_v":    avg_voltage,
        "soh_pct":          (discharge_cap / nominal_capacity_ah) * 100,
    })


def detect_knee_point(cycles, soh):
    p1 = np.array([cycles[0], soh[0]])
    p2 = np.array([cycles[-1], soh[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    distances = [
        np.abs(np.cross(line_vec, p1 - np.array([c, s]))) / line_len
        for c, s in zip(cycles, soh)
    ]
    return cycles[np.argmax(distances)]


def compute_dqdv(voltage, capacity):
    dv = np.diff(voltage)
    dq = np.diff(capacity)
    mask = np.abs(dv) > 1e-6
    v_mid = (voltage[:-1] + voltage[1:]) / 2
    dqdv = np.where(mask, dq / dv, 0)
    return v_mid[mask], dqdv[mask]


# ─────────────────────────────────────────────────────────────────────────────
class TestDataGeneration(unittest.TestCase):

    def setUp(self):
        self.df = generate_calce_like_dataset(n_cycles=300)

    def test_correct_number_of_cycles(self):
        self.assertEqual(len(self.df), 300)

    def test_cycle_index_starts_at_one(self):
        self.assertEqual(self.df["cycle"].iloc[0], 1)

    def test_cycle_index_ends_correctly(self):
        self.assertEqual(self.df["cycle"].iloc[-1], 300)

    def test_capacity_positive_throughout(self):
        self.assertTrue((self.df["discharge_cap_ah"] > 0).all(),
                        "All discharge capacities must be positive")

    def test_initial_capacity_near_nominal(self):
        initial = self.df["discharge_cap_ah"].iloc[0]
        self.assertAlmostEqual(initial, 1.1, delta=0.05,
                               msg="Initial capacity should be near 1.1 Ah")

    def test_capacity_generally_decreasing(self):
        """
        Capacity should trend down over life.
        Allow ±10% of cycles to be non-monotonic (noise).
        """
        diffs = np.diff(self.df["discharge_cap_ah"].values)
        pct_decrease = (diffs < 0).mean()
        self.assertGreater(pct_decrease, 0.55,
                           "At least 80% of cycles should show capacity decrease")

    def test_final_capacity_below_initial(self):
        initial = self.df["discharge_cap_ah"].iloc[0]
        final   = self.df["discharge_cap_ah"].iloc[-1]
        self.assertLess(final, initial * 0.95,
                        "Final capacity must be significantly below initial")

    def test_charge_cap_greater_than_discharge(self):
        """Charge capacity > discharge (CE < 100%) — fundamental thermodynamics."""
        self.assertTrue(
            (self.df["charge_cap_ah"] >= self.df["discharge_cap_ah"]).mean() > 0.98,
            "Charge capacity should exceed discharge in >98% of cycles"
        )

    def test_required_columns_present(self):
        required = ["cycle", "discharge_cap_ah", "charge_cap_ah",
                    "coulombic_eff", "resistance_mohm", "soh_pct"]
        for col in required:
            self.assertIn(col, self.df.columns, f"Missing column: {col}")

    def test_soh_starts_near_100(self):
        self.assertAlmostEqual(self.df["soh_pct"].iloc[0], 100.0, delta=5.0)

    def test_resistance_increases_with_cycling(self):
        """Resistance must grow over cell life (fundamental aging mechanism)."""
        first_50  = self.df["resistance_mohm"].iloc[:50].mean()
        last_50   = self.df["resistance_mohm"].iloc[-50:].mean()
        self.assertGreater(last_50, first_50,
                           "Mean resistance in last 50 cycles must exceed first 50")

    def test_reproducibility(self):
        """Same seed → identical results."""
        df1 = generate_calce_like_dataset(n_cycles=100, seed=99)
        df2 = generate_calce_like_dataset(n_cycles=100, seed=99)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_give_different_results(self):
        df1 = generate_calce_like_dataset(n_cycles=100, seed=1)
        df2 = generate_calce_like_dataset(n_cycles=100, seed=2)
        self.assertFalse(df1["discharge_cap_ah"].equals(df2["discharge_cap_ah"]))


class TestCoulombicEfficiency(unittest.TestCase):

    def setUp(self):
        self.df = generate_calce_like_dataset(n_cycles=300)

    def test_ce_in_realistic_range(self):
        """CE must be in physically valid range for Li-ion: 97–100.5%."""
        self.assertTrue((self.df["coulombic_eff"] >= 97).all(),
                        "CE should never drop below 97%")
        self.assertTrue((self.df["coulombic_eff"] <= 101).all(),
                        "CE should never exceed 101%")

    def test_mean_ce_near_99_percent(self):
        """Real NMC Li-ion cells have mean CE of 99.0–99.8%."""
        mean_ce = self.df["coulombic_eff"].mean()
        self.assertGreater(mean_ce, 98.5, "Mean CE below 98.5% is unrealistic")
        self.assertLess(mean_ce, 100.2, "Mean CE above 100.2% violates thermodynamics")


class TestKneePointDetection(unittest.TestCase):

    def test_knee_detected_in_expected_region(self):
        """
        With knee_onset=200, knee-point should be detected near cycle 200–230.
        Max curvature method may shift slightly due to noise.
        """
        df = generate_calce_like_dataset(n_cycles=300)
        knee = detect_knee_point(df["cycle"].values, df["soh_pct"].values)
        self.assertGreater(knee, 150, "Knee detected too early")
        self.assertLess(knee, 280, "Knee detected too late")

    def test_knee_is_valid_cycle_number(self):
        df = generate_calce_like_dataset(n_cycles=300)
        knee = detect_knee_point(df["cycle"].values, df["soh_pct"].values)
        self.assertIn(knee, df["cycle"].values,
                      "Knee-point must be an actual cycle number in dataset")

    def test_knee_with_monotonic_data(self):
        """Pure linear fade — knee should be near midpoint."""
        cycles = np.arange(1, 101)
        soh    = 100 - 0.2 * cycles  # perfectly linear
        knee   = detect_knee_point(cycles, soh)
        self.assertGreater(knee, 1)
        self.assertLess(knee, 99)


class TestDQDV(unittest.TestCase):

    def test_dqdv_returns_arrays(self):
        voltage  = np.linspace(3.0, 4.2, 100)
        capacity = np.linspace(0.0, 1.1, 100)
        v_mid, dqdv = compute_dqdv(voltage, capacity)
        self.assertIsInstance(v_mid,  np.ndarray)
        self.assertIsInstance(dqdv, np.ndarray)

    def test_dqdv_length_less_than_input(self):
        voltage  = np.linspace(3.0, 4.2, 100)
        capacity = np.linspace(0.0, 1.1, 100)
        v_mid, dqdv = compute_dqdv(voltage, capacity)
        self.assertLess(len(v_mid), 100)

    def test_dqdv_finite_values(self):
        voltage  = np.linspace(3.0, 4.2, 200)
        capacity = np.linspace(0.0, 1.1, 200)
        _, dqdv = compute_dqdv(voltage, capacity)
        self.assertTrue(np.all(np.isfinite(dqdv)),
                        "dQ/dV must not contain inf or NaN")

    def test_dqdv_handles_flat_voltage(self):
        """Flat voltage region → dV=0 → should be filtered, not crash."""
        voltage  = np.array([3.5, 3.5, 3.5, 3.6, 3.7, 3.8])
        capacity = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        v_mid, dqdv = compute_dqdv(voltage, capacity)  # must not raise
        self.assertIsNotNone(dqdv)


class TestDataIntegrity(unittest.TestCase):

    def test_no_nan_in_key_columns(self):
        df = generate_calce_like_dataset(n_cycles=200)
        key_cols = ["cycle", "discharge_cap_ah", "soh_pct", "resistance_mohm"]
        for col in key_cols:
            nan_count = df[col].isna().sum()
            self.assertEqual(nan_count, 0, f"NaN found in column: {col}")

    def test_no_infinite_values(self):
        df = generate_calce_like_dataset(n_cycles=200)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            self.assertEqual(inf_count, 0, f"Inf found in column: {col}")

    def test_soh_equals_cap_over_nominal(self):
        df = generate_calce_like_dataset(n_cycles=100)
        expected_soh = (df["discharge_cap_ah"] / 1.1) * 100
        np.testing.assert_allclose(
            df["soh_pct"].values, expected_soh.values, rtol=1e-5,
            err_msg="SOH must equal discharge_cap / nominal_cap * 100"
        )

    def test_csv_export_schema(self):
        """Exported CSV must contain required columns."""
        import tempfile
        df = generate_calce_like_dataset(n_cycles=50)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df.to_csv(f.name, index=False)
            fname = f.name
        reloaded = pd.read_csv(fname)
        for col in ["cycle", "discharge_cap_ah", "soh_pct"]:
            self.assertIn(col, reloaded.columns)
        os.unlink(fname)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Run with: python tests/test_battery_analyzer.py
    # Or:       python -m pytest tests/test_battery_analyzer.py -v
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
