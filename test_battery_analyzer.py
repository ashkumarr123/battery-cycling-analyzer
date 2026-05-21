"""
Unit tests for battery_analyzer module.
Run with: python -m pytest test_battery_analyzer.py -v
"""

import numpy as np
import pandas as pd
import pytest
from battery_analyzer import (
    generate_calce_like_dataset,
    detect_knee_point,
    compute_dqdv,
    generate_voltage_curves,
)


class TestGenerateDataset:
    """Tests for generate_calce_like_dataset()"""

    def test_basic_generation(self):
        """Test basic dataset generation"""
        df = generate_calce_like_dataset(n_cycles=100)
        assert len(df) == 100
        assert "cycle" in df.columns
        assert "discharge_cap_ah" in df.columns
        assert "coulombic_eff" in df.columns

    def test_capacity_monotonic_decrease(self):
        """Capacity should generally decrease (with noise)"""
        df = generate_calce_like_dataset(n_cycles=200, seed=42)
        # Check that final capacity is less than initial (allowing for small noise)
        assert df.discharge_cap_ah.iloc[-1] < df.discharge_cap_ah.iloc[0]

    def test_reproducibility(self):
        """Same seed should produce same data"""
        df1 = generate_calce_like_dataset(n_cycles=100, seed=42)
        df2 = generate_calce_like_dataset(n_cycles=100, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_invalid_inputs(self):
        """Test error handling for invalid inputs"""
        with pytest.raises(ValueError):
            generate_calce_like_dataset(n_cycles=5)  # Too few cycles
        with pytest.raises(ValueError):
            generate_calce_like_dataset(nominal_capacity_ah=-1)  # Negative capacity


class TestKneePointDetection:
    """Tests for detect_knee_point()"""

    def test_simple_knee_detection(self):
        """Test knee detection on synthetic data"""
        cycles = np.array([1, 50, 100, 150, 200, 250, 300])
        # Create data with clear knee around cycle 200
        soh = np.array([100, 98, 95, 90, 70, 40, 20])
        knee = detect_knee_point(cycles, soh)
        # Knee should be detected in the accelerated region
        assert knee >= 150

    def test_linear_degradation(self):
        """Test on linear degradation (no knee)"""
        cycles = np.arange(1, 101)
        soh = 100 - cycles * 0.5
        knee = detect_knee_point(cycles, soh)
        # With linear degradation, knee should be near middle
        assert 40 < knee < 60

    def test_invalid_inputs(self):
        """Test error handling"""
        with pytest.raises(ValueError):
            detect_knee_point(np.array([1]), np.array([100]))  # Too short
        with pytest.raises(ValueError):
            detect_knee_point(np.array([1, 2]), np.array([100]))  # Length mismatch


class TestComputeDqdv:
    """Tests for compute_dqdv()"""

    def test_basic_computation(self):
        """Test basic dQ/dV computation"""
        voltage = np.linspace(2.7, 4.2, 500)
        capacity = np.linspace(0, 1.1, 500)
        v_mid, dqdv = compute_dqdv(voltage, capacity)
        assert len(v_mid) > 0
        assert len(dqdv) > 0

    def test_invalid_inputs(self):
        """Test error handling"""
        with pytest.raises(ValueError):
            compute_dqdv(np.array([1, 2]), np.array([1, 2, 3]))  # Length mismatch
        with pytest.raises(ValueError):
            compute_dqdv(np.array([1]), np.array([1]))  # Too short


class TestGenerateVoltageCurves:
    """Tests for generate_voltage_curves()"""

    def test_curve_generation(self):
        """Test voltage curve generation"""
        curves = generate_voltage_curves([1, 100, 200])
        assert len(curves) == 3
        assert 1 in curves
        assert 100 in curves
        assert 200 in curves

    def test_curve_structure(self):
        """Test curve data structure"""
        curves = generate_voltage_curves([50])
        assert "q" in curves[50]
        assert "v_charge" in curves[50]
        assert "v_discharge" in curves[50]

    def test_voltage_bounds(self):
        """Test that voltages are within physical bounds"""
        curves = generate_voltage_curves([100])
        v_discharge = curves[100]["v_discharge"]
        assert np.all(v_discharge >= 2.5)
        assert np.all(v_discharge <= 4.2)

    def test_invalid_input_type(self):
        """Test error handling for invalid type"""
        with pytest.raises(TypeError):
            generate_voltage_curves("not_a_list")


class TestIntegration:
    """Integration tests"""

    def test_full_pipeline(self):
        """Test complete analysis pipeline"""
        # Generate data
        df = generate_calce_like_dataset(n_cycles=150)
        
        # Detect knee
        knee = detect_knee_point(df.cycle.values, df.soh_pct.values)
        assert 1 <= knee <= 150
        
        # Generate curves
        curves = generate_voltage_curves([1, 50, 100, 150])
        assert len(curves) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
