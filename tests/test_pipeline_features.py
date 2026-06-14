import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.features import extract_ecg_features


def test_extract_ecg_features_returns_expected_feature_count_and_names():
    ecg = np.ones((5000, 12), dtype=float)

    features = extract_ecg_features(ecg)

    assert len(features) == 358
    assert "lead1_mean" in features
    assert "lead12_cv" in features
    assert "corr_I_II" in features
    assert "global_energy_total" in features


def test_extract_ecg_features_constant_signal_has_known_statistics():
    ecg = np.full((5000, 12), 2.0, dtype=float)

    features = extract_ecg_features(ecg)

    assert features["lead1_mean"] == pytest.approx(2.0)
    assert features["lead1_median"] == pytest.approx(2.0)
    assert features["lead1_std"] == pytest.approx(0.0)
    assert features["lead1_min"] == pytest.approx(2.0)
    assert features["lead1_max"] == pytest.approx(2.0)
    assert features["lead1_range"] == pytest.approx(0.0)
    assert features["lead1_rms"] == pytest.approx(2.0)
    assert features["lead1_energy"] == pytest.approx(5000 * 2.0**2)
    assert features["lead1_norm_energy"] == pytest.approx(4.0)
    assert features["lead1_mad"] == pytest.approx(0.0)
    assert features["lead1_zcr"] == pytest.approx(0.0)
    assert features["lead1_autocorr1"] == pytest.approx(0.0)
    assert features["global_energy_total"] == pytest.approx(12 * 5000 * 2.0**2)


def test_extract_ecg_features_sine_signal_detects_dominant_frequency():
    sampling_rate = 500
    frequency_hz = 10
    t = np.arange(5000) / sampling_rate
    sine = np.sin(2 * np.pi * frequency_hz * t)
    ecg = np.tile(sine.reshape(-1, 1), (1, 12))

    features = extract_ecg_features(ecg)

    assert features["lead1_mean"] == pytest.approx(0.0, abs=1e-3)
    assert features["lead1_rms"] == pytest.approx(math.sqrt(0.5), rel=1e-2)
    assert features["lead1_dom_freq"] == pytest.approx(frequency_hz, abs=1.0)
    assert features["corr_I_II"] == pytest.approx(1.0, abs=1e-6)


def test_extract_ecg_features_cross_lead_correlation_for_inverse_signals():
    sampling_rate = 500
    t = np.arange(5000) / sampling_rate
    base = np.sin(2 * np.pi * 5 * t)
    ecg = np.zeros((5000, 12), dtype=float)
    ecg[:, 0] = base
    ecg[:, 1] = -base
    ecg[:, 2:] = np.tile(base.reshape(-1, 1), (1, 10))

    features = extract_ecg_features(ecg)

    assert features["corr_I_II"] == pytest.approx(-1.0, abs=1e-6)
    assert features["corr_I_III"] == pytest.approx(1.0, abs=1e-6)


def test_extract_ecg_features_contains_only_finite_values_for_flat_signal():
    ecg = np.zeros((5000, 12), dtype=float)

    features = extract_ecg_features(ecg)

    non_finite = {
        name: value
        for name, value in features.items()
        if not np.isfinite(value)
    }
    assert non_finite == {}
