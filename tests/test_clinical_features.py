from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.clinical_features import (
    detect_r_peaks,
    extract_clinical_ecg_features,
    extract_rhythm_features,
)


def _synthetic_ecg_with_regular_r_peaks(
    sampling_rate: int = 500,
    seconds: int = 10,
    bpm: int = 60,
) -> np.ndarray:
    n_samples = sampling_rate * seconds
    ecg = np.zeros((n_samples, 12), dtype=float)
    period = int(sampling_rate * 60 / bpm)
    peak_positions = np.arange(period, n_samples - period, period)

    for pos in peak_positions:
        width = int(0.02 * sampling_rate)
        idx = np.arange(max(0, pos - width), min(n_samples, pos + width + 1))
        pulse = np.exp(-0.5 * ((idx - pos) / max(1, width / 3)) ** 2)
        ecg[idx, 1] += pulse
        ecg[idx, 6] -= 0.4 * pulse
        ecg[idx, 10] += 1.2 * pulse
        ecg[idx, 11] += 1.0 * pulse

    return ecg


def test_detect_r_peaks_finds_regular_synthetic_peaks():
    ecg = _synthetic_ecg_with_regular_r_peaks()

    peaks = detect_r_peaks(ecg[:, 1], sampling_rate=500, prefer_neurokit=False)

    assert len(peaks) == 8


def test_extract_rhythm_features_regular_signal_has_expected_heart_rate():
    ecg = _synthetic_ecg_with_regular_r_peaks(bpm=60)

    features = extract_rhythm_features(ecg, sampling_rate=500, lead_index=1)

    assert features["clinical_r_peak_count"] == pytest.approx(8)
    assert features["clinical_rr_mean_ms"] == pytest.approx(1000.0, abs=5.0)
    assert features["clinical_heart_rate_mean_bpm"] == pytest.approx(60.0, abs=1.0)
    assert features["clinical_rmssd_ms"] == pytest.approx(0.0, abs=1e-6)
    assert features["clinical_pnn50"] == pytest.approx(0.0, abs=1e-6)


def test_extract_clinical_ecg_features_returns_finite_values_for_flat_signal():
    ecg = np.zeros((5000, 12), dtype=float)

    features = extract_clinical_ecg_features(ecg, sampling_rate=500)

    assert features["clinical_r_peak_count"] == 0.0
    assert all(np.isfinite(value) for value in features.values())


def test_extract_clinical_ecg_features_includes_voltage_features():
    ecg = _synthetic_ecg_with_regular_r_peaks()

    features = extract_clinical_ecg_features(ecg, sampling_rate=500)

    assert "clinical_sokolow_lyon_voltage" in features
    assert "clinical_cornell_voltage" in features
    assert features["clinical_sokolow_lyon_voltage"] > 0


def test_extract_clinical_ecg_features_includes_interval_and_morphology_features():
    ecg = _synthetic_ecg_with_regular_r_peaks()

    features = extract_clinical_ecg_features(ecg, sampling_rate=500)

    assert "clinical_qrs_duration_mean_ms" in features
    assert "clinical_qt_interval_mean_ms" in features
    assert "clinical_lead2_st_deviation_mean" in features
    assert "clinical_lead2_q_r_ratio_mean" in features
    assert "clinical_t_inversion_global_ratio" in features
    assert all(np.isfinite(value) for value in features.values())
