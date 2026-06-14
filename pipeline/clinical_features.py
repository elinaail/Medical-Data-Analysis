"""Experimental clinically interpretable ECG features.

This module is intentionally separate from the production FastAPI path.
It can be used in research notebooks/scripts to compare baseline features
with additional rhythm, interval, morphology, and QRS-voltage features.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.signal import find_peaks

DEFAULT_SAMPLING_RATE = 500
DEFAULT_RHYTHM_LEAD_INDEX = 1
_EPS = 1e-12


def _as_float(value: float | int | np.floating) -> float:
    value = float(value)
    if np.isfinite(value):
        return value
    return 0.0


def _lead_name(lead_index: int) -> str:
    return f"lead{lead_index + 1}"


def _empty_rhythm_features(prefix: str = "clinical") -> dict[str, float]:
    return {
        f"{prefix}_r_peak_count": 0.0,
        f"{prefix}_rr_mean_ms": 0.0,
        f"{prefix}_rr_std_ms": 0.0,
        f"{prefix}_rr_min_ms": 0.0,
        f"{prefix}_rr_max_ms": 0.0,
        f"{prefix}_heart_rate_mean_bpm": 0.0,
        f"{prefix}_heart_rate_std_bpm": 0.0,
        f"{prefix}_sdnn_ms": 0.0,
        f"{prefix}_rmssd_ms": 0.0,
        f"{prefix}_pnn50": 0.0,
        f"{prefix}_r_amp_mean": 0.0,
        f"{prefix}_r_amp_std": 0.0,
        f"{prefix}_r_amp_min": 0.0,
        f"{prefix}_r_amp_max": 0.0,
        f"{prefix}_r_amp_range": 0.0,
    }


def _empty_qrs_voltage_features(prefix: str = "clinical") -> dict[str, float]:
    features: dict[str, float] = {}
    for lead_idx in range(12):
        name = _lead_name(lead_idx)
        features[f"{prefix}_{name}_qrs_max_mean"] = 0.0
        features[f"{prefix}_{name}_qrs_min_mean"] = 0.0
        features[f"{prefix}_{name}_qrs_ptp_mean"] = 0.0

    features.update(
        {
            f"{prefix}_qrs_global_ptp_mean": 0.0,
            f"{prefix}_sokolow_lyon_voltage": 0.0,
            f"{prefix}_cornell_voltage": 0.0,
        }
    )
    return features


def _empty_interval_features(prefix: str = "clinical") -> dict[str, float]:
    return {
        f"{prefix}_delineated_beat_count": 0.0,
        f"{prefix}_pr_interval_mean_ms": 0.0,
        f"{prefix}_p_duration_mean_ms": 0.0,
        f"{prefix}_qrs_duration_mean_ms": 0.0,
        f"{prefix}_qt_interval_mean_ms": 0.0,
        f"{prefix}_qtc_bazett_mean_ms": 0.0,
        f"{prefix}_qtc_fridericia_mean_ms": 0.0,
        f"{prefix}_st_segment_mean_ms": 0.0,
        f"{prefix}_t_duration_mean_ms": 0.0,
    }


def _empty_wave_morphology_features(prefix: str = "clinical") -> dict[str, float]:
    features: dict[str, float] = {}
    for lead_idx in range(12):
        name = _lead_name(lead_idx)
        features[f"{prefix}_{name}_q_depth_mean"] = 0.0
        features[f"{prefix}_{name}_q_width_mean_ms"] = 0.0
        features[f"{prefix}_{name}_q_r_ratio_mean"] = 0.0
        features[f"{prefix}_{name}_pathologic_q_ratio"] = 0.0
        features[f"{prefix}_{name}_st_deviation_mean"] = 0.0
        features[f"{prefix}_{name}_st_slope_mean"] = 0.0
        features[f"{prefix}_{name}_t_amp_mean"] = 0.0
        features[f"{prefix}_{name}_t_inversion_ratio"] = 0.0

    features.update(
        {
            f"{prefix}_q_depth_global_mean": 0.0,
            f"{prefix}_st_deviation_global_mean": 0.0,
            f"{prefix}_t_amp_global_mean": 0.0,
            f"{prefix}_t_inversion_global_ratio": 0.0,
        }
    )
    return features


def _detect_r_peaks_neurokit(lead: np.ndarray, sampling_rate: int) -> np.ndarray:
    try:
        import neurokit2 as nk
    except ImportError as exc:
        raise RuntimeError("neurokit2 is not installed") from exc

    _, info = nk.ecg_peaks(lead, sampling_rate=sampling_rate)
    return np.asarray(info.get("ECG_R_Peaks", []), dtype=int)


def _detect_r_peaks_scipy(lead: np.ndarray, sampling_rate: int) -> np.ndarray:
    if len(lead) == 0 or np.std(lead) < _EPS:
        return np.array([], dtype=int)

    centered = lead - np.median(lead)
    distance = max(1, int(0.25 * sampling_rate))
    prominence = max(float(np.std(centered)) * 0.5, _EPS)
    height = np.percentile(centered, 75)
    peaks, _ = find_peaks(centered, distance=distance, prominence=prominence, height=height)
    return peaks.astype(int)


def _valid_wave_array(waves: dict, key: str) -> np.ndarray:
    values = np.asarray(waves.get(key, []), dtype=np.float64)
    if values.size == 0:
        return values
    return values[np.isfinite(values)]


def _mean_interval_ms(start: np.ndarray, end: np.ndarray, sampling_rate: int) -> float:
    n = min(len(start), len(end))
    if n == 0:
        return 0.0
    duration = (end[:n] - start[:n]) / sampling_rate * 1000.0
    duration = duration[np.isfinite(duration) & (duration > 0)]
    if len(duration) == 0:
        return 0.0
    return _as_float(np.mean(duration))


def detect_r_peaks(
    lead: Iterable[float] | np.ndarray,
    sampling_rate: int = DEFAULT_SAMPLING_RATE,
    prefer_neurokit: bool = True,
) -> np.ndarray:
    """Detect R-peaks in a single ECG lead.

    NeuroKit2 is used when available because it is designed for ECG signals.
    A SciPy fallback keeps the experiment runnable in minimal environments.
    """

    lead_arr = np.asarray(lead, dtype=np.float64)
    if prefer_neurokit:
        try:
            peaks = _detect_r_peaks_neurokit(lead_arr, sampling_rate)
            if len(peaks) > 0:
                return peaks
        except Exception:
            pass
    return _detect_r_peaks_scipy(lead_arr, sampling_rate)


def extract_rhythm_features(
    ecg: np.ndarray,
    sampling_rate: int = DEFAULT_SAMPLING_RATE,
    lead_index: int = DEFAULT_RHYTHM_LEAD_INDEX,
) -> dict[str, float]:
    """Extract rhythm and HRV-like features from one rhythm lead."""

    ecg = np.asarray(ecg, dtype=np.float64)
    if ecg.ndim != 2 or ecg.shape[1] <= lead_index:
        raise ValueError("ecg must be a 2D array with the requested lead")

    lead = ecg[:, lead_index]
    features = _empty_rhythm_features()
    r_peaks = detect_r_peaks(lead, sampling_rate=sampling_rate)
    features["clinical_r_peak_count"] = float(len(r_peaks))

    if len(r_peaks) > 0:
        r_amps = lead[r_peaks]
        features["clinical_r_amp_mean"] = _as_float(np.mean(r_amps))
        features["clinical_r_amp_std"] = _as_float(np.std(r_amps))
        features["clinical_r_amp_min"] = _as_float(np.min(r_amps))
        features["clinical_r_amp_max"] = _as_float(np.max(r_amps))
        features["clinical_r_amp_range"] = _as_float(np.max(r_amps) - np.min(r_amps))

    if len(r_peaks) < 2:
        return features

    rr_ms = np.diff(r_peaks) / sampling_rate * 1000.0
    heart_rate = 60000.0 / np.maximum(rr_ms, _EPS)
    rr_diff = np.diff(rr_ms)

    features.update(
        {
            "clinical_rr_mean_ms": _as_float(np.mean(rr_ms)),
            "clinical_rr_std_ms": _as_float(np.std(rr_ms)),
            "clinical_rr_min_ms": _as_float(np.min(rr_ms)),
            "clinical_rr_max_ms": _as_float(np.max(rr_ms)),
            "clinical_heart_rate_mean_bpm": _as_float(np.mean(heart_rate)),
            "clinical_heart_rate_std_bpm": _as_float(np.std(heart_rate)),
            "clinical_sdnn_ms": _as_float(np.std(rr_ms, ddof=1)) if len(rr_ms) > 1 else 0.0,
            "clinical_rmssd_ms": _as_float(np.sqrt(np.mean(rr_diff**2))) if len(rr_diff) > 0 else 0.0,
            "clinical_pnn50": _as_float(np.mean(np.abs(rr_diff) > 50.0)) if len(rr_diff) > 0 else 0.0,
        }
    )
    return features


def extract_qrs_voltage_features(
    ecg: np.ndarray,
    r_peaks: np.ndarray,
    sampling_rate: int = DEFAULT_SAMPLING_RATE,
) -> dict[str, float]:
    """Approximate QRS voltage features around detected R-peaks.

    The assumed 12-lead order is I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6.
    """

    ecg = np.asarray(ecg, dtype=np.float64)
    features = _empty_qrs_voltage_features()
    if ecg.ndim != 2 or ecg.shape[1] < 12 or len(r_peaks) == 0:
        return features

    half_window = max(1, int(0.04 * sampling_rate))
    per_lead_max: list[np.ndarray] = []
    per_lead_min: list[np.ndarray] = []

    for lead_idx in range(12):
        maxima = []
        minima = []
        for peak in r_peaks:
            start = max(0, int(peak) - half_window)
            end = min(ecg.shape[0], int(peak) + half_window + 1)
            segment = ecg[start:end, lead_idx]
            if len(segment) == 0:
                continue
            maxima.append(float(np.max(segment)))
            minima.append(float(np.min(segment)))

        name = _lead_name(lead_idx)
        if maxima and minima:
            max_arr = np.asarray(maxima)
            min_arr = np.asarray(minima)
            per_lead_max.append(max_arr)
            per_lead_min.append(min_arr)
            features[f"clinical_{name}_qrs_max_mean"] = _as_float(np.mean(max_arr))
            features[f"clinical_{name}_qrs_min_mean"] = _as_float(np.mean(min_arr))
            features[f"clinical_{name}_qrs_ptp_mean"] = _as_float(np.mean(max_arr - min_arr))

    if len(per_lead_max) == 12 and len(per_lead_min) == 12:
        qrs_ptp = [np.mean(mx - mn) for mx, mn in zip(per_lead_max, per_lead_min, strict=True)]
        features["clinical_qrs_global_ptp_mean"] = _as_float(np.mean(qrs_ptp))

        v1_s = abs(np.mean(per_lead_min[6]))
        v5_r = np.mean(per_lead_max[10])
        v6_r = np.mean(per_lead_max[11])
        avl_r = np.mean(per_lead_max[4])
        v3_s = abs(np.mean(per_lead_min[8]))
        features["clinical_sokolow_lyon_voltage"] = _as_float(max(v5_r, v6_r) + v1_s)
        features["clinical_cornell_voltage"] = _as_float(avl_r + v3_s)

    return features


def extract_interval_features(
    ecg: np.ndarray,
    r_peaks: np.ndarray,
    sampling_rate: int = DEFAULT_SAMPLING_RATE,
    lead_index: int = DEFAULT_RHYTHM_LEAD_INDEX,
) -> dict[str, float]:
    """Extract PR, QRS, QT/QTc, ST, and T-duration features from a rhythm lead."""

    features = _empty_interval_features()
    if len(r_peaks) < 2:
        return features

    try:
        import neurokit2 as nk
    except ImportError:
        return features

    try:
        lead = np.asarray(ecg[:, lead_index], dtype=np.float64)
        _, waves = nk.ecg_delineate(
            lead,
            rpeaks=r_peaks,
            sampling_rate=sampling_rate,
            method="dwt",
            show=False,
        )
    except Exception:
        return features

    p_onsets = _valid_wave_array(waves, "ECG_P_Onsets")
    p_offsets = _valid_wave_array(waves, "ECG_P_Offsets")
    r_onsets = _valid_wave_array(waves, "ECG_R_Onsets")
    r_offsets = _valid_wave_array(waves, "ECG_R_Offsets")
    t_onsets = _valid_wave_array(waves, "ECG_T_Onsets")
    t_offsets = _valid_wave_array(waves, "ECG_T_Offsets")

    rr_s = np.diff(r_peaks) / sampling_rate
    rr_mean_s = float(np.mean(rr_s)) if len(rr_s) > 0 else 0.0
    qt_ms = _mean_interval_ms(r_onsets, t_offsets, sampling_rate)

    features.update(
        {
            "clinical_delineated_beat_count": float(len(r_onsets)),
            "clinical_pr_interval_mean_ms": _mean_interval_ms(p_onsets, r_onsets, sampling_rate),
            "clinical_p_duration_mean_ms": _mean_interval_ms(p_onsets, p_offsets, sampling_rate),
            "clinical_qrs_duration_mean_ms": _mean_interval_ms(r_onsets, r_offsets, sampling_rate),
            "clinical_qt_interval_mean_ms": qt_ms,
            "clinical_st_segment_mean_ms": _mean_interval_ms(r_offsets, t_onsets, sampling_rate),
            "clinical_t_duration_mean_ms": _mean_interval_ms(t_onsets, t_offsets, sampling_rate),
        }
    )

    if qt_ms > 0 and rr_mean_s > 0:
        features["clinical_qtc_bazett_mean_ms"] = _as_float(qt_ms / np.sqrt(rr_mean_s))
        features["clinical_qtc_fridericia_mean_ms"] = _as_float(qt_ms / np.cbrt(rr_mean_s))

    return features


def extract_wave_morphology_features(
    ecg: np.ndarray,
    r_peaks: np.ndarray,
    sampling_rate: int = DEFAULT_SAMPLING_RATE,
) -> dict[str, float]:
    """Approximate Q-wave, ST-segment, and T-wave morphology features per lead."""

    ecg = np.asarray(ecg, dtype=np.float64)
    features = _empty_wave_morphology_features()
    if ecg.ndim != 2 or ecg.shape[1] < 12 or len(r_peaks) == 0:
        return features

    q_start = int(0.06 * sampling_rate)
    q_end = int(0.01 * sampling_rate)
    st_60 = int(0.06 * sampling_rate)
    st_80 = int(0.08 * sampling_rate)
    st_120 = int(0.12 * sampling_rate)
    t_start = int(0.16 * sampling_rate)
    t_end = int(0.36 * sampling_rate)

    global_q_depths = []
    global_st_devs = []
    global_t_amps = []
    global_t_inversions = []

    for lead_idx in range(12):
        lead = ecg[:, lead_idx]
        q_depths = []
        q_widths_ms = []
        q_r_ratios = []
        pathologic_q_flags = []
        st_devs = []
        st_slopes = []
        t_amps = []
        t_inversions = []

        for peak in r_peaks:
            peak = int(peak)
            if peak <= q_start or peak + t_end >= len(lead):
                continue

            baseline = float(np.median(lead[max(0, peak - int(0.2 * sampling_rate)): peak - q_start]))
            r_amp = float(lead[peak] - baseline)
            q_segment = lead[peak - q_start: peak - q_end] - baseline
            st_60_value = float(lead[peak + st_60] - baseline)
            st_80_value = float(lead[peak + st_80] - baseline)
            st_120_value = float(lead[peak + st_120] - baseline)
            t_segment = lead[peak + t_start: peak + t_end] - baseline

            if len(q_segment) > 0:
                q_depth = abs(float(np.min(q_segment)))
                q_width_ms = float(np.sum(q_segment < -0.02) / sampling_rate * 1000.0)
                q_depths.append(q_depth)
                q_widths_ms.append(q_width_ms)
                q_r_ratios.append(q_depth / (abs(r_amp) + _EPS))
                pathologic_q_flags.append(float(q_width_ms >= 30.0 and q_depth / (abs(r_amp) + _EPS) >= 0.25))

            st_devs.append(st_80_value)
            st_slopes.append((st_120_value - st_60_value) / 0.06)

            if len(t_segment) > 0:
                t_peak = float(t_segment[np.argmax(np.abs(t_segment))])
                t_amps.append(t_peak)
                t_inversions.append(float(t_peak < 0))

        name = _lead_name(lead_idx)
        if q_depths:
            features[f"clinical_{name}_q_depth_mean"] = _as_float(np.mean(q_depths))
            features[f"clinical_{name}_q_width_mean_ms"] = _as_float(np.mean(q_widths_ms))
            features[f"clinical_{name}_q_r_ratio_mean"] = _as_float(np.mean(q_r_ratios))
            features[f"clinical_{name}_pathologic_q_ratio"] = _as_float(np.mean(pathologic_q_flags))
            global_q_depths.extend(q_depths)

        if st_devs:
            features[f"clinical_{name}_st_deviation_mean"] = _as_float(np.mean(st_devs))
            features[f"clinical_{name}_st_slope_mean"] = _as_float(np.mean(st_slopes))
            global_st_devs.extend(st_devs)

        if t_amps:
            features[f"clinical_{name}_t_amp_mean"] = _as_float(np.mean(t_amps))
            features[f"clinical_{name}_t_inversion_ratio"] = _as_float(np.mean(t_inversions))
            global_t_amps.extend(t_amps)
            global_t_inversions.extend(t_inversions)

    if global_q_depths:
        features["clinical_q_depth_global_mean"] = _as_float(np.mean(global_q_depths))
    if global_st_devs:
        features["clinical_st_deviation_global_mean"] = _as_float(np.mean(global_st_devs))
    if global_t_amps:
        features["clinical_t_amp_global_mean"] = _as_float(np.mean(global_t_amps))
    if global_t_inversions:
        features["clinical_t_inversion_global_ratio"] = _as_float(np.mean(global_t_inversions))

    return features


def extract_clinical_ecg_features(
    ecg: np.ndarray,
    sampling_rate: int = DEFAULT_SAMPLING_RATE,
    rhythm_lead_index: int = DEFAULT_RHYTHM_LEAD_INDEX,
) -> dict[str, float]:
    """Extract experimental rhythm, interval, morphology, and QRS-voltage features."""

    ecg = np.asarray(ecg, dtype=np.float64)
    rhythm_features = extract_rhythm_features(
        ecg,
        sampling_rate=sampling_rate,
        lead_index=rhythm_lead_index,
    )
    r_peaks = detect_r_peaks(ecg[:, rhythm_lead_index], sampling_rate=sampling_rate)
    qrs_features = extract_qrs_voltage_features(ecg, r_peaks, sampling_rate=sampling_rate)
    interval_features = extract_interval_features(
        ecg,
        r_peaks,
        sampling_rate=sampling_rate,
        lead_index=rhythm_lead_index,
    )
    wave_features = extract_wave_morphology_features(ecg, r_peaks, sampling_rate=sampling_rate)
    return {**rhythm_features, **qrs_features, **interval_features, **wave_features}
