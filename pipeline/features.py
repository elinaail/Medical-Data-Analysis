"""Модуль извлечения признаков из ЭКГ-сигналов PTB-XL.
"""

import logging
import multiprocessing as mp
import pickle
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis, skew
from tqdm import tqdm

logger = logging.getLogger(__name__)


# константы — группы признаков
_BASE_FEATS: list[str] = ["age", "sex", "height", "weight", "bmi"]
_HEART_AXIS_PREFIX: str = "heart_axis_"
_ECG_LEAD_PREFIX: str = "lead"
_ECG_CORR_PREFIX: str = "corr_"
_ECG_GLOBAL_PREFIX: str = "global_"

_CROSS_LEAD_PAIRS: list[tuple[int, int]] = [(0, 1), (0, 2), (1, 2), (3, 4), (6, 10), (7, 11)]
_CROSS_LEAD_NAMES: list[str] = ["I_II", "I_III", "II_III", "aVR_aVL", "V1_V5", "V2_V6"]


class FeatureCols(NamedTuple):
    """Именованный кортеж с группами столбцов признаков."""

    base: list[str]
    heart_axis: list[str]
    ecg: list[str]
    all: list[str]


def extract_ecg_features(ecg: np.ndarray) -> dict:
    """Извлекает признаки из одной ЭКГ-записи (5000×12).

    Извлекаются следующие группы признаков на каждое из 12 отведений:
    - Базовые статистики: mean, median, std, min, max, range, rms
    - Форма: skew, kurtosis
    - Энергия: energy, norm_energy
    - Частотные (Welch PSD): dom_freq, spec_entropy, lf_energy, hf_energy,
      lf_hf_ratio, hf2_energy, lf_rel, hf_rel
    - Квартили: p25, p75, iqr
    - Робастные характеристики: mad, zcr, mean_abs_diff, std_diff,
      max_abs_diff, autocorr1, cv
    - Межотводные корреляции: 6 физиологически значимых пар
    - Глобальные: global_mean_std, global_max_range, global_energy_var,
      global_energy_total

    Parameters
    ----------
    ecg : np.ndarray
        Матрица сигнала формы (n_samples, 12).

    Returns
    -------
    dict
        Словарь {название_признака: значение}.
    """
    features: dict = {}
    num_leads = ecg.shape[1]
    all_stds: list[float] = []
    all_ranges: list[float] = []
    all_energies: list[float] = []

    for lead_idx in range(num_leads):
        lead = ecg[:, lead_idx].astype(np.float64)
        p = f"lead{lead_idx + 1}"

        # Базовые статистики
        features[f"{p}_mean"] = float(np.mean(lead))
        features[f"{p}_median"] = float(np.median(lead))
        features[f"{p}_std"] = float(np.std(lead))
        features[f"{p}_min"] = float(np.min(lead))
        features[f"{p}_max"] = float(np.max(lead))
        features[f"{p}_range"] = float(np.max(lead) - np.min(lead))
        features[f"{p}_rms"] = float(np.sqrt(np.mean(lead**2)))

        # Форма распределения
        features[f"{p}_skew"] = float(skew(lead))
        features[f"{p}_kurt"] = float(kurtosis(lead))

        # Энергия
        energy = float(np.sum(lead**2))
        features[f"{p}_energy"] = energy
        features[f"{p}_norm_energy"] = energy / len(lead)

        # Частотные (Welch PSD)
        freqs, psd = welch(lead, fs=500)
        psd_norm = psd / (np.sum(psd) + 1e-12)
        features[f"{p}_dom_freq"] = float(freqs[np.argmax(psd)])
        features[f"{p}_spec_entropy"] = float(-np.sum(psd_norm * np.log(psd_norm + 1e-12)))

        lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
        hf_mask = (freqs >= 0.15) & (freqs <= 0.40)
        hf2_mask = (freqs >= 0.4) & (freqs <= 2.0)
        lf_e = float(np.sum(psd[lf_mask]))
        hf_e = float(np.sum(psd[hf_mask]))
        total_e = float(np.sum(psd)) + 1e-12
        features[f"{p}_lf_energy"] = lf_e
        features[f"{p}_hf_energy"] = hf_e
        features[f"{p}_lf_hf_ratio"] = lf_e / (hf_e + 1e-6)
        features[f"{p}_hf2_energy"] = float(np.sum(psd[hf2_mask]))
        features[f"{p}_lf_rel"] = lf_e / total_e
        features[f"{p}_hf_rel"] = hf_e / total_e

        # Квартили и IQR
        p25, p75 = float(np.percentile(lead, 25)), float(np.percentile(lead, 75))
        features[f"{p}_p25"] = p25
        features[f"{p}_p75"] = p75
        features[f"{p}_iqr"] = p75 - p25

        # Робастные характеристики
        features[f"{p}_mad"] = float(np.mean(np.abs(lead - np.mean(lead))))

        signs = np.sign(lead)
        signs[signs == 0] = 1
        features[f"{p}_zcr"] = float(np.sum(np.diff(signs) != 0) / len(lead))

        diff = np.diff(lead)
        features[f"{p}_mean_abs_diff"] = float(np.mean(np.abs(diff)))
        features[f"{p}_std_diff"] = float(np.std(diff))
        features[f"{p}_max_abs_diff"] = float(np.max(np.abs(diff)))

        if np.std(lead) > 1e-6:
            features[f"{p}_autocorr1"] = float(np.corrcoef(lead[:-1], lead[1:])[0, 1])
        else:
            features[f"{p}_autocorr1"] = 0.0

        features[f"{p}_cv"] = float(np.std(lead) / (np.abs(np.mean(lead)) + 1e-6))

        all_stds.append(float(np.std(lead)))
        all_ranges.append(float(np.max(lead) - np.min(lead)))
        all_energies.append(energy)

    # Межотводные корреляции
    for (i, j), name in zip(_CROSS_LEAD_PAIRS, _CROSS_LEAD_NAMES, strict=True):
        li = ecg[:, i].astype(np.float64)
        lj = ecg[:, j].astype(np.float64)
        if np.std(li) > 1e-6 and np.std(lj) > 1e-6:
            features[f"corr_{name}"] = float(np.corrcoef(li, lj)[0, 1])
        else:
            features[f"corr_{name}"] = 0.0

    # Глобальные признаки уровня записи
    features["global_mean_std"] = float(np.mean(all_stds))
    features["global_max_range"] = float(np.max(all_ranges))
    features["global_energy_var"] = float(np.var(all_energies))
    features["global_energy_total"] = float(np.sum(all_energies))

    return features


def generate_features(
    df: pd.DataFrame,
    n_jobs: int = -1,
    cache_path: str | Path | None = "datasets/features_cache.pkl",
) -> tuple[pd.DataFrame, FeatureCols]:
    """Сформировывает полный набор признаков для честного сравнения с CatBoost.

    Набор признаков идентичен ml_experiments.ipynb:

    - **Демографические** (base): age, sex, height, weight, bmi
    - **Ось сердца** (heart_axis): OHE-столбцы ``heart_axis_*``
    - **ЭКГ-сигналы** (ecg): статистические, частотные, межотводные признаки
      на каждое из 12 отведений + глобальные (итого ~234 признака)

    Параметры OHE: те же, что в ml_experiments.ipynb (pd.get_dummies, без drop_first).
    Признак bmi = weight / (height / 100) ** 2.

    При задании ``cache_path`` результат сохраняется в pickle-файл после первого
    вычисления и загружается из него при последующих запусках, что позволяет
    избежать повторного вычисления дорогостоящих ЭКГ-признаков.

    Parameters
    ----------
    df : pd.DataFrame
        Предобработанный датафрейм из ``preprocess()`` (содержит ``ecg_signals``).
    n_jobs : int
        Число параллельных процессов для извлечения ЭКГ-признаков.
        -1 — использовать все доступные ядра.
    cache_path : str | Path | None
        Путь к pickle-файлу кэша. ``None`` — кэширование отключено.

    Returns
    -------
    tuple[pd.DataFrame, FeatureCols]
        - Датафрейм с добавленными признаками (без столбца ``ecg_signals``).
        - :class:`FeatureCols` с группами столбцов (base, heart_axis, ecg, all).
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            logger.info("Загрузка признаков из кэша: %s", cache_path)
            with cache_path.open("rb") as fh:
                return pickle.load(fh)  # noqa: S301

    logger.info("Извлечение ЭКГ-признаков, записей: %d", len(df))

    signals = df["ecg_signals"].tolist()
    n_workers = mp.cpu_count() if n_jobs == -1 else max(1, n_jobs)

    with mp.Pool(processes=n_workers) as pool:
        ecg_records = list(
            tqdm(
                pool.imap(extract_ecg_features, signals, chunksize=32),
                total=len(signals),
                desc=f"Извлечение ЭКГ-признаков ({n_workers} воркеров)",
            )
        )

    ecg_df = pd.DataFrame(ecg_records)

    # OHE для heart_axis (совместимо с ml_experiments.ipynb)
    result = pd.get_dummies(
        df.drop(columns=["ecg_signals"]).reset_index(drop=True),
        columns=["heart_axis"],
        prefix="heart_axis",
    )
    result = pd.concat([result, ecg_df.reset_index(drop=True)], axis=1)

    # BMI
    result["bmi"] = result["weight"] / ((result["height"] / 100) ** 2)

    heart_feats = sorted(c for c in result.columns if c.startswith(_HEART_AXIS_PREFIX))
    ecg_feats = sorted(
        c
        for c in result.columns
        if c.startswith(_ECG_LEAD_PREFIX)
        or c.startswith(_ECG_CORR_PREFIX)
        or c.startswith(_ECG_GLOBAL_PREFIX)
    )
    all_feats = _BASE_FEATS + heart_feats + ecg_feats

    # приводим bool-столбцы (get_dummies) к int для NumPy-совместимости
    bool_cols = [c for c in heart_feats if result[c].dtype == bool]
    if bool_cols:
        result[bool_cols] = result[bool_cols].astype(int)

    logger.info(
        "Матрица признаков: %d строк × %d признаков (base=%d, heart_axis=%d, ecg=%d)",
        len(result),
        len(all_feats),
        len(_BASE_FEATS),
        len(heart_feats),
        len(ecg_feats),
    )

    feat_cols = FeatureCols(
        base=_BASE_FEATS,
        heart_axis=heart_feats,
        ecg=ecg_feats,
        all=all_feats,
    )

    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as fh:
            pickle.dump((result, feat_cols), fh)
        logger.info("Признаки сохранены в кэш: %s", cache_path)

    return result, feat_cols
