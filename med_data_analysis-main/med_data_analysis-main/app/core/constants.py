"""
Константы приложения.
"""

# маппинг индексов классов на названия
CLASS_MAPPING = {
    0: "N",      # Normal ECG
    1: "S",      # ST/T Change
    2: "MS",     # Myocardial Infarction + ST/T Change
    3: "C",      # Conduction Disturbance
    4: "MC",     # Myocardial Infarction + Conduction Disturbance
    5: "CS",     # Conduction Disturbance + ST/T Change
    6: "M",      # Myocardial Infarction
    7: "H",      # Hypertrophy
    8: "MHS",    # Myocardial Infarction + Hypertrophy + ST/T Change
    9: "MHCS",   # Myocardial Infarction + Hypertrophy + Conduction Disturbance + ST/T Change
    10: "HC",    # Hypertrophy + Conduction Disturbance
    11: "MCS",   # Myocardial Infarction + Conduction Disturbance + ST/T Change
    12: "HS",    # Hypertrophy + ST/T Change
    13: "HCS",   # Hypertrophy + Conduction Disturbance + ST/T Change
    14: "MH",    # Myocardial Infarction + Hypertrophy
    15: "MHC",   # Myocardial Infarction + Hypertrophy + Conduction Disturbance
}

CLASS_DESCRIPTIONS = {
    "N": "Normal ECG",
    "S": "ST/T Change",
    "MS": "Myocardial Infarction + ST/T Change",
    "C": "Conduction Disturbance",
    "MC": "Myocardial Infarction + Conduction Disturbance",
    "CS": "Conduction Disturbance + ST/T Change",
    "M": "Myocardial Infarction",
    "H": "Hypertrophy",
    "MHS": "Myocardial Infarction + Hypertrophy + ST/T Change",
    "MHCS": "Myocardial Infarction + Hypertrophy + Conduction Disturbance + ST/T Change",
    "HC": "Hypertrophy + Conduction Disturbance",
    "MCS": "Myocardial Infarction + Conduction Disturbance + ST/T Change",
    "HS": "Hypertrophy + ST/T Change",
    "HCS": "Hypertrophy + Conduction Disturbance + ST/T Change",
    "MH": "Myocardial Infarction + Hypertrophy",
    "MHC": "Myocardial Infarction + Hypertrophy + Conduction Disturbance",
}

# допустимые значения heart_axis на основе имеющихся данных
VALID_HEART_AXIS = [
    "ALAD",
    "ARAD",
    "AXL",
    "AXR",
    "LAD",
    "MID",
    "RAD",
    "SAG",
    "NO_DATA",
]


# список признаков, которые ожидает модель
BASE_FEATURES = ["age", "sex", "height", "weight"]

# Heart axis признаки (One Hot Encoded)
HEART_AXIS_FEATURES = [
    "heart_axis_ALAD",
    "heart_axis_ARAD",
    "heart_axis_AXL",
    "heart_axis_AXR",
    "heart_axis_LAD",
    "heart_axis_MID",
    "heart_axis_NO_DATA",
    "heart_axis_RAD",
    "heart_axis_SAG",
]

# ECG признаки для каждого из 12 отведений
ECG_FEATURE_SUFFIXES = [
    "mean", "median", "std", "min", "max", "range", "rms",
    "skew", "kurt", "energy", "norm_energy",
    "dom_freq", "spec_entropy", "lf_energy", "hf_energy", "lf_hf_ratio"
]

# генерируем список ECG признаков для 12 отведений
ECG_FEATURES = [
    f"lead{i}_{suffix}"
    for i in range(1, 13)
    for suffix in ECG_FEATURE_SUFFIXES
]

# полный список признаков
ALL_FEATURES = BASE_FEATURES + HEART_AXIS_FEATURES + ECG_FEATURES
