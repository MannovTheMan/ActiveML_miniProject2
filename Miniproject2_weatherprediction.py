"""
DMI Copenhagen climate-data pipeline with QBC (Query by Committee)
------------------------------------------------------------------
Finder den bedste station i København-området for 4 variable,
bygger et clean datasæt med 4 målinger pr. dag (00, 06, 12, 18 lokal tid),
og træner en første QBC-baseret model.

Krav:
    pip install requests pandas numpy scikit-learn matplotlib

Bemærk:
- API: DMI Climate Data API
- Data: stationValue
- Time resolution: hour
- Lokaltid: Europe/Copenhagen

QBC = Query by Committee
------------------------
Idéen er:
1. Start med et lille labellet datasæt
2. Træn flere forskellige regressionsmodeller
3. Find de punkter hvor modellerne er mest uenige
4. Tilføj disse punkter til det labellede datasæt
5. Gentag

I dette script simuleres active learning, fordi vi allerede har labels
fra DMI-dataene.
"""

from __future__ import annotations

import os
import sys
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter issues
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ============================================================
# Logging (console output saved in timestamped log file)
# ============================================================

os.makedirs("logs", exist_ok=True)
timestamp = datetime.now().strftime("%d-%b-%Y_%H-%M-%S")

main_log = f"logs/output_{timestamp}.log"
debug_log = f"logs/debug_{timestamp}.log"

# Main logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(message)s")

# Main file handler
file_handler = logging.FileHandler(main_log, encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# ============================================================
# Dedicated debug-only logger (EMPTY unless you use it)
# ============================================================

debug_logger = logging.getLogger("debug_only")
debug_logger.setLevel(logging.DEBUG)
debug_logger.propagate = False  # prevent leaking into root logger

debug_handler = logging.FileHandler(debug_log, encoding="utf-8")
debug_handler.setLevel(logging.DEBUG)
debug_handler.setFormatter(formatter)

debug_logger.addHandler(debug_handler)




# ============================================================
# Konfiguration
# ============================================================
os.makedirs("plots", exist_ok=True)
os.makedirs("data", exist_ok=True)

BASE_URL = "https://opendataapi.dmi.dk/v2/climateData"
COPENHAGEN_TZ = "Europe/Copenhagen"

# Cirka centrum af København
COPENHAGEN_CENTER_LAT = 55.6761
COPENHAGEN_CENTER_LON = 12.5683

# Bounding box omkring København
BBOX = "12.35,55.55,12.80,55.80"

# Fire tidspunkter pr. dag, lokal tid
TARGET_HOURS = {0, 6, 12, 18}

# Hvor mange dage tilbage der hentes data
DAYS_BACK = 365

# DMI-parametre -> outputkolonnenavne
PARAMETERS: Dict[str, str] = {
    "mean_temp": "temperature_c",
    "mean_relative_hum": "relative_humidity_pct",
    "acc_precip": "precip_mm",
    "mean_pressure": "pressure_hpa",
}

# Pagination
PAGE_LIMIT = 10000

# Hvor mange stationer der evalueres
MAX_STATION_CANDIDATES = 20

# Sæt True hvis I kun vil bruge manuelt QC'ede data
MANUAL_QC_ONLY = False

# Standardization factors for normalized error metrics
# Used to normalize each variable's error by its typical range/variability
STANDARDIZATION_FACTORS: Dict[str, float] = {
    "temperature_c": 5.0,           # °C (typical std dev)
    "relative_humidity_pct": 50.0,  # % (half of 0-100 range)
    "pressure_hpa": 10.0,           # hPa (typical daily variation)
    "precip_mm": 2.0,               # mm (threshold for significant rain)
}

# QBC and Model Configuration
# ============================================================
# Active Learning Parameters
QBC_INITIAL_LABELED_SIZE = 80           # Number of samples to start with
QBC_QUERY_BATCH_SIZE = 1                # Number of samples to query per iteration
QBC_N_QUERIES = 25                      # Number of active learning iterations
QBC_RANDOM_STATE = 42                   # Random seed for reproducibility

# Committee Model Parameters
RF_N_ESTIMATORS = 300                   # Random Forest tree count
RF_MIN_SAMPLES_LEAF = 2                 # Random Forest minimum samples per leaf
ET_N_ESTIMATORS = 300                   # Extra Trees tree count
ET_MIN_SAMPLES_LEAF = 2                 # Extra Trees minimum samples per leaf
RIDGE_ALPHA = 1.0                       # Ridge regression alpha (L2 penalty)


# ============================================================
# Dataklasser
# ============================================================
@dataclass
class StationScore:
    station_id: str
    station_name: str
    distance_km: float
    coverage_score: float
    completeness_ratio: float
    rows_expected: int
    rows_complete_all_vars: int
    rows_by_variable: dict


@dataclass
class QBCResult:
    """
    Resultatobjekt fra QBC-træning.
    """
    final_model: Pipeline
    metrics: Dict[str, float]
    predictions: pd.DataFrame
    learning_curve: pd.DataFrame
    final_committee_disagreement: pd.DataFrame
    final_pool_predictions: pd.DataFrame
    final_selected_points: pd.DataFrame


# ============================================================
# Hjælpefunktioner
# ============================================================
def get_json(url: str, params: dict | None = None, retries: int = 3, sleep_s: float = 1.5) -> dict:
    """
    Simpelt helper-kald til DMI API med retries.
    """
    last_error = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=90)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_error = e
            if attempt < retries - 1:
                time.sleep(sleep_s * (attempt + 1))
    raise RuntimeError(f"API-kald fejlede: {url} params={params}") from last_error


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Beregner afstand i kilometer mellem to GPS-koordinater.
    """
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def iso_interval_last_n_days(days_back: int = DAYS_BACK) -> str:
    """
    Bygger et ISO-tidsinterval for de seneste N dage.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)
    return f"{start.strftime('%Y-%m-%dT00:00:00Z')}/{end.strftime('%Y-%m-%dT23:59:59Z')}"


def expected_rows_last_year(days_back: int = DAYS_BACK) -> int:
    """
    Omtrent forventet antal rækker ved 4 målinger pr. dag.
    """
    return days_back * 4


# ============================================================
# API-lag
# ============================================================
def get_station_candidates(bbox: str = BBOX) -> pd.DataFrame:
    """
    Henter aktive stationer i bbox omkring København.
    """
    url = f"{BASE_URL}/collections/station/items"
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    params = {
        "bbox": bbox,
        "status": "Active",
        "datetime": f"{now_utc}/..",
        "limit": 500,
    }

    data = get_json(url, params=params)
    rows = []

    for feat in data.get("features", []):
        props = feat.get("properties", {})
        geom = feat.get("geometry", {}) or {}
        coords = geom.get("coordinates", [None, None])

        lon = coords[0] if len(coords) > 0 else None
        lat = coords[1] if len(coords) > 1 else None

        if lat is None or lon is None:
            continue

        dist = haversine_km(COPENHAGEN_CENTER_LAT, COPENHAGEN_CENTER_LON, lat, lon)

        rows.append(
            {
                "stationId": props.get("stationId"),
                "name": props.get("name"),
                "status": props.get("status"),
                "type": props.get("type"),
                "lat": lat,
                "lon": lon,
                "validFrom": props.get("validFrom"),
                "validTo": props.get("validTo"),
                "distance_km": dist,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Ingen aktive stationer fundet i København-bbox.")

    return df.sort_values(["distance_km", "stationId"]).reset_index(drop=True)


def fetch_station_parameter(
    station_id: str,
    parameter_id: str,
    days_back: int = DAYS_BACK,
    time_resolution: str = "hour",
    manual_qc_only: bool = MANUAL_QC_ONLY,
) -> pd.DataFrame:
    """
    Henter stationValue-data for én station og ét parameter.
    """
    url = f"{BASE_URL}/collections/stationValue/items"
    interval = iso_interval_last_n_days(days_back)

    offset = 0
    rows = []

    while True:
        params = {
            "stationId": station_id,
            "parameterId": parameter_id,
            "timeResolution": time_resolution,
            "datetime": interval,
            "limit": PAGE_LIMIT,
            "offset": offset,
        }

        if manual_qc_only:
            params["qcStatus"] = "manual"

        data = get_json(url, params=params)
        feats = data.get("features", [])

        if not feats:
            break

        for feat in feats:
            props = feat.get("properties", {})
            rows.append(
                {
                    "stationId": props.get("stationId"),
                    "parameterId": props.get("parameterId"),
                    "from": props.get("from"),
                    "to": props.get("to"),
                    "value": props.get("value"),
                    "timeResolution": props.get("timeResolution"),
                    "qcStatus": props.get("qcStatus"),
                    "validity": props.get("validity"),
                }
            )

        if len(feats) < PAGE_LIMIT:
            break

        offset += PAGE_LIMIT

    return pd.DataFrame(rows)


def prepare_four_times_daily(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Konverterer timestamps til lokal tid og filtrerer til:
    00, 06, 12, 18 lokal tid.
    """
    if df.empty:
        return pd.DataFrame(columns=["timestamp_local", "date_local", "hour_local", value_col])

    out = df.copy()
    out["from"] = pd.to_datetime(out["from"], utc=True, errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["from", "value"])

    if "timeResolution" in out.columns:
        out = out[out["timeResolution"].astype(str).eq("hour")]

    out["timestamp_local"] = out["from"].dt.tz_convert(COPENHAGEN_TZ)
    out["hour_local"] = out["timestamp_local"].dt.hour
    out = out[out["hour_local"].isin(TARGET_HOURS)].copy()

    out["date_local"] = out["timestamp_local"].dt.date

    out = (
        out[["timestamp_local", "date_local", "hour_local", "value"]]
        .drop_duplicates(subset=["timestamp_local"], keep="last")
        .sort_values("timestamp_local")
        .rename(columns={"value": value_col})
        .reset_index(drop=True)
    )

    return out


# ============================================================
# Stationscoring
# ============================================================
def merge_parameter_tables(tables: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Fletter flere parameter-tabeller på fælles timestamp.
    """
    if not tables:
        return pd.DataFrame()

    merged = tables[0].copy()
    for t in tables[1:]:
        merged = merged.merge(
            t,
            on=["timestamp_local", "date_local", "hour_local"],
            how="outer",
        )

    return merged.sort_values("timestamp_local").reset_index(drop=True)


def evaluate_station(station_row: pd.Series) -> Tuple[StationScore, pd.DataFrame]:
    """
    Henter de fire parametre, bygger et merged dataset og scorer stationen.
    """
    station_id = str(station_row["stationId"])
    prepared_tables = []
    rows_by_variable = {}

    for parameter_id, output_col in PARAMETERS.items():
        raw = fetch_station_parameter(station_id, parameter_id)
        tidy = prepare_four_times_daily(raw, output_col)
        rows_by_variable[output_col] = len(tidy)
        prepared_tables.append(tidy)

    merged = merge_parameter_tables(prepared_tables)

    expected = expected_rows_last_year(DAYS_BACK)
    all_vars = list(PARAMETERS.values())

    if merged.empty:
        complete_rows = 0
    else:
        complete_rows = int(merged[all_vars].notna().all(axis=1).sum())

    completeness_ratio = complete_rows / expected if expected else 0.0

    # Score = mest vægt på datadækning, lidt vægt på nærhed
    distance_penalty = min(float(station_row["distance_km"]) / 25.0, 1.0)
    coverage_score = 0.85 * completeness_ratio + 0.15 * (1.0 - distance_penalty)

    score = StationScore(
        station_id=station_id,
        station_name=str(station_row.get("name", "")),
        distance_km=float(station_row["distance_km"]),
        coverage_score=coverage_score,
        completeness_ratio=completeness_ratio,
        rows_expected=expected,
        rows_complete_all_vars=complete_rows,
        rows_by_variable=rows_by_variable,
    )
    return score, merged


def find_best_station() -> Tuple[StationScore, pd.DataFrame, pd.DataFrame]:
    """
    Finder bedste station blandt kandidater i København-området.
    """
    candidates = get_station_candidates().head(MAX_STATION_CANDIDATES).copy()

    scores = []
    best_score = None
    best_dataset = None

    for _, row in candidates.iterrows():
        try:
            score, merged = evaluate_station(row)

            scores.append(
                {
                    "stationId": score.station_id,
                    "name": score.station_name,
                    "distance_km": round(score.distance_km, 2),
                    "coverage_score": round(score.coverage_score, 4),
                    "completeness_ratio": round(score.completeness_ratio, 4),
                    "rows_expected": score.rows_expected,
                    "rows_complete_all_vars": score.rows_complete_all_vars,
                    **score.rows_by_variable,
                }
            )

            if best_score is None or score.coverage_score > best_score.coverage_score:
                best_score = score
                best_dataset = merged

        except Exception as e:
            scores.append(
                {
                    "stationId": row["stationId"],
                    "name": row.get("name", ""),
                    "distance_km": round(float(row["distance_km"]), 2),
                    "coverage_score": np.nan,
                    "completeness_ratio": np.nan,
                    "rows_expected": expected_rows_last_year(DAYS_BACK),
                    "rows_complete_all_vars": 0,
                    "error": str(e),
                }
            )

    score_df = pd.DataFrame(scores).sort_values(
        by=["coverage_score", "completeness_ratio", "distance_km"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)

    if best_score is None or best_dataset is None:
        raise RuntimeError("Kunne ikke finde en brugbar station.")

    return best_score, best_dataset, score_df


# ============================================================
# Databehandling
# ============================================================
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gør datasættet modelklart:
    - bygger en fuld 6-timers tidsakse
    - merger med eksisterende data
    - interpolerer kontinuerte variable på DatetimeIndex
    - fylder nedbør med 0 i stedet for NaN-værdier
    """
    if df.empty:
        raise RuntimeError("Tomt datasæt efter merge.")

    out = df.copy()
    debug_logger.debug(out.to_string() + "\n\n\n\n =========================================================================================================== \n\n\n\n")
    out["timestamp_local"] = pd.to_datetime(out["timestamp_local"], errors="coerce")
    out = out.dropna(subset=["timestamp_local"]).sort_values("timestamp_local")

    start = out["timestamp_local"].min().floor("D").tz_convert("UTC")
    end = out["timestamp_local"].max().ceil("D").tz_convert("UTC")

    
    full_index = (
        pd.date_range(start=start, end=end, freq="6h", tz="UTC")
        .tz_convert("Europe/Copenhagen")
    )


    full = pd.DataFrame({"timestamp_local": full_index})
    full["date_local"] = full["timestamp_local"].dt.date
    full["hour_local"] = full["timestamp_local"].dt.hour
    # full = full[full["hour_local"].isin(sorted(TARGET_HOURS))]

    out = full.merge(
        out,
        on=["timestamp_local", "date_local", "hour_local"],
        how="left",
        suffixes=("", "_orig"),
    )

    keep = ["timestamp_local", "date_local", "hour_local"] + list(PARAMETERS.values())
    out = out[keep].sort_values("timestamp_local").reset_index(drop=True)

    for col in PARAMETERS.values():
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.set_index("timestamp_local").sort_index()

    for col in ["temperature_c", "relative_humidity_pct", "pressure_hpa"]:
        if col in out.columns:
            out[col] = out[col].interpolate(method="time", limit_direction="both")

    if "precip_mm" in out.columns:
        out["precip_mm"] = out["precip_mm"].fillna(0.0)

    out = out.ffill().bfill()
    out = out.reset_index()

    out["date_local"] = out["timestamp_local"].dt.date
    out["hour_local"] = out["timestamp_local"].dt.hour
    debug_logger.debug(out.to_string() + "\n\n\n\n =========================================================================================================== \n\n\n\n")

    return out


def add_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Bygger features og 16 targets for multi-output regression.
    Targets = alle 4 variabler ved 4 tidsskridt (6h, 12h, 24h, 48h ahead).
    
    Returns:
        Tuple of (dataframe_with_features_and_16_targets, list_of_16_target_column_names)
    """
    logging.info("\n⏳ Creating features and 16 targets...")
    out = df.copy()
    original_rows = len(out)
    ts = out["timestamp_local"]

    # Temporal features
    logging.info("  • Adding temporal features (month, weekday, dayofyear, hour_sin, hour_cos, doy_sin, doy_cos)...")
    out["month"] = ts.dt.month
    out["weekday"] = ts.dt.weekday
    out["dayofyear"] = ts.dt.dayofyear

    out["hour_sin"] = np.sin(2 * np.pi * out["hour_local"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour_local"] / 24.0)
    out["doy_sin"] = np.sin(2 * np.pi * out["dayofyear"] / 366.0)
    out["doy_cos"] = np.cos(2 * np.pi * out["dayofyear"] / 366.0)

    # Lagged features for all 4 variables
    logging.info("  • Adding lagged features (lag1, lag2, lag4 for all 4 variables)...")
    base_cols = list(PARAMETERS.values())
    for col in base_cols:
        out[f"{col}_lag1"] = out[col].shift(1)
        out[f"{col}_lag2"] = out[col].shift(2)
        out[f"{col}_lag4"] = out[col].shift(4)

    # Rolling aggregation features
    logging.info("  • Adding rolling aggregation features (temp_roll4_mean, rh_roll4_mean, etc.)...")
    out["temp_roll4_mean"] = out["temperature_c"].shift(1).rolling(4).mean()
    out["rh_roll4_mean"] = out["relative_humidity_pct"].shift(1).rolling(4).mean()
    out["pressure_roll4_mean"] = out["pressure_hpa"].shift(1).rolling(4).mean()
    out["precip_roll4_sum"] = out["precip_mm"].shift(1).rolling(4).sum()

    # Create 16 target columns: 4 variables × 4 time steps
    logging.info("  • Creating 16 targets (4 variables × 4 horizons: h1, h2, h4, h8)...")
    variable_names = list(PARAMETERS.values())
    time_steps = [1, 2, 4, 8]  # 6h, 12h, 24h, 48h ahead
    
    target_cols = []
    for var in variable_names:
        for step in time_steps:
            target_col = f"target_{var}_h{step}"
            out[target_col] = out[var].shift(-step)
            target_cols.append(target_col)

    # Drop rows where ANY target is NaN
    logging.info(f"  • Dropping rows with missing targets...")
    out = out.dropna(subset=target_cols).reset_index(drop=True)
    rows_after = len(out)
    rows_lost = original_rows - rows_after
    pct_lost = (rows_lost / original_rows * 100) if original_rows > 0 else 0
    
    logging.info(f"  ✓ Features created: {len([c for c in out.columns if not c.startswith('target')] + ['timestamp_local'])} features")
    logging.info(f"  ✓ Targets created: {len(target_cols)} targets")
    logging.info(f"  ✓ Sample counts: {original_rows} → {rows_after} rows (lost {rows_lost} rows: {pct_lost:.1f}%)")
    
    return out, target_cols


# ============================================================
# QBC / modellering
# ============================================================
def get_feature_columns() -> List[str]:
    """
    Featurekolonner brugt i temperaturmodellen.
    """
    return [
        "hour_local",
        "month",
        "weekday",
        "dayofyear",
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
        "temperature_c",
        "relative_humidity_pct",
        "precip_mm",
        "pressure_hpa",
        "temperature_c_lag1",
        "temperature_c_lag2",
        "temperature_c_lag4",
        "relative_humidity_pct_lag1",
        "relative_humidity_pct_lag2",
        "relative_humidity_pct_lag4",
        "precip_mm_lag1",
        "precip_mm_lag2",
        "precip_mm_lag4",
        "pressure_hpa_lag1",
        "pressure_hpa_lag2",
        "pressure_hpa_lag4",
        "temp_roll4_mean",
        "rh_roll4_mean",
        "pressure_roll4_mean",
        "precip_roll4_sum",
    ]


def get_target_columns() -> List[str]:
    """
    Returnerer alle 16 target-kolonnerne (4 variabler × 4 tidsskridt).
    """
    variable_names = list(PARAMETERS.values())
    time_steps = [1, 2, 4, 8]
    
    target_cols = []
    for var in variable_names:
        for step in time_steps:
            target_cols.append(f"target_{var}_h{step}")
    return target_cols


def build_committee(random_state: int = QBC_RANDOM_STATE) -> List[Pipeline]:
    """
    Bygger en lille committee af forskellige regressionsmodeller.
    """
    return [
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=RF_N_ESTIMATORS,
                min_samples_leaf=RF_MIN_SAMPLES_LEAF,
                random_state=random_state,
                n_jobs=-1,
            ))
        ]),
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(
                n_estimators=ET_N_ESTIMATORS,
                min_samples_leaf=ET_MIN_SAMPLES_LEAF,
                random_state=random_state + 1,
                n_jobs=-1,
            ))
        ]),
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=RIDGE_ALPHA))
        ]),
    ]


def fit_committee(
    committee: List[Pipeline],
    X_labeled: pd.DataFrame,
    y_labeled: pd.DataFrame | pd.Series,
) -> List[Pipeline]:
    """
    Træner friske kopier af alle modellerne i committee.
    Håndterer både single-output (Series) og multi-output (DataFrame).
    """
    fitted = []
    for model in committee:
        m = clone(model)
        m.fit(X_labeled, y_labeled)
        fitted.append(m)
    return fitted


def disagreement_std(
    committee: List[Pipeline],
    X_pool: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For multi-output regression:
    - Input: X_pool is (n_samples, n_features)
    - Output predictions_matrix shape = (n_models, n_samples, n_outputs=16) for multi-output
    - disagreement shape = (n_samples,) — mean disagreement across all outputs per sample
    """
    preds = []
    for model in committee:
        pred = model.predict(X_pool)
        preds.append(pred)

    # preds[i] shape: (n_samples,) for single-output or (n_samples, n_outputs) for multi-output
    predictions_matrix = np.array(preds)
    
    # For multi-output case: predictions_matrix shape is (n_models, n_samples, n_outputs)
    # For single-output case: predictions_matrix shape is (n_models, n_samples)
    if predictions_matrix.ndim == 3:
        # Multi-output: compute std across models per output, then average across outputs
        disagreement_per_output = predictions_matrix.std(axis=0)  # shape: (n_samples, n_outputs)
        disagreement = disagreement_per_output.mean(axis=1)       # shape: (n_samples,)
    else:
        # Single-output: compute std across models directly
        disagreement = predictions_matrix.std(axis=0)             # shape: (n_samples,)
    
    return predictions_matrix, disagreement


def committee_mean_prediction(
    committee: List[Pipeline],
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Committee-prediction = gennemsnit af model-predictions.
    Håndterer både single-output (1D) og multi-output (2D) predictions.
    """
    preds = [model.predict(X) for model in committee]
    # preds[i] is either 1D (n_samples,) or 2D (n_samples, n_outputs)
    preds_array = np.array(preds)
    
    if preds_array.ndim == 3:
        # Multi-output: preds_array shape is (n_models, n_samples, n_outputs)
        return preds_array.mean(axis=0)  # shape: (n_samples, n_outputs)
    else:
        # Single-output: preds_array shape is (n_models, n_samples) after vstack-like behavior
        return preds_array.mean(axis=0)  # shape: (n_samples,)


def train_qbc_model(
    df_feat: pd.DataFrame,
    initial_labeled_size: int = QBC_INITIAL_LABELED_SIZE,
    query_batch_size: int = QBC_QUERY_BATCH_SIZE,
    n_queries: int = QBC_N_QUERIES,
    random_state: int = QBC_RANDOM_STATE,
) -> QBCResult:
    """
    Træner multi-output QBC model med 16 targets.
    
    Args:
        df_feat: DataFrame with features and 16 target columns
        initial_labeled_size: Initial labeled set size
        query_batch_size: Number of samples to query per iteration
        n_queries: Number of active learning iterations
        random_state: Random seed for reproducibility
        
    Returns:
        QBCResult with multi-output predictions and metrics
    """
    logging.info("\n⏳ Setting up QBC training...")
    feature_cols = get_feature_columns()
    target_cols = get_target_columns()
    
    logging.info(f"  • Features: {len(feature_cols)} features")
    logging.info(f"  • Targets: {len(target_cols)} targets (4 variables × 4 horizons)")

    # Train/test split (80/20)
    split_idx = int(len(df_feat) * 0.8)
    train_df = df_feat.iloc[:split_idx].copy()
    test_df = df_feat.iloc[split_idx:].copy()

    X_train_full = train_df[feature_cols].reset_index(drop=True)
    y_train_full = train_df[target_cols].reset_index(drop=True)  # DataFrame with 16 columns

    X_test = test_df[feature_cols].reset_index(drop=True)
    y_test = test_df[target_cols].reset_index(drop=True)  # DataFrame with 16 columns

    if initial_labeled_size >= len(X_train_full):
        raise ValueError("initial_labeled_size er for stor i forhold til train-datasættet.")

    # Initial split: labeled and pool
    X_labeled = X_train_full.iloc[:initial_labeled_size].copy()
    y_labeled = y_train_full.iloc[:initial_labeled_size].copy()

    X_pool = X_train_full.iloc[initial_labeled_size:].copy()
    y_pool = y_train_full.iloc[initial_labeled_size:].copy()

    ts_pool = train_df.iloc[initial_labeled_size:]["timestamp_local"].reset_index(drop=True)
    ts_test = test_df["timestamp_local"].reset_index(drop=True)

    logging.info(f"\n⏳ Train/test split:")
    logging.info(f"  • Train (labeled + pool): {len(X_train_full)} rows")
    logging.info(f"  • Test: {len(X_test)} rows")
    logging.info(f"  • Labeled (initial): {len(X_labeled)} rows")
    logging.info(f"  • Pool: {len(X_pool)} rows")

    base_committee = build_committee(random_state=random_state)

    learning_curve_rows = []
    last_pool_predictions_df = pd.DataFrame()
    last_selected_points_df = pd.DataFrame()

    logging.info(f"\n⏳ Starting QBC active learning loop ({n_queries} iterations)...")
    
    for step in tqdm(range(n_queries), desc="QBC Iterations", unit="it"):
        if len(X_pool) == 0:
            tqdm.write(f"  ℹ Pool exhausted at iteration {step}")
            break

        # Train committee
        fitted_committee = fit_committee(base_committee, X_labeled, y_labeled)

        # Evaluate on test set - aggregate MAE/RMSE across all 16 targets
        test_pred = committee_mean_prediction(fitted_committee, X_test)
        # For multi-output: y_test and test_pred are both (n_samples, 16)
        mae = mean_absolute_error(y_test.values.flatten(), test_pred.flatten())
        rmse = float(np.sqrt(mean_squared_error(y_test.values.flatten(), test_pred.flatten())))

        learning_curve_rows.append({
            "iteration": step,
            "labeled_size": len(X_labeled),
            "pool_size": len(X_pool),
            "test_mae": mae,
            "test_rmse": rmse,
        })

        # Compute disagreement on pool
        pred_matrix, disagreement = disagreement_std(fitted_committee, X_pool)
        
        # Compute committee mean for pool
        if pred_matrix.ndim == 3:
            # Multi-output: shape (n_models, n_samples, n_outputs)
            committee_mean = pred_matrix.mean(axis=0)  # shape: (n_samples, n_outputs)
        else:
            # Single-output: shape (n_models, n_samples)
            committee_mean = pred_matrix.mean(axis=0)  # shape: (n_samples,)

        # Select query samples (top disagreement)
        n_select = min(query_batch_size, len(X_pool))
        query_indices = np.argsort(disagreement)[-n_select:]
        query_indices = np.sort(query_indices)

        # Disagreement stats for logging
        disagreement_mean = disagreement.mean()
        disagreement_std_val = disagreement.std()
        disagreement_max = disagreement.max()

        tqdm.write(f"    Iter {step}: Train={len(X_labeled)}, Pool={len(X_pool)}, "
                  f"TestMAE={mae:.4f}, Disagreement=(mean={disagreement_mean:.4f}, max={disagreement_max:.4f}), "
                  f"Selected={n_select}")

        # Compute committee mean across all models
        if pred_matrix.ndim == 3:
            # Multi-output: mean across models and outputs
            committee_mean_all = pred_matrix.mean(axis=0).mean(axis=1)  # shape: (n_samples,)
        else:
            # Single-output: mean across models
            committee_mean_all = pred_matrix.mean(axis=0)  # shape: (n_samples,)

        # Store pool predictions
        pool_pred_dict = {
            "timestamp_local": ts_pool.values,
            "committee_mean": committee_mean_all,
            "disagreement_std": disagreement,
            "selected_by_qbc": False,
        }

        # Store predictions per model
        for model_idx in range(len(fitted_committee)):
            if pred_matrix.ndim == 3:
                # Multi-output
                pool_pred_dict[f"model_{model_idx+1}_pred_mean"] = pred_matrix[model_idx].mean(axis=1)
            else:
                # Single-output
                pool_pred_dict[f"model_{model_idx+1}_pred"] = pred_matrix[model_idx]

        iteration_pool_predictions_df = pd.DataFrame(pool_pred_dict)
        iteration_pool_predictions_df.loc[query_indices, "selected_by_qbc"] = True

        last_pool_predictions_df = iteration_pool_predictions_df.copy()
        last_pool_predictions_df["iteration"] = step

        last_selected_points_df = (
            last_pool_predictions_df[last_pool_predictions_df["selected_by_qbc"]]
            .copy()
            .sort_values("disagreement_std", ascending=False)
            .reset_index(drop=True)
        )

        # Move selected samples from pool to labeled
        X_new = X_pool.iloc[query_indices].copy()
        y_new = y_pool.iloc[query_indices].copy()

        X_labeled = pd.concat([X_labeled, X_new], ignore_index=True)
        y_labeled = pd.concat([y_labeled, y_new], ignore_index=True)

        # Update pool
        keep_mask = np.ones(len(X_pool), dtype=bool)
        keep_mask[query_indices] = False

        X_pool = X_pool.iloc[keep_mask].reset_index(drop=True)
        y_pool = y_pool.iloc[keep_mask].reset_index(drop=True)
        ts_pool = ts_pool.iloc[keep_mask].reset_index(drop=True)

    # Final training on all labeled data
    logging.info(f"\n⏳ Training final committee on {len(X_labeled)} labeled samples...")
    final_committee = fit_committee(base_committee, X_labeled, y_labeled)
    logging.info(f"  ✓ Final committee trained")

    # Final evaluation
    logging.info(f"\n⏳ Computing final predictions on {len(X_test)} test samples...")
    final_pred = committee_mean_prediction(final_committee, X_test)
    final_mae = mean_absolute_error(y_test.values.flatten(), final_pred.flatten())
    final_rmse = float(np.sqrt(mean_squared_error(y_test.values.flatten(), final_pred.flatten())))
    logging.info(f"  ✓ Final MAE: {final_mae:.4f}, final RMSE: {final_rmse:.4f}")

    # Remaining pool disagreement (for visualization)
    if len(X_pool) > 0:
        _, final_disagreement = disagreement_std(final_committee, X_pool)
        final_committee_disagreement = pd.DataFrame({
            "timestamp_local": ts_pool,
            "disagreement_std": final_disagreement,
        }).sort_values("disagreement_std", ascending=False).reset_index(drop=True)
    else:
        final_committee_disagreement = pd.DataFrame(
            columns=["timestamp_local", "disagreement_std"]
        )

    # Build predictions DataFrame with all 16 targets
    logging.info(f"⏳ Building predictions DataFrame with {len(target_cols)} targets...")
    pred_dict = {"timestamp_local": ts_test.values}
    for target_col in target_cols:
        pred_dict[f"actual_{target_col}"] = y_test[target_col].values
        pred_dict[f"pred_{target_col}"] = final_pred[:, target_cols.index(target_col)]
    
    predictions = pd.DataFrame(pred_dict)
    logging.info(f"  ✓ Predictions DataFrame created: {predictions.shape}")

    metrics = {
        "train_rows_final": len(X_labeled),
        "test_rows": len(X_test),
        "mae": float(final_mae),
        "rmse": float(final_rmse),
        "n_queries_completed": len(learning_curve_rows),
        "n_features": len(feature_cols),
        "n_targets": len(target_cols),
    }

    learning_curve = pd.DataFrame(learning_curve_rows)
    final_model = final_committee[0]

    return QBCResult(
        final_model=final_model,
        metrics=metrics,
        predictions=predictions,
        learning_curve=learning_curve,
        final_committee_disagreement=final_committee_disagreement,
        final_pool_predictions=last_pool_predictions_df,
        final_selected_points=last_selected_points_df,
    )


# ============================================================
# Metric Computation Functions
# ============================================================
def compute_metrics_per_variable_horizon(
    y_test: pd.DataFrame,
    y_pred: pd.DataFrame,
    target_cols: List[str],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Compute MAE and RMSE for each target variable and horizon.
    
    Args:
        y_test: DataFrame with actual values (columns are "actual_target_*")
        y_pred: DataFrame with predictions (columns are "pred_target_*")
        target_cols: List of target column names (e.g., "target_temperature_c_h1")
    
    Returns:
        Nested dict: {variable: {horizon: {'mae': float, 'rmse': float}}}
    """
    metrics_dict = {}
    
    logging.info(f"\n⏳ Computing MAE/RMSE for {len(target_cols)} targets...")
    for target_col in tqdm(target_cols, desc="Metrics", unit="target"):
        # Extract variable and horizon from column name
        # Format: target_{variable}_h{step}
        parts = target_col.split("_h")
        horizon_step = int(parts[-1])
        
        # Map step to human-readable horizon
        horizon_map = {1: "6h", 2: "12h", 4: "24h", 8: "48h"}
        horizon = horizon_map.get(horizon_step, f"{horizon_step}h")
        
        # Extract variable (everything after "target_" and before "_h")
        temp = target_col.replace("target_", "").rsplit("_h", 1)[0]
        var_name = temp
        
        # Get actual and predicted values from y_pred DataFrame
        actual_col = f"actual_{target_col}"
        pred_col = f"pred_{target_col}"
        
        if actual_col not in y_pred.columns:
            continue
        if pred_col not in y_pred.columns:
            continue
            
        actual = y_pred[actual_col].values
        pred = y_pred[pred_col].values
        
        mae = mean_absolute_error(actual, pred)
        rmse = float(np.sqrt(mean_squared_error(actual, pred)))
        
        if var_name not in metrics_dict:
            metrics_dict[var_name] = {}
        metrics_dict[var_name][horizon] = {"mae": mae, "rmse": rmse}
    
    logging.info(f"  ✓ Metrics computed for all targets")
    return metrics_dict


def compute_standardized_metrics(
    y_test: pd.DataFrame,
    y_pred: pd.DataFrame,
    target_cols: List[str],
    std_factors: Dict[str, float],
) -> Dict[str, any]:
    """
    Compute normalized MAE for each target using standardization factors.
    
    Args:
        y_test: DataFrame with actual values (columns are "actual_target_*")
        y_pred: DataFrame with predictions (columns are "pred_target_*")
        target_cols: List of target column names
        std_factors: Dict mapping variable names to standardization factors
    
    Returns:
        Dict with keys:
        - 'common_error': overall normalized MAE
        - 'by_variable': {var: normalized_mae}
        - 'by_horizon': {horizon: normalized_mae}
    """
    logging.info(f"\n⏳ Computing standardized metrics with {len(std_factors)} standardization factors...")
    
    all_normalized_maes = []
    by_variable = {}
    by_horizon = {}
    
    horizon_map = {1: "6h", 2: "12h", 4: "24h", 8: "48h"}
    
    for target_col in tqdm(target_cols, desc="Standardized metrics", unit="target"):
        # Parse target name: target_{variable}_h{step}
        parts = target_col.split("_h")
        horizon_step = int(parts[-1])
        horizon = horizon_map.get(horizon_step, f"{horizon_step}h")
        
        # Extract variable name
        var_name = target_col.replace("target_", "").rsplit("_h", 1)[0]
        
        # Get actual and predicted values from y_pred DataFrame
        actual_col = f"actual_{target_col}"
        pred_col = f"pred_{target_col}"
        
        if actual_col not in y_pred.columns or pred_col not in y_pred.columns:
            continue
            
        actual = y_pred[actual_col].values
        pred = y_pred[pred_col].values
        
        mae = mean_absolute_error(actual, pred)
        std_factor = std_factors.get(var_name, 1.0)
        normalized_mae = mae / std_factor
        
        all_normalized_maes.append(normalized_mae)
        
        # Track by variable
        if var_name not in by_variable:
            by_variable[var_name] = []
        by_variable[var_name].append(normalized_mae)
        
        # Track by horizon
        if horizon not in by_horizon:
            by_horizon[horizon] = []
        by_horizon[horizon].append(normalized_mae)
    
    # Average within each group
    by_variable_avg = {var: np.mean(vals) for var, vals in by_variable.items()}
    by_horizon_avg = {horizon: np.mean(vals) for horizon, vals in by_horizon.items()}
    
    logging.info(f"  ✓ Standardized metrics computed")
    
    return {
        "common_error": np.mean(all_normalized_maes),
        "by_variable": by_variable_avg,
        "by_horizon": by_horizon_avg,
        "all_normalized_maes": all_normalized_maes,
    }


def format_metrics_table(metrics_dict: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    """Format per-variable metrics into readable tables."""
    output = "\n" + "=" * 80 + "\n"
    output += "METRICS PER VARIABLE AND HORIZON (MAE / RMSE)\n"
    output += "=" * 80 + "\n\n"
    
    horizon_order = ["6h", "12h", "24h", "48h"]
    
    for var_name in sorted(metrics_dict.keys()):
        output += f"{var_name.upper()}:\n"
        horiz_metrics = metrics_dict[var_name]
        
        for horizon in horizon_order:
            if horizon in horiz_metrics:
                mae = horiz_metrics[horizon]["mae"]
                rmse = horiz_metrics[horizon]["rmse"]
                output += f"  {horizon:>6} ahead:  MAE={mae:.4f}, RMSE={rmse:.4f}\n"
        
        output += "\n"
    
    return output


def format_standardized_metrics(std_metrics: Dict[str, any], std_factors: Dict[str, float]) -> str:
    """Format standardized metrics into readable output."""
    output = "\n" + "=" * 80 + "\n"
    output += "STANDARDIZED COMMON ERROR METRIC\n"
    output += "=" * 80 + "\n\n"
    
    output += f"Normalized MAE (across all 16 targets): {std_metrics['common_error']:.4f}\n\n"
    
    output += "Breakdown by variable:\n"
    for var_name in sorted(std_metrics["by_variable"].keys()):
        norm_mae = std_metrics["by_variable"][var_name]
        std_factor = std_factors.get(var_name, "N/A")
        output += f"  {var_name.ljust(25)} (std factor={str(std_factor):>6}):  Normalized MAE = {norm_mae:.4f}\n"
    
    output += "\nBreakdown by horizon:\n"
    horizon_order = ["6h", "12h", "24h", "48h"]
    for horizon in horizon_order:
        if horizon in std_metrics["by_horizon"]:
            norm_mae = std_metrics["by_horizon"][horizon]
            output += f"  {horizon:>4}:  Normalized MAE = {norm_mae:.4f}\n"
    
    return output


# ============================================================
# Baseline Comparison Functions
# ============================================================
def train_random_baseline(
    df_feat: pd.DataFrame,
    initial_labeled_size: int = 80,
    query_batch_size: int = 12,
    n_queries: int = 25,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Random sampling baseline for active learning.

    Same setup as QBC:
    - same train/test split
    - same initial labeled size
    - same query batch size
    - same number of iterations
    - same committee

    Difference:
    - samples are chosen uniformly at random from the pool
      instead of by committee disagreement.

    Returns:
        DataFrame with learning curve:
        iteration, labeled_size, pool_size, test_mae, test_rmse
    """
    rng = np.random.default_rng(random_state)

    feature_cols = get_feature_columns()
    target_cols = get_target_columns()

    # Train/test split (same as QBC)
    split_idx = int(len(df_feat) * 0.8)
    train_df = df_feat.iloc[:split_idx].copy()
    test_df = df_feat.iloc[split_idx:].copy()

    X_train_full = train_df[feature_cols].reset_index(drop=True)
    y_train_full = train_df[target_cols].reset_index(drop=True)

    X_test = test_df[feature_cols].reset_index(drop=True)
    y_test = test_df[target_cols].reset_index(drop=True)

    if initial_labeled_size >= len(X_train_full):
        raise ValueError("initial_labeled_size er for stor i forhold til train-datasættet.")

    # Initial split
    X_labeled = X_train_full.iloc[:initial_labeled_size].copy()
    y_labeled = y_train_full.iloc[:initial_labeled_size].copy()

    X_pool = X_train_full.iloc[initial_labeled_size:].copy()
    y_pool = y_train_full.iloc[initial_labeled_size:].copy()

    base_committee = build_committee(random_state=random_state)
    learning_curve_rows = []

    logging.info(f"\n⏳ Starting RANDOM baseline loop ({n_queries} iterations, seed={random_state})...")

    for step in tqdm(range(n_queries), desc=f"Random Baseline {random_state}", unit="it"):
        if len(X_pool) == 0:
            tqdm.write(f"  ℹ Pool exhausted at iteration {step}")
            break

        # Train committee on currently labeled data
        fitted_committee = fit_committee(base_committee, X_labeled, y_labeled)

        # Evaluate on test set
        test_pred = committee_mean_prediction(fitted_committee, X_test)
        mae = mean_absolute_error(y_test.values.flatten(), test_pred.flatten())
        rmse = float(np.sqrt(mean_squared_error(y_test.values.flatten(), test_pred.flatten())))

        learning_curve_rows.append({
            "iteration": step,
            "labeled_size": len(X_labeled),
            "pool_size": len(X_pool),
            "test_mae": mae,
            "test_rmse": rmse,
        })

        # Random selection instead of QBC disagreement
        n_select = min(query_batch_size, len(X_pool))
        query_indices = rng.choice(len(X_pool), size=n_select, replace=False)
        query_indices = np.sort(query_indices)

        # Move selected samples from pool to labeled
        X_new = X_pool.iloc[query_indices].copy()
        y_new = y_pool.iloc[query_indices].copy()

        X_labeled = pd.concat([X_labeled, X_new], ignore_index=True)
        y_labeled = pd.concat([y_labeled, y_new], ignore_index=True)

        # Update pool
        keep_mask = np.ones(len(X_pool), dtype=bool)
        keep_mask[query_indices] = False

        X_pool = X_pool.iloc[keep_mask].reset_index(drop=True)
        y_pool = y_pool.iloc[keep_mask].reset_index(drop=True)

    return pd.DataFrame(learning_curve_rows)


def run_random_baseline_repeated(
    df_feat: pd.DataFrame,
    initial_labeled_size: int = 80,
    query_batch_size: int = 12,
    n_queries: int = 25,
    base_random_state: int = 100,
    n_repeats: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the random baseline multiple times to get a fairer comparison
    against a single QBC run.

    Returns:
        all_runs_df: one row per iteration per repeat
        summary_df: mean/std per iteration
    """
    all_runs = []

    logging.info(f"\n⏳ Running random baseline {n_repeats} times...")
    for repeat in range(n_repeats):
        seed = base_random_state + repeat

        curve_df = train_random_baseline(
            df_feat=df_feat,
            initial_labeled_size=initial_labeled_size,
            query_batch_size=query_batch_size,
            n_queries=n_queries,
            random_state=seed,
        ).copy()

        curve_df["repeat"] = repeat
        curve_df["seed"] = seed
        all_runs.append(curve_df)

    all_runs_df = pd.concat(all_runs, ignore_index=True)

    summary_df = (
        all_runs_df
        .groupby(["iteration", "labeled_size"], as_index=False)
        .agg(
            random_mae_mean=("test_mae", "mean"),
            random_mae_std=("test_mae", "std"),
            random_rmse_mean=("test_rmse", "mean"),
            random_rmse_std=("test_rmse", "std"),
        )
        .sort_values("iteration")
        .reset_index(drop=True)
    )

    # std can be NaN if n_repeats == 1
    for col in ["random_mae_std", "random_rmse_std"]:
        summary_df[col] = summary_df[col].fillna(0.0)

    return all_runs_df, summary_df


def plot_qbc_vs_random(
    qbc_learning_curve: pd.DataFrame,
    random_summary_df: pd.DataFrame,
    save_path: str = f"plots/{timestamp}/qbc_vs_random_baseline.png",
) -> None:
    """
    Plot QBC learning curve against repeated random baseline.
    Saves a graph showing whether QBC performs better on this dataset.

    Lower is better for both MAE and RMSE.
    """
    if qbc_learning_curve.empty:
        logging.info("Ingen QBC learning curve data.")
        return

    if random_summary_df.empty:
        logging.info("Ingen random baseline data.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    # --- MAE ---
    axes[0].plot(
        qbc_learning_curve["labeled_size"],
        qbc_learning_curve["test_mae"],
        marker="o",
        linewidth=2.5,
        label="QBC MAE",
    )

    axes[0].plot(
        random_summary_df["labeled_size"],
        random_summary_df["random_mae_mean"],
        marker="s",
        linewidth=2.5,
        label="Random baseline MAE (mean)",
    )

    axes[0].fill_between(
        random_summary_df["labeled_size"],
        random_summary_df["random_mae_mean"] - random_summary_df["random_mae_std"],
        random_summary_df["random_mae_mean"] + random_summary_df["random_mae_std"],
        alpha=0.2,
        label="Random baseline ±1 std",
    )

    axes[0].set_title("QBC vs Random Baseline — MAE")
    axes[0].set_ylabel("Test MAE")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # --- RMSE ---
    axes[1].plot(
        qbc_learning_curve["labeled_size"],
        qbc_learning_curve["test_rmse"],
        marker="o",
        linewidth=2.5,
        label="QBC RMSE",
    )

    axes[1].plot(
        random_summary_df["labeled_size"],
        random_summary_df["random_rmse_mean"],
        marker="s",
        linewidth=2.5,
        label="Random baseline RMSE (mean)",
    )

    axes[1].fill_between(
        random_summary_df["labeled_size"],
        random_summary_df["random_rmse_mean"] - random_summary_df["random_rmse_std"],
        random_summary_df["random_rmse_mean"] + random_summary_df["random_rmse_std"],
        alpha=0.2,
        label="Random baseline ±1 std",
    )

    axes[1].set_title("QBC vs Random Baseline — RMSE")
    axes[1].set_xlabel("Number of labeled samples")
    axes[1].set_ylabel("Test RMSE")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    os.makedirs(f"plots/{timestamp}", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logging.info(f"  ✓ Saved {save_path}")


# ============================================================
# Visualisering
# ============================================================
def plot_qbc_selection(result: QBCResult, max_points: int = 120) -> None:
    """
    Visualiserer den seneste QBC-iteration med committee mean.
    """
    df = result.final_pool_predictions.copy()

    if df.empty:
        logging.info("Ingen pool-predictions gemt til plotting.")
        return

    if len(df) > max_points:
        df = df.tail(max_points).reset_index(drop=True)

    selected_df = df[df["selected_by_qbc"]].copy()
    # Match both "model_X_pred" and "model_X_pred_mean" column names
    model_cols = [c for c in df.columns if c.startswith("model_") and ("_pred" in c)]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for col in model_cols:
        axes[0].plot(df["timestamp_local"], df[col], alpha=0.7, label=col)

    if "committee_mean" in df.columns:
        axes[0].plot(
            df["timestamp_local"],
            df["committee_mean"],
            linewidth=2,
            label="committee_mean",
        )

        if not selected_df.empty:
            axes[0].scatter(
                selected_df["timestamp_local"],
                selected_df["committee_mean"],
                s=70,
                marker="o",
                label="QBC selected",
            )

    axes[0].set_title("Committee predictioner på unlabeled pool")
    axes[0].set_ylabel("Forudsagt værdi (aggregeret over 16 targets)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        df["timestamp_local"],
        df["disagreement_std"],
        label="disagreement_std",
    )

    if not selected_df.empty:
        axes[1].scatter(
            selected_df["timestamp_local"],
            selected_df["disagreement_std"],
            s=70,
            marker="o",
            label="QBC selected",
        )

    axes[1].set_title("QBC disagreement (standardafvigelse mellem modeller)")
    axes[1].set_ylabel("Disagreement")
    axes[1].set_xlabel("Tidspunkt")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(f"plots/{timestamp}", exist_ok=True)
    plt.savefig(f"plots/{timestamp}/qbc_selection.png")


def plot_qbc_learning_curve(result: QBCResult) -> None:
    """
    Plotter model performance (MAE og RMSE) over QBC-iterationer.
    """
    df = result.learning_curve.copy()

    if df.empty:
        logging.info("Ingen learning curve data.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(df["iteration"], df["test_mae"], label="MAE")
    ax.plot(df["iteration"], df["test_rmse"], label="RMSE")

    ax.set_title("QBC learning curve (model performance over iterationer)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fejl (°C)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(f"plots/{timestamp}", exist_ok=True)
    plt.savefig(f"plots/{timestamp}/qbc_learning_curve.png")


def plot_predictions_per_variable_horizon(pred_df: pd.DataFrame, target_cols: List[str]) -> None:
    """
    Create comprehensive plots for all 4 variables × 4 horizons.
    Each subplot shows actual vs predicted values over time.
    """
    logging.info("\n⏳ Creating detailed prediction plots for all variables and horizons...")
    
    variables = ["temperature_c", "relative_humidity_pct", "pressure_hpa", "precip_mm"]
    var_titles = {"temperature_c": "Temperature (°C)", 
                  "relative_humidity_pct": "Relative Humidity (%)",
                  "pressure_hpa": "Pressure (hPa)",
                  "precip_mm": "Precipitation (mm)"}
    
    for var in variables:
        # Create a 2x2 subplot grid for each variable (one per horizon)
        fig, axes = plt.subplots(2, 2, figsize=(16, 11))
        fig.suptitle(f"Predictions vs Actual: {var_titles.get(var, var.upper())}", 
                     fontsize=16, fontweight='bold', y=0.995)
        axes = axes.flatten()
        
        for horizon_idx, (step, horizon) in enumerate([(1, "6h"), (2, "12h"), (4, "24h"), (8, "48h")]):
            target_col = f"target_{var}_h{step}"
            actual_col = f"actual_{target_col}"
            pred_col = f"pred_{target_col}"
            
            if actual_col not in pred_df.columns or pred_col not in pred_df.columns:
                axes[horizon_idx].text(0.5, 0.5, f"No data for {horizon}", ha='center', va='center')
                axes[horizon_idx].set_title(f"{horizon} ahead")
                continue
            
            actual = pred_df[actual_col].values
            predictions = pred_df[pred_col].values
            x_axis = range(len(actual))
            
            # Plot actual vs predicted
            axes[horizon_idx].plot(x_axis, actual, 'o-', label='Actual', linewidth=2.5, markersize=5, alpha=0.8, color='#2E86AB')
            axes[horizon_idx].plot(x_axis, predictions, 's--', label='Predicted', linewidth=2.5, markersize=5, alpha=0.8, color='#A23B72')
            
            # Compute MAE and RMSE for this specific target
            mae = mean_absolute_error(actual, predictions)
            rmse = float(np.sqrt(mean_squared_error(actual, predictions)))
            
            axes[horizon_idx].set_title(f"{horizon} ahead\nMAE={mae:.4f}, RMSE={rmse:.4f}", fontsize=11, fontweight='bold')
            axes[horizon_idx].set_xlabel("Test Sample Index", fontsize=10)
            axes[horizon_idx].set_ylabel(var_titles.get(var, var), fontsize=10)
            axes[horizon_idx].legend(loc='best', fontsize=10)
            axes[horizon_idx].grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        os.makedirs(f"plots/{timestamp}", exist_ok=True)
        plot_filename = f"plots/{timestamp}/predictions_{var}.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.close()
        logging.info(f"  ✓ Saved {plot_filename}")


def plot_error_degradation(pred_df: pd.DataFrame, target_cols: List[str]) -> None:
    """
    Plot how prediction error increases with forecast horizon for each variable.
    Shows error degradation across 6h, 12h, 24h, 48h horizons.
    """
    logging.info("⏳ Creating error degradation plots...")
    
    variables = ["temperature_c", "relative_humidity_pct", "pressure_hpa", "precip_mm"]
    var_titles = {"temperature_c": "Temperature", 
                  "relative_humidity_pct": "Relative Humidity",
                  "pressure_hpa": "Pressure",
                  "precip_mm": "Precipitation"}
    units = {"temperature_c": "°C", "relative_humidity_pct": "%", "pressure_hpa": "hPa", "precip_mm": "mm"}
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Prediction Error Degradation by Forecast Horizon", fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    for var_idx, var in enumerate(variables):
        horizons_data = {"6h": [], "12h": [], "24h": [], "48h": []}
        horizon_steps = {1: "6h", 2: "12h", 4: "24h", 8: "48h"}
        
        for step, horizon in horizon_steps.items():
            target_col = f"target_{var}_h{step}"
            actual_col = f"actual_{target_col}"
            pred_col = f"pred_{target_col}"
            
            if actual_col in pred_df.columns and pred_col in pred_df.columns:
                actual = pred_df[actual_col].values
                predictions = pred_df[pred_col].values
                mae = mean_absolute_error(actual, predictions)
                horizons_data[horizon].append(mae)
        
        # Plot MAE degradation
        horizons_list = ["6h", "12h", "24h", "48h"]
        maes = [horizons_data[h][0] if horizons_data[h] else 0 for h in horizons_list]
        
        x_pos = range(len(horizons_list))
        axes[var_idx].plot(x_pos, maes, 'o-', linewidth=3, markersize=10, 
                          color=colors[var_idx], markerfacecolor='white', markeredgewidth=2.5)
        axes[var_idx].fill_between(x_pos, maes, alpha=0.2, color=colors[var_idx])
        
        axes[var_idx].set_title(f"{var_titles.get(var, var.upper())}", fontsize=12, fontweight='bold')
        axes[var_idx].set_xlabel("Forecast Horizon", fontsize=11)
        axes[var_idx].set_ylabel(f"MAE ({units.get(var, 'unit')})", fontsize=11)
        axes[var_idx].set_xticks(x_pos)
        axes[var_idx].set_xticklabels(horizons_list)
        axes[var_idx].grid(True, alpha=0.3, linestyle='--')
        
        # Add value labels on points
        for i, (h, mae) in enumerate(zip(horizons_list, maes)):
            if mae > 0:
                axes[var_idx].text(i, mae + max(maes)*0.08, f"{mae:.3f}", 
                                  ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    os.makedirs(f"plots/{timestamp}", exist_ok=True)
    plot_filename = f"plots/{timestamp}/error_degradation.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"  ✓ Saved {plot_filename}")


# ============================================================
# Main
# ============================================================
def main() -> None:
    logging.info("=" * 80)
    logging.info("DMI Copenhagen Multi-Variable, Multi-Horizon Weather Prediction with QBC")
    logging.info("=" * 80)
    
    logging.info("\n⏳ Finder bedste station i København-området baseret på dennes dækning...")
    best_score, raw_best_dataset, station_ranking = find_best_station()

    logging.info("\n✓ Bedste station valgt:")
    logging.info(f"  • stationId      : {best_score.station_id}")
    logging.info(f"  • navn           : {best_score.station_name}")
    logging.info(f"  • afstand (km)   : {best_score.distance_km:.2f}")
    logging.info(f"  • coverage_score : {best_score.coverage_score:.4f}")
    logging.info(f"  • completeness   : {best_score.completeness_ratio:.2%}")
    logging.info(f"  • komplette rækker (4 variabler): {best_score.rows_complete_all_vars}/{best_score.rows_expected}")
    logging.info(f"  • rå rækker pr. variabel: {best_score.rows_by_variable}")

    logging.info(f"\n⏳ Cleaning dataset ({len(raw_best_dataset)} rows)...")
    clean_df = clean_dataset(raw_best_dataset)
    logging.info(f"  ✓ Dataset cleaned: {len(clean_df)} rows retained")

    logging.info(f"\n⏳ Adding features and 16 targets ({len(clean_df)} samples)...")
    feat_df, target_cols = add_features_and_target(clean_df)
    logging.info(f"  ✓ Features and targets added: {feat_df.shape[0]} samples × {feat_df.shape[1]} columns")
    logging.info(f"     → {len([c for c in feat_df.columns if not c.startswith('target')])} features")
    logging.info(f"     → {len(target_cols)} targets (4 variables × 4 horizons)")

    logging.info(f"\n⏳ Training QBC model with multi-output regression...")
    qbc_result = train_qbc_model(
        feat_df,
        initial_labeled_size=QBC_INITIAL_LABELED_SIZE,
        query_batch_size=QBC_QUERY_BATCH_SIZE,
        n_queries=QBC_N_QUERIES,
        random_state=QBC_RANDOM_STATE,
    )

    logging.info(f"\n⏳ Plotting QBC results...")
    plot_qbc_learning_curve(qbc_result)
    plot_qbc_selection(qbc_result, max_points=120)
    logging.info(f"  ✓ QBC plots saved to plots/")
    
    # Extract results from QBC
    metrics = qbc_result.metrics
    pred_df = qbc_result.predictions
    learning_curve_df = qbc_result.learning_curve
    disagreement_df = qbc_result.final_committee_disagreement
    
    # Extract target columns for metric computation
    target_cols = get_target_columns()
    
    # Create detailed prediction plots for all variables and horizons
    plot_predictions_per_variable_horizon(pred_df, target_cols)
    plot_error_degradation(pred_df, target_cols)

    # Print QBC summary
    logging.info("\n" + "=" * 80)
    logging.info("QBC MODEL RESULTS")
    logging.info("=" * 80)
    logging.info(f"\nDataset Summary:")
    logging.info(f"  • Final labeled rows (train)   : {metrics['train_rows_final']}")
    logging.info(f"  • Test rows                    : {metrics['test_rows']}")
    logging.info(f"  • Input features               : {metrics['n_features']}")
    logging.info(f"  • Prediction targets           : {metrics['n_targets']} (4 variables × 4 horizons)")
    logging.info(f"  • Active learning iterations   : {metrics['n_queries_completed']}")

    logging.info(f"\nAggregated Performance (across all 16 targets):")
    logging.info(f"  • MAE (Mean Absolute Error)    : {metrics['mae']:.4f}")
    logging.info(f"  • RMSE (Root Mean Squared Error): {metrics['rmse']:.4f}")

    logging.info(f"\nLearning Curve (last 10 iterations):")
    logging.info(learning_curve_df.tail(10).to_string(index=False))

    logging.info(f"\nRecent Test Predictions (last 10 samples):")
    logging.info(pred_df.tail(10).to_string(index=False))

    if not disagreement_df.empty:
        logging.info(f"\nTop 10 most uncertain remaining pool points:")
        logging.info(disagreement_df.head(10).to_string(index=False))
    else:
        logging.info("\nNo remaining pool points.")

    # Compute per-variable, per-horizon metrics
    logging.info("\n⏳ Computing metrics per variable and horizon...")
    var_horizon_metrics = compute_metrics_per_variable_horizon(pred_df, pred_df, target_cols)
    
    # logging.info formatted metrics table
    metrics_output = format_metrics_table(var_horizon_metrics)
    logging.info(metrics_output)

    # Compute standardized metrics
    logging.info("\n⏳ Computing standardized metrics...")
    std_metrics = compute_standardized_metrics(pred_df, pred_df, target_cols, STANDARDIZATION_FACTORS)
    
    # Print formatted standardized metrics
    std_output = format_standardized_metrics(std_metrics, STANDARDIZATION_FACTORS)
    logging.info(std_output)

    # Run baseline comparison (random sampling)
    logging.info(f"\n⏳ Running random baseline comparison ({10} repeats)...")
    random_runs_df, random_summary_df = run_random_baseline_repeated(
        feat_df,
        initial_labeled_size=QBC_INITIAL_LABELED_SIZE,
        query_batch_size=QBC_QUERY_BATCH_SIZE,
        n_queries=QBC_N_QUERIES,
        base_random_state=100,
        n_repeats=10,
    )
    
    logging.info(f"\n⏳ Plotting QBC vs Random Baseline...")
    plot_qbc_vs_random(
        qbc_result.learning_curve,
        random_summary_df,
        save_path="plots/qbc_vs_random_baseline.png",
    )
    
    # Save results to CSV files
    logging.info("\n⏳ Saving results to CSV files...")
    os.makedirs("data", exist_ok=True)

    logging.info("  • Saving station ranking...")
    station_ranking.to_csv("data/dmi_station_ranking.csv", index=False)
    
    logging.info("  • Saving raw dataset...")
    raw_best_dataset.to_csv("data/dmi_best_station_raw_4x_daily.csv", index=False)
    
    logging.info("  • Saving cleaned dataset...")
    clean_df.to_csv("data/dmi_copenhagen_clean_4x_daily.csv", index=False)
    
    logging.info("  • Saving features and targets...")
    feat_df.to_csv("data/dmi_copenhagen_features_4x_daily.csv", index=False)
    
    logging.info("  • Saving multi-output predictions...")
    pred_df.to_csv("data/dmi_qbc_multioutput_predictions.csv", index=False)
    
    logging.info("  • Saving learning curve...")
    learning_curve_df.to_csv("data/dmi_qbc_learning_curve.csv", index=False)
    
    logging.info("  • Saving pool disagreement...")
    disagreement_df.to_csv("data/dmi_qbc_pool_disagreement.csv", index=False)
    
    logging.info("  • Saving final pool predictions...")
    qbc_result.final_pool_predictions.to_csv("data/dmi_qbc_final_pool_predictions.csv", index=False)
    
    logging.info("  • Saving selected points...")
    qbc_result.final_selected_points.to_csv("data/dmi_qbc_selected_points.csv", index=False)
    
    logging.info("  • Saving random baseline results...")
    random_runs_df.to_csv("data/dmi_random_baseline_runs.csv", index=False)
    random_summary_df.to_csv("data/dmi_random_baseline_summary.csv", index=False)
    
    # Save metrics tables
    logging.info("  • Saving per-variable, per-horizon metrics...")
    metrics_rows = []
    for var_name in var_horizon_metrics:
        for horizon in var_horizon_metrics[var_name]:
            metrics_rows.append({
                "variable": var_name,
                "horizon": horizon,
                "mae": var_horizon_metrics[var_name][horizon]["mae"],
                "rmse": var_horizon_metrics[var_name][horizon]["rmse"],
            })
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv("data/dmi_metrics_by_variable_horizon.csv", index=False)
    
    logging.info("  • Saving standardized metrics...")
    std_metrics_rows = []
    for var_name in std_metrics["by_variable"]:
        std_metrics_rows.append({
            "variable": var_name,
            "normalized_mae": std_metrics["by_variable"][var_name],
        })
    for horizon in std_metrics["by_horizon"]:
        std_metrics_rows.append({
            "horizon": horizon,
            "normalized_mae": std_metrics["by_horizon"][horizon],
        })
    std_metrics_rows.append({
        "metric": "common_error",
        "value": std_metrics["common_error"],
    })
    std_metrics_df = pd.DataFrame(std_metrics_rows)
    std_metrics_df.to_csv("data/dmi_metrics_standardized.csv", index=False)

    logging.info("\n" + "=" * 80)
    logging.info("✓ ALL FILES SAVED SUCCESSFULLY")
    logging.info("=" * 80)
    logging.info("\nOutput files created:")
    logging.info("  • dmi_station_ranking.csv")
    logging.info("  • dmi_best_station_raw_4x_daily.csv")
    logging.info("  • dmi_copenhagen_clean_4x_daily.csv")
    logging.info("  • dmi_copenhagen_features_4x_daily.csv")
    logging.info("  • dmi_qbc_multioutput_predictions.csv")
    logging.info("  • dmi_qbc_learning_curve.csv")
    logging.info("  • dmi_qbc_pool_disagreement.csv")
    logging.info("  • dmi_qbc_final_pool_predictions.csv")
    logging.info("  • dmi_qbc_selected_points.csv")
    logging.info("  • dmi_random_baseline_runs.csv")
    logging.info("  • dmi_random_baseline_summary.csv")
    logging.info("  • dmi_metrics_by_variable_horizon.csv")
    logging.info("  • dmi_metrics_standardized.csv")
    logging.info("\nPlot files created:")
    logging.info(f"  • plots/{timestamp}/qbc_learning_curve.png")
    logging.info(f"  • plots/{timestamp}/qbc_selection.png")
    logging.info(f"  • plots/qbc_vs_random_baseline.png")


if __name__ == "__main__":
    main()