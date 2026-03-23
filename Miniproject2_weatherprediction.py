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

import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# Konfiguration
# ============================================================
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
    - fylder nedbør med 0 som simpel baseline
    """
    if df.empty:
        raise RuntimeError("Tomt datasæt efter merge.")

    out = df.copy()
    out["timestamp_local"] = pd.to_datetime(out["timestamp_local"], errors="coerce")
    out = out.dropna(subset=["timestamp_local"]).sort_values("timestamp_local")

    start = out["timestamp_local"].min().floor("D")
    end = out["timestamp_local"].max().ceil("D")

    full_index = pd.date_range(
        start=start,
        end=end,
        freq="6h",
        tz=COPENHAGEN_TZ,
    )

    full = pd.DataFrame({"timestamp_local": full_index})
    full["date_local"] = full["timestamp_local"].dt.date
    full["hour_local"] = full["timestamp_local"].dt.hour
    full = full[full["hour_local"].isin(sorted(TARGET_HOURS))]

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

    return out


def add_features_and_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bygger features og target.
    Target = temperaturen ved næste 6-timers måling.
    """
    out = df.copy()
    ts = out["timestamp_local"]

    out["month"] = ts.dt.month
    out["weekday"] = ts.dt.weekday
    out["dayofyear"] = ts.dt.dayofyear

    out["hour_sin"] = np.sin(2 * np.pi * out["hour_local"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour_local"] / 24.0)
    out["doy_sin"] = np.sin(2 * np.pi * out["dayofyear"] / 366.0)
    out["doy_cos"] = np.cos(2 * np.pi * out["dayofyear"] / 366.0)

    base_cols = list(PARAMETERS.values())

    for col in base_cols:
        out[f"{col}_lag1"] = out[col].shift(1)
        out[f"{col}_lag2"] = out[col].shift(2)
        out[f"{col}_lag4"] = out[col].shift(4)

    out["temp_roll4_mean"] = out["temperature_c"].shift(1).rolling(4).mean()
    out["rh_roll4_mean"] = out["relative_humidity_pct"].shift(1).rolling(4).mean()
    out["pressure_roll4_mean"] = out["pressure_hpa"].shift(1).rolling(4).mean()
    out["precip_roll4_sum"] = out["precip_mm"].shift(1).rolling(4).sum()

    out["target_next_temperature_c"] = out["temperature_c"].shift(-1)

    out = out.dropna(subset=["target_next_temperature_c"]).reset_index(drop=True)
    return out


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


def build_committee(random_state: int = 42) -> List[Pipeline]:
    """
    Bygger en lille committee af forskellige regressionsmodeller.
    """
    return [
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1,
            ))
        ]),
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                random_state=random_state + 1,
                n_jobs=-1,
            ))
        ]),
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),
    ]


def fit_committee(
    committee: List[Pipeline],
    X_labeled: pd.DataFrame,
    y_labeled: pd.Series,
) -> List[Pipeline]:
    """
    Træner friske kopier af alle modellerne i committee.
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
    Returnerer:
    - predictions_matrix: shape = (n_models, n_pool_samples)
    - disagreement: std mellem model-predictions pr. punkt
    """
    preds = []
    for model in committee:
        preds.append(model.predict(X_pool))

    predictions_matrix = np.vstack(preds)
    disagreement = predictions_matrix.std(axis=0)
    return predictions_matrix, disagreement


def committee_mean_prediction(
    committee: List[Pipeline],
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Endelig committee-prediction = gennemsnit af model-predictions.
    """
    preds = np.vstack([model.predict(X) for model in committee])
    return preds.mean(axis=0)


def train_qbc_model(
    df_feat: pd.DataFrame,
    initial_labeled_size: int = 80,
    query_batch_size: int = 12,
    n_queries: int = 25,
    random_state: int = 42,
) -> QBCResult:
    """
    Træner temperaturmodellen med Query by Committee.
    """
    feature_cols = get_feature_columns()
    target_col = "target_next_temperature_c"

    split_idx = int(len(df_feat) * 0.8)
    train_df = df_feat.iloc[:split_idx].copy()
    test_df = df_feat.iloc[split_idx:].copy()

    X_train_full = train_df[feature_cols].reset_index(drop=True)
    y_train_full = train_df[target_col].reset_index(drop=True)

    X_test = test_df[feature_cols].reset_index(drop=True)
    y_test = test_df[target_col].reset_index(drop=True)

    if initial_labeled_size >= len(X_train_full):
        raise ValueError("initial_labeled_size er for stor i forhold til train-datasættet.")

    X_labeled = X_train_full.iloc[:initial_labeled_size].copy()
    y_labeled = y_train_full.iloc[:initial_labeled_size].copy()

    X_pool = X_train_full.iloc[initial_labeled_size:].copy()
    y_pool = y_train_full.iloc[initial_labeled_size:].copy()

    ts_pool = train_df.iloc[initial_labeled_size:]["timestamp_local"].reset_index(drop=True)
    ts_test = test_df["timestamp_local"].reset_index(drop=True)

    base_committee = build_committee(random_state=random_state)

    learning_curve_rows = []
    last_pool_predictions_df = pd.DataFrame()
    last_selected_points_df = pd.DataFrame()

    for step in range(n_queries):
        if len(X_pool) == 0:
            break

        fitted_committee = fit_committee(base_committee, X_labeled, y_labeled)

        test_pred = committee_mean_prediction(fitted_committee, X_test)
        mae = mean_absolute_error(y_test, test_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))

        learning_curve_rows.append({
            "iteration": step,
            "labeled_size": len(X_labeled),
            "pool_size": len(X_pool),
            "test_mae": mae,
            "test_rmse": rmse,
        })

        pred_matrix, disagreement = disagreement_std(fitted_committee, X_pool)
        committee_mean = pred_matrix.mean(axis=0)

        n_select = min(query_batch_size, len(X_pool))
        query_indices = np.argsort(disagreement)[-n_select:]
        query_indices = np.sort(query_indices)

        pool_pred_dict = {
            "timestamp_local": ts_pool.values,
            "committee_mean": committee_mean,
            "disagreement_std": disagreement,
            "selected_by_qbc": False,
        }

        for model_idx in range(pred_matrix.shape[0]):
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

        X_new = X_pool.iloc[query_indices].copy()
        y_new = y_pool.iloc[query_indices].copy()

        X_labeled = pd.concat([X_labeled, X_new], ignore_index=True)
        y_labeled = pd.concat([y_labeled, y_new], ignore_index=True)

        keep_mask = np.ones(len(X_pool), dtype=bool)
        keep_mask[query_indices] = False

        X_pool = X_pool.iloc[keep_mask].reset_index(drop=True)
        y_pool = y_pool.iloc[keep_mask].reset_index(drop=True)
        ts_pool = ts_pool.iloc[keep_mask].reset_index(drop=True)

    final_committee = fit_committee(base_committee, X_labeled, y_labeled)

    final_pred = committee_mean_prediction(final_committee, X_test)
    final_mae = mean_absolute_error(y_test, final_pred)
    final_rmse = float(np.sqrt(mean_squared_error(y_test, final_pred)))

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

    predictions = pd.DataFrame({
        "timestamp_local": ts_test,
        "actual_next_temp_c": y_test.values,
        "predicted_next_temp_c": final_pred,
    })

    metrics = {
        "train_rows_final": len(X_labeled),
        "test_rows": len(X_test),
        "mae": float(final_mae),
        "rmse": float(final_rmse),
        "n_queries_completed": len(learning_curve_rows),
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
# Visualisering
# ============================================================
def plot_qbc_selection(result: QBCResult, max_points: int = 120) -> None:
    """
    Visualiserer den seneste QBC-iteration.
    """
    df = result.final_pool_predictions.copy()

    if df.empty:
        print("Ingen pool-predictions gemt til plotting.")
        return

    if len(df) > max_points:
        df = df.tail(max_points).reset_index(drop=True)

    selected_df = df[df["selected_by_qbc"]].copy()
    model_cols = [c for c in df.columns if c.startswith("model_") and c.endswith("_pred")]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for col in model_cols:
        axes[0].plot(df["timestamp_local"], df[col], alpha=0.7, label=col)

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
    axes[0].set_ylabel("Forudsagt næste temperatur (°C)")
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
    plt.show()


def plot_qbc_learning_curve(result: QBCResult) -> None:
    """
    Plotter model performance (MAE og RMSE) over QBC-iterationer.
    """
    df = result.learning_curve.copy()

    if df.empty:
        print("Ingen learning curve data.")
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
    plt.show()


# ============================================================
# Main
# ============================================================
def main() -> None:
    print("Finder bedste station i København-området baseret på dennes dækning (coverage_score)...")
    best_score, raw_best_dataset, station_ranking = find_best_station()

    print("\nBedste station valgt:")
    print(f"  stationId      : {best_score.station_id}")
    print(f"  navn           : {best_score.station_name}")
    print(f"  afstand (km)   : {best_score.distance_km:.2f}")
    print(f"  coverage_score : {best_score.coverage_score:.4f}")
    print(f"  completeness   : {best_score.completeness_ratio:.2%}")
    print(f"  komplette rækker (4 variabler): {best_score.rows_complete_all_vars}/{best_score.rows_expected}")
    print(f"  rå rækker pr. variabel: {best_score.rows_by_variable}")

    clean_df = clean_dataset(raw_best_dataset)
    feat_df = add_features_and_target(clean_df)

    qbc_result = train_qbc_model(
        feat_df,
        initial_labeled_size=80,
        query_batch_size=12,
        n_queries=25,
        random_state=42,
    )

    plot_qbc_learning_curve(qbc_result)
    plot_qbc_selection(qbc_result, max_points=120)

    metrics = qbc_result.metrics
    pred_df = qbc_result.predictions
    learning_curve_df = qbc_result.learning_curve
    disagreement_df = qbc_result.final_committee_disagreement

    print("\nQBC-modelresultater:")
    print(f"  final train rows : {metrics['train_rows_final']}")
    print(f"  test rows        : {metrics['test_rows']}")
    print(f"  MAE              : {metrics['mae']:.3f} °C")
    print(f"  RMSE             : {metrics['rmse']:.3f} °C")
    print(f"  query iterations : {metrics['n_queries_completed']}")

    print("\nLearning curve (sidste 10 iterationer):")
    print(learning_curve_df.tail(10).to_string(index=False))

    print("\nSeneste 10 prediction-rækker:")
    print(pred_df.tail(10).to_string(index=False))

    print("\nTop 10 mest usikre resterende pool-punkter:")
    if disagreement_df.empty:
        print("Ingen resterende pool-punkter.")
    else:
        print(disagreement_df.head(10).to_string(index=False))

    station_ranking.to_csv("dmi_station_ranking.csv", index=False)
    raw_best_dataset.to_csv("dmi_best_station_raw_4x_daily.csv", index=False)
    clean_df.to_csv("dmi_copenhagen_clean_4x_daily.csv", index=False)
    feat_df.to_csv("dmi_copenhagen_features_4x_daily.csv", index=False)
    pred_df.to_csv("dmi_qbc_temperature_predictions.csv", index=False)
    learning_curve_df.to_csv("dmi_qbc_learning_curve.csv", index=False)
    disagreement_df.to_csv("dmi_qbc_pool_disagreement.csv", index=False)
    qbc_result.final_pool_predictions.to_csv("dmi_qbc_final_pool_predictions.csv", index=False)
    qbc_result.final_selected_points.to_csv("dmi_qbc_selected_points.csv", index=False)

    print("\nFiler gemt:")
    print("  - dmi_station_ranking.csv")
    print("  - dmi_best_station_raw_4x_daily.csv")
    print("  - dmi_copenhagen_clean_4x_daily.csv")
    print("  - dmi_copenhagen_features_4x_daily.csv")
    print("  - dmi_qbc_temperature_predictions.csv")
    print("  - dmi_qbc_learning_curve.csv")
    print("  - dmi_qbc_pool_disagreement.csv")
    print("  - dmi_qbc_final_pool_predictions.csv")
    print("  - dmi_qbc_selected_points.csv")


if __name__ == "__main__":
    main()