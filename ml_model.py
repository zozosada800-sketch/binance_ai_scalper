"""
ml_model.py
===========

Task 4: labeling + model training for the ML scalping engine.

Pipeline:
    1. Triple-barrier labels on the 5m bars (profit-take / stop-loss / time
       barrier), sized in ATR so the target is volatility-normalised.
    2. Pool all symbols into one cross-sectional dataset (features are
       scale-free, so symbols can be trained on jointly).
    3. Strictly chronological train/test split with an embargo gap so no
       label window straddles the boundary (no lookahead leakage).
    4. Train LightGBM and XGBoost classifiers (class imbalance handled via
       scale_pos_weight), optionally tune LightGBM with Optuna over
       TimeSeriesSplit CV.
    5. Evaluate both on the untouched test tail, keep the best model by
       ROC AUC, and persist it (joblib) together with its feature list and
       the label configuration.

Usage (standalone smoke test):
    python ml_model.py            # quick train with default hyper-parameters
    python ml_model.py --tune     # + Optuna hyper-parameter search (slower)

Usage (from other modules):
    from ml_model import load_model, predict_latest
    bundle = load_model("models/scalp_model.joblib")
    proba_up = predict_latest(features["BTC/USDT"], bundle)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

from feature_engineering import PRIMARY_TF, RAW_COLUMNS, build_features

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_DIR = "models"
MODEL_FILENAME = "scalp_model.joblib"
DEFAULT_MODEL_PATH = str(Path(MODEL_DIR) / MODEL_FILENAME)

# --- labeling: triple-barrier ------------------------------------------------
PT_ATR_MULT = 1.0        # profit-take barrier: +PT_ATR_MULT x ATR above entry
SL_ATR_MULT = 1.0        # stop-loss barrier:   -SL_ATR_MULT x ATR below entry
LABEL_HORIZON = 6        # time barrier: 6 x 5m bars = 30 minutes

# --- training ------------------------------------------------------------------
TEST_FRACTION = 0.2      # last 20% of the pooled timeline = test set
EMBARGO_BARS = LABEL_HORIZON   # gap (in 5m bars) between train and test
CV_SPLITS = 3            # TimeSeriesSplit folds used by Optuna
N_OPTUNA_TRIALS = 25
RANDOM_STATE = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("ml_model")


# ---------------------------------------------------------------------------
# Labeling: triple-barrier
# ---------------------------------------------------------------------------

def triple_barrier_labels(
    ohlcv: pd.DataFrame,
    atr: pd.Series,
    pt_mult: float = PT_ATR_MULT,
    sl_mult: float = SL_ATR_MULT,
    horizon: int = LABEL_HORIZON,
) -> pd.Series:
    """
    Triple-barrier labels on 5m bars (entry = close of bar t).

        upper barrier = close[t] + pt_mult * atr[t]   -> label 1 (profit-take)
        lower barrier = close[t] - sl_mult * atr[t]   -> label 0 (stop-loss)
        time barrier  = horizon bars later

    Scan bars t+1 .. t+horizon in order:
        * high >= upper first            -> 1
        * low  <= lower first            -> 0
        * both hit inside the same bar   -> 0 (conservative: assume worst case)
        * neither hit within the horizon -> 1 if close[t+horizon] > close[t] else 0

    Bars without a full forward window or with NaN/zero ATR get NaN labels.
    """
    n = len(ohlcv)
    labels = np.full(n, np.nan)
    close = ohlcv["close"].to_numpy()
    high = ohlcv["high"].to_numpy()
    low = ohlcv["low"].to_numpy()
    atr_arr = atr.to_numpy()

    for t in range(n - horizon):
        a = atr_arr[t]
        if np.isnan(a) or a <= 0:
            continue
        upper = close[t] + pt_mult * a
        lower = close[t] - sl_mult * a

        label = np.nan
        for k in range(1, horizon + 1):
            hit_up = high[t + k] >= upper
            hit_dn = low[t + k] <= lower
            if hit_up and hit_dn:
                label = 0.0          # ambiguous bar -> conservative stop
                break
            if hit_up:
                label = 1.0
                break
            if hit_dn:
                label = 0.0
                break
        if np.isnan(label):
            label = 1.0 if close[t + horizon] > close[t] else 0.0
        labels[t] = label

    return pd.Series(labels, index=ohlcv.index, name="label")


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

def assemble_dataset(
    symbols: list,
    data: Dict[Tuple[str, str], pd.DataFrame],
    features: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Pool every symbol into one labeled dataset.

    Returns (X, y, meta) where meta carries the symbol per row for traceability.
    """
    exclude = list(RAW_COLUMNS) + ["label", "symbol"]
    frames = []

    for symbol in symbols:
        feat = features.get(symbol)
        ohlcv = data.get((symbol, PRIMARY_TF))
        if feat is None or feat.empty or ohlcv is None:
            continue
        df = feat.copy()
        df["label"] = triple_barrier_labels(ohlcv, feat["atr"])
        df["symbol"] = symbol
        frames.append(df)

    if not frames:
        raise RuntimeError("No labeled samples could be assembled.")

    pooled = pd.concat(frames).sort_index()
    pooled = pooled.dropna(subset=["label"])

    feature_cols = [c for c in pooled.columns if c not in exclude]
    X = pooled[feature_cols].dropna()
    y = pooled["label"].astype(int).loc[X.index]
    meta = pooled.loc[X.index, ["symbol"]]

    logger.info(
        "Dataset: %d samples x %d features | up-share %.1f%%",
        len(X), X.shape[1], 100.0 * y.mean(),
    )
    return X, y, meta


def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_fraction: float = TEST_FRACTION,
    embargo_bars: int = EMBARGO_BARS,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Time-ordered split with an embargo gap.

    The embargo removes training rows whose label window (LABEL_HORIZON bars
    into the future) would overlap the test period.
    """
    n = len(X)
    split_pos = int(n * (1.0 - test_fraction))
    if split_pos <= 0 or split_pos >= n:
        raise RuntimeError("Not enough samples for a chronological split.")

    test_start = X.index[split_pos]
    embargo_start = test_start - pd.Timedelta(PRIMARY_TF) * embargo_bars

    train_mask = X.index < embargo_start
    test_mask = X.index >= test_start

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_te, y_te = X[test_mask], y[test_mask]

    if X_tr.empty or X_te.empty:
        raise RuntimeError(
            "Empty train or test split - lower TEST_FRACTION or fetch more data."
        )
    logger.info(
        "Split: %d train / %d test rows (embargo from %s to %s)",
        len(X_tr), len(X_te), embargo_start, test_start,
    )
    return X_tr, X_te, y_tr, y_te


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def _scale_pos_weight(y: pd.Series) -> float:
    neg = int((y == 0).sum())
    pos = int((y == 1).sum())
    return neg / max(pos, 1)


def _lgbm_params(scale_pos_weight: float, **overrides) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "objective": "binary",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 40,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "n_estimators": 400,
        "scale_pos_weight": scale_pos_weight,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
    }
    params.update(overrides)
    return params


def _xgb_params(scale_pos_weight: float, **overrides) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 5,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "n_estimators": 400,
        "scale_pos_weight": scale_pos_weight,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    params.update(overrides)
    return params


def tune_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = N_OPTUNA_TRIALS,
    cv_splits: int = CV_SPLITS,
) -> Dict[str, Any]:
    """Optuna search maximising mean ROC AUC over TimeSeriesSplit CV."""
    import optuna  # imported lazily: only needed when tuning is requested

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    spw = _scale_pos_weight(y_train)
    tscv = TimeSeriesSplit(n_splits=cv_splits)

    def objective(trial: "optuna.Trial") -> float:
        params = _lgbm_params(
            spw,
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 200),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
            n_estimators=trial.suggest_int("n_estimators", 200, 800),
        )
        aucs = []
        for tr_idx, va_idx in tscv.split(X_train):
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            proba = model.predict_proba(X_train.iloc[va_idx])[:, 1]
            aucs.append(roc_auc_score(y_train.iloc[va_idx], proba))
        return float(np.mean(aucs))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    logger.info(
        "Optuna best CV AUC %.4f with params: %s",
        study.best_value, study.best_params,
    )
    return study.best_params


def evaluate(model: Any, X_test: pd.DataFrame, y_test: pd.Series, name: str) -> Dict[str, float]:
    """Compute standard binary-classification metrics on the test set."""
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(y_test, proba))
            if y_test.nunique() > 1 else float("nan")
        ),
        "log_loss": float(log_loss(y_test, proba)),
        "brier": float(brier_score_loss(y_test, proba)),
    }
    pretty = {k: (round(v, 4) if not pd.isna(v) else v) for k, v in metrics.items()}
    logger.info("%s test metrics: %s", name, pretty)
    return metrics


# ---------------------------------------------------------------------------
# Persistence & inference
# ---------------------------------------------------------------------------

def save_model(
    model: Any,
    feature_cols: list,
    metrics: Dict[str, float],
    path: str = DEFAULT_MODEL_PATH,
) -> str:
    """Persist model + feature list + label config as a single joblib bundle."""
    bundle = {
        "model": model,
        "feature_columns": list(feature_cols),
        "label_config": {
            "pt_atr_mult": PT_ATR_MULT,
            "sl_atr_mult": SL_ATR_MULT,
            "horizon": LABEL_HORIZON,
        },
        "test_metrics": metrics,
        "trained_at": pd.Timestamp.utcnow().isoformat(),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    logger.info("Saved best model to %s", path)
    return path


def load_model(path: str = DEFAULT_MODEL_PATH) -> Dict[str, Any]:
    """Load a bundle written by save_model()."""
    return joblib.load(path)


def predict_latest(features_df: pd.DataFrame, bundle: Dict[str, Any]) -> float:
    """
    Probability of an 'up' outcome for the most recent bar of a feature frame.

    Returns NaN if the latest row is incomplete (missing feature values).
    """
    cols = bundle["feature_columns"]
    row = features_df[cols].iloc[[-1]]
    if row.isna().to_numpy().any():
        logger.warning("Latest feature row is incomplete - returning NaN.")
        return float("nan")
    return float(bundle["model"].predict_proba(row)[0, 1])


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def train_models(
    symbols: list,
    data: Dict[Tuple[str, str], pd.DataFrame],
    features: Dict[str, pd.DataFrame],
    tune: bool = False,
    model_path: str = DEFAULT_MODEL_PATH,
) -> Tuple[str, Any, Dict[str, Dict[str, float]]]:
    """
    Label, split, train LightGBM + XGBoost, evaluate, and persist the best.

    Returns (best_name, best_model, metrics_by_model).
    """
    X, y, _meta = assemble_dataset(symbols, data, features)
    X_tr, X_te, y_tr, y_te = chronological_split(X, y)
    spw = _scale_pos_weight(y_tr)

    # --- LightGBM (optionally tuned) -----------------------------------------
    best_params = tune_lightgbm(X_tr, y_tr) if tune else {}
    lgb_model = lgb.LGBMClassifier(**_lgbm_params(spw, **best_params))
    lgb_model.fit(X_tr, y_tr)

    # --- XGBoost (sane defaults, no tuning) ------------------------------------
    xgb_model = xgb.XGBClassifier(**_xgb_params(spw))
    xgb_model.fit(X_tr, y_tr)

    # --- evaluation & model selection -------------------------------------------
    metrics = {
        "lightgbm": evaluate(lgb_model, X_te, y_te, "LightGBM"),
        "xgboost": evaluate(xgb_model, X_te, y_te, "XGBoost"),
    }

    def _auc(m: Dict[str, float]) -> float:
        v = m.get("roc_auc", float("nan"))
        return 0.5 if pd.isna(v) else v

    best_name = max(metrics, key=lambda n: _auc(metrics[n]))
    best_model = lgb_model if best_name == "lightgbm" else xgb_model
    logger.info("Best model by ROC AUC: %s", best_name)

    feature_cols = list(X.columns)
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(
            best_model.feature_importances_, index=feature_cols
        ).sort_values(ascending=False)
        logger.info(
            "Top 15 features (%s):\n%s", best_name, importances.head(15).to_string()
        )

    save_model(best_model, feature_cols, metrics[best_name], model_path)
    return best_name, best_model, metrics


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

async def _main(tune: bool = False) -> None:
    from data_pipeline import load_market_data

    logger.info("Stage 1/3: downloading market data ...")
    symbols, data = await load_market_data()

    logger.info("Stage 2/3: building features ...")
    features = build_features(symbols, data)

    logger.info("Stage 3/3: labeling + training ...")
    best_name, _model, metrics = train_models(symbols, data, features, tune=tune)

    print("\n" + "=" * 78)
    print("TASK 4 SMOKE TEST - TEST-SET COMPARISON")
    print("=" * 78)
    header = f"{'model':<12} {'accuracy':>9} {'precision':>10} {'recall':>8} {'f1':>7} {'roc_auc':>8} {'logloss':>8}"
    print(header)
    print("-" * len(header))
    for name, m in metrics.items():
        auc = m["roc_auc"]
        auc_str = f"{auc:>8.4f}" if not pd.isna(auc) else f"{'n/a':>8}"
        print(
            f"{name:<12} {m['accuracy']:>9.4f} {m['precision']:>10.4f} "
            f"{m['recall']:>8.4f} {m['f1']:>7.4f} {auc_str} {m['log_loss']:>8.4f}"
        )
    print("-" * len(header))
    print(f"Best model: {best_name} (saved to {DEFAULT_MODEL_PATH})")
    print("=" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Task 4 smoke test: triple-barrier labels + model training"
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="run the Optuna hyper-parameter search for LightGBM (slower)",
    )
    args = parser.parse_args()
    try:
        asyncio.run(_main(tune=args.tune))
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
