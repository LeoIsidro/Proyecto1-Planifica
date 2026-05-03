import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import OrdinalEncoder

DROP_FROM_FEATURES = ["TransactionID", "isFraud", "timestamp", "TransactionDT"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reentrenamiento end-to-end del modelo de fraude. "
            "Reproduce el pipeline de los notebooks 02 y 03 sin marimo."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--reference-date", type=str, default="2017-12-01")
    parser.add_argument(
        "--train-end",
        type=str,
        default=None,
        help="Fecha límite del set de train (YYYY-MM-DD). Si no se pasa, usa el 80% del periodo.",
    )
    parser.add_argument("--val-days", type=int, default=22)
    parser.add_argument("--high-null-threshold", type=float, default=0.90)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Dispositivo XGBoost. Cambiar a cpu si CUDA no está disponible.",
    )
    parser.add_argument("--early-stopping", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_and_merge(data_dir: Path, reference_date: str) -> pd.DataFrame:
    transactions = pd.read_csv(data_dir / "train_transaction.csv")
    identity = pd.read_csv(data_dir / "train_identity.csv")
    df = transactions.merge(identity, on="TransactionID", how="left")
    df["timestamp"] = pd.Timestamp(reference_date) + pd.to_timedelta(
        df["TransactionDT"], unit="s"
    )
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["days_since_start"] = (df["timestamp"] - df["timestamp"].min()).dt.days
    return df


def temporal_split(
    df: pd.DataFrame, train_end: pd.Timestamp, val_days: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    val_end = train_end + pd.Timedelta(days=val_days)
    train_df = df[df["timestamp"] <= train_end].reset_index(drop=True)
    val_df = df[
        (df["timestamp"] > train_end) & (df["timestamp"] <= val_end)
    ].reset_index(drop=True)
    return train_df, val_df


def fit_preprocessor(train_df: pd.DataFrame, high_null_threshold: float) -> dict:
    null_pct = train_df.isna().mean()
    high_null_cols = null_pct[null_pct > high_null_threshold].index.tolist()
    train_clean = train_df.drop(columns=high_null_cols)

    feature_cols = [c for c in train_clean.columns if c not in DROP_FROM_FEATURES]
    cat_cols = [c for c in feature_cols if train_clean[c].dtype == "object"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        dtype="int32",
    )
    encoder.fit(train_clean[cat_cols].fillna("__missing__").astype(str))

    return {
        "encoder": encoder,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "dropped_high_null_cols": high_null_cols,
        "high_null_threshold": high_null_threshold,
    }


def transform(df: pd.DataFrame, preprocessor: dict) -> pd.DataFrame:
    df = df.drop(columns=preprocessor["dropped_high_null_cols"], errors="ignore")
    cat_cols = preprocessor["cat_cols"]
    num_cols = preprocessor["num_cols"]

    for c in cat_cols:
        df[c] = df[c].fillna("__missing__").astype(str)
    df[cat_cols] = preprocessor["encoder"].transform(df[cat_cols])

    for c in num_cols:
        if df[c].dtype == "float64":
            df[c] = df[c].astype(np.float32)
        elif df[c].dtype == "int64":
            df[c] = df[c].astype(np.int32)
    return df


def train_xgboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    preprocessor: dict,
    args: argparse.Namespace,
) -> tuple[xgb.XGBClassifier, dict]:
    feature_cols = preprocessor["feature_cols"]
    X_train = train_df[feature_cols]
    y_train = train_df["isFraud"].values
    X_val = val_df[feature_cols]
    y_val = val_df["isFraud"].values

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos

    clf = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=0.9,
        colsample_bytree=0.7,
        min_child_weight=5,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        device=args.device,
        eval_metric="aucpr",
        early_stopping_rounds=args.early_stopping,
        random_state=args.random_state,
    )
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

    proba = clf.predict_proba(X_val)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_val, proba)),
        "pr_auc": float(average_precision_score(y_val, proba)),
        "f1": float(f1_score(y_val, pred)),
        "best_iteration": int(clf.best_iteration),
    }
    return clf, metrics


def main() -> None:
    args = parse_args()
    args.artifacts_dir.mkdir(exist_ok=True)

    log(f"Cargando CSVs desde {args.data_dir}")
    df = load_and_merge(args.data_dir, args.reference_date)
    df = add_temporal_features(df)

    if args.train_end is None:
        period_days = (df["timestamp"].max() - df["timestamp"].min()).days
        train_end = df["timestamp"].min() + pd.Timedelta(days=int(period_days * 0.8))
    else:
        train_end = pd.Timestamp(args.train_end + " 23:59:59")
    log(f"Split temporal: train hasta {train_end.date()}, val={args.val_days} días")

    train_df, val_df = temporal_split(df, train_end, args.val_days)
    log(
        f"  train={len(train_df):,} (fraud={train_df['isFraud'].mean():.4f})  "
        f"val={len(val_df):,} (fraud={val_df['isFraud'].mean():.4f})"
    )

    log("Ajustando preprocessor sobre train")
    preprocessor = fit_preprocessor(train_df, args.high_null_threshold)
    preprocessor["train_end"] = str(train_end)
    log(
        f"  features={len(preprocessor['feature_cols'])}  "
        f"cat={len(preprocessor['cat_cols'])}  "
        f"droppeadas={len(preprocessor['dropped_high_null_cols'])}"
    )

    train_df = transform(train_df, preprocessor)
    val_df = transform(val_df, preprocessor)

    log(f"Entrenando XGBoost (device={args.device}, n_estimators={args.n_estimators})")
    t0 = time.time()
    clf, metrics = train_xgboost(train_df, val_df, preprocessor, args)
    elapsed = time.time() - t0
    log(
        f"  done en {elapsed:.1f}s | "
        f"ROC-AUC={metrics['roc_auc']:.4f} | "
        f"PR-AUC={metrics['pr_auc']:.4f} | "
        f"F1={metrics['f1']:.4f}"
    )

    log("Guardando artifacts")
    joblib.dump(preprocessor, args.artifacts_dir / "preprocessor.joblib")
    joblib.dump(clf, args.artifacts_dir / "xgb_model.joblib")
    clf.save_model(str(args.artifacts_dir / "xgb_model.json"))

    metadata = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_end": str(train_end),
        "val_days": args.val_days,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "n_features": len(preprocessor["feature_cols"]),
        "n_dropped_cols": len(preprocessor["dropped_high_null_cols"]),
        "device": args.device,
        "metrics": metrics,
        "training_time_seconds": round(elapsed, 1),
    }
    with open(args.artifacts_dir / "retrain_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)

    log(f"Listo. Reentrenamiento completo en {time.time() - t0:.1f}s totales.")


if __name__ == "__main__":
    main()
