import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
        # Preprocesamiento — IEEE-CIS Fraud Detection

        Este notebook prepara los datos para los modelos. Decisiones clave:

        1. **Split temporal estricto** train / val / test (sin shuffle).
        2. **Drop de columnas con >90% nulos**, calculado **solo sobre train** para evitar leakage.
        3. **Label encoding** de categóricas (compatible con XGBoost/LightGBM, sin one-hot).
        4. **Feature engineering temporal:** hora, día de semana, día del mes, mes y `days_since_start`.
        5. **Optimización de dtypes** (float32, int32) para reducir memoria.
        6. **Persistencia en `artifacts/`**: parquets + `preprocessor.joblib` con encoders y metadata.

        El resto de notebooks (training, drift) cargarán directamente los parquets sin re-procesar.
        """)
    return


@app.cell
def _():
    import json
    from pathlib import Path

    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import OrdinalEncoder

    DATA_DIR = Path("../data")
    ARTIFACTS_DIR = Path("../artifacts")
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    REFERENCE_DATE = pd.Timestamp("2017-12-01")

    TRAIN_END = pd.Timestamp("2018-04-08 23:59:59")
    VAL_END = pd.Timestamp("2018-04-30 23:59:59")

    HIGH_NULL_THRESHOLD = 0.90
    return (
        ARTIFACTS_DIR,
        DATA_DIR,
        HIGH_NULL_THRESHOLD,
        OrdinalEncoder,
        Path,
        REFERENCE_DATE,
        TRAIN_END,
        VAL_END,
        joblib,
        json,
        np,
        pd,
    )


@app.cell
def _(mo):
    mo.md(r"""## 1. Carga y construcción del DataFrame base""")
    return


@app.cell
def _(DATA_DIR, REFERENCE_DATE, pd):
    transactions = pd.read_csv(DATA_DIR / "train_transaction.csv")
    identity = pd.read_csv(DATA_DIR / "train_identity.csv")
    df = transactions.merge(identity, on="TransactionID", how="left")

    df["timestamp"] = REFERENCE_DATE + pd.to_timedelta(df["TransactionDT"], unit="s")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.shape
    return df, identity, transactions


@app.cell
def _(mo):
    mo.md(r"""
        ## 2. Feature engineering temporal

        Generamos variables derivadas del timestamp que el modelo puede aprovechar directamente.
        `days_since_start` captura tendencias seculares; las otras capturan estacionalidad.
        """)
    return


@app.cell
def _(df):
    df_fe = df.copy()
    df_fe["hour"] = df_fe["timestamp"].dt.hour
    df_fe["dayofweek"] = df_fe["timestamp"].dt.dayofweek
    df_fe["day"] = df_fe["timestamp"].dt.day
    df_fe["month"] = df_fe["timestamp"].dt.month
    df_fe["days_since_start"] = (df_fe["timestamp"] - df_fe["timestamp"].min()).dt.days

    df_fe[["timestamp", "hour", "dayofweek", "day", "month", "days_since_start"]].head()
    return (df_fe,)


@app.cell
def _(mo):
    mo.md(r"""
        ## 3. Split temporal

        - **Train:** desde el inicio hasta `2018-04-08` (~70% de los días)
        - **Val:**   `2018-04-09` → `2018-04-30` (~12%)
        - **Test:**  `2018-05-01` en adelante (~18%)

        Este corte respeta el orden cronológico: no hay ninguna fila de val/test anterior a una de train.
        """)
    return


@app.cell
def _(TRAIN_END, VAL_END, df_fe):
    train_mask = df_fe["timestamp"] <= TRAIN_END
    val_mask = (df_fe["timestamp"] > TRAIN_END) & (df_fe["timestamp"] <= VAL_END)
    test_mask = df_fe["timestamp"] > VAL_END

    train_df = df_fe[train_mask].reset_index(drop=True)
    val_df = df_fe[val_mask].reset_index(drop=True)
    test_df = df_fe[test_mask].reset_index(drop=True)

    split_summary = {
        "train": {
            "filas": len(train_df),
            "fecha_min": str(train_df["timestamp"].min()),
            "fecha_max": str(train_df["timestamp"].max()),
            "fraud_rate": round(train_df["isFraud"].mean(), 4),
        },
        "val": {
            "filas": len(val_df),
            "fecha_min": str(val_df["timestamp"].min()),
            "fecha_max": str(val_df["timestamp"].max()),
            "fraud_rate": round(val_df["isFraud"].mean(), 4),
        },
        "test": {
            "filas": len(test_df),
            "fecha_min": str(test_df["timestamp"].min()),
            "fecha_max": str(test_df["timestamp"].max()),
            "fraud_rate": round(test_df["isFraud"].mean(), 4),
        },
    }
    split_summary
    return (
        split_summary,
        test_df,
        test_mask,
        train_df,
        train_mask,
        val_df,
        val_mask,
    )


@app.cell
def _(mo):
    mo.md(r"""
        ## 4. Drop de columnas con demasiados nulos

        Calculamos el % de nulos **solo sobre train** y eliminamos las columnas que superan el
        umbral en los tres splits. Hacerlo solo sobre train evita un sutil data leakage: una columna
        podría tener pocos nulos en train pero muchos en test (o viceversa) y eso es info futura.
        """)
    return


@app.cell
def _(HIGH_NULL_THRESHOLD, train_df):
    null_pct_train = train_df.isna().mean()
    high_null_cols = null_pct_train[null_pct_train > HIGH_NULL_THRESHOLD].index.tolist()
    {
        "umbral": HIGH_NULL_THRESHOLD,
        "columnas_eliminadas": len(high_null_cols),
        "primeras_10": high_null_cols[:10],
    }
    return high_null_cols, null_pct_train


@app.cell
def _(high_null_cols, test_df, train_df, val_df):
    train_clean = train_df.drop(columns=high_null_cols)
    val_clean = val_df.drop(columns=high_null_cols)
    test_clean = test_df.drop(columns=high_null_cols)
    {
        "train_cols": train_clean.shape[1],
        "val_cols": val_clean.shape[1],
        "test_cols": test_clean.shape[1],
    }
    return test_clean, train_clean, val_clean


@app.cell
def _(mo):
    mo.md(r"""
        ## 5. Identificación de columnas categóricas y numéricas

        Excluimos del set de features:

        - `TransactionID` (identificador, no informativo)
        - `isFraud` (target)
        - `timestamp` (no se pasa al modelo; ya extrajimos sus componentes)
        - `TransactionDT` (redundante con `days_since_start` y derivados)
        """)
    return


@app.cell
def _(train_clean):
    DROP_FROM_FEATURES = ["TransactionID", "isFraud", "timestamp", "TransactionDT"]

    feature_cols = [c for c in train_clean.columns if c not in DROP_FROM_FEATURES]
    cat_cols = [c for c in feature_cols if train_clean[c].dtype == "object"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    {
        "total_features": len(feature_cols),
        "categóricas": len(cat_cols),
        "numéricas": len(num_cols),
    }
    return DROP_FROM_FEATURES, cat_cols, feature_cols, num_cols


@app.cell
def _(mo):
    mo.md(r"""
        ## 6. Label encoding

        - Rellenamos NaN en categóricas con el sentinela `"__missing__"` antes de encodear.
        - Usamos `OrdinalEncoder` ajustado **solo sobre train**.
        - Categorías nunca vistas en val/test se mapean a `-1` (`handle_unknown="use_encoded_value"`).

        XGBoost puede manejar enteros directamente, así que no hace falta one-hot.
        """)
    return


@app.cell
def _(OrdinalEncoder, cat_cols, test_clean, train_clean, val_clean):
    train_enc = train_clean.copy()
    val_enc = val_clean.copy()
    test_enc = test_clean.copy()

    for c in cat_cols:
        train_enc[c] = train_enc[c].fillna("__missing__").astype(str)
        val_enc[c] = val_enc[c].fillna("__missing__").astype(str)
        test_enc[c] = test_enc[c].fillna("__missing__").astype(str)

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        dtype="int32",
    )
    encoder.fit(train_enc[cat_cols])

    train_enc[cat_cols] = encoder.transform(train_enc[cat_cols])
    val_enc[cat_cols] = encoder.transform(val_enc[cat_cols])
    test_enc[cat_cols] = encoder.transform(test_enc[cat_cols])

    {"categorías_por_columna": {c: len(cats) for c, cats in zip(cat_cols, encoder.categories_)}}
    return c, encoder, test_enc, train_enc, val_enc


@app.cell
def _(mo):
    mo.md(r"""
        ## 7. Optimización de dtypes

        Convertimos float64 → float32 e int64 → int32 donde sea seguro. Esto reduce el uso
        de memoria a aproximadamente la mitad y acelera el entrenamiento en GPU.
        """)
    return


@app.cell
def _(np, num_cols, test_enc, train_enc, val_enc):
    def downcast(df_in, num_cols_):
        df_out = df_in.copy()
        for col in num_cols_:
            if df_out[col].dtype == "float64":
                df_out[col] = df_out[col].astype(np.float32)
            elif df_out[col].dtype == "int64":
                df_out[col] = df_out[col].astype(np.int32)
        return df_out

    train_final = downcast(train_enc, num_cols)
    val_final = downcast(val_enc, num_cols)
    test_final = downcast(test_enc, num_cols)

    {
        "train_mb": round(train_final.memory_usage(deep=True).sum() / 1024**2, 1),
        "val_mb": round(val_final.memory_usage(deep=True).sum() / 1024**2, 1),
        "test_mb": round(test_final.memory_usage(deep=True).sum() / 1024**2, 1),
    }
    return downcast, test_final, train_final, val_final


@app.cell
def _(mo):
    mo.md(r"""
        ## 8. Persistencia de artifacts

        Guardamos:

        - `train.parquet`, `val.parquet`, `test.parquet` — datasets listos para modelar
        - `preprocessor.joblib` — diccionario con el encoder, columnas droppeadas, listas de
          features (categóricas / numéricas) y fechas de corte
        - `metadata.json` — resumen legible de los splits y decisiones
        """)
    return


@app.cell
def _(
    ARTIFACTS_DIR,
    HIGH_NULL_THRESHOLD,
    TRAIN_END,
    VAL_END,
    cat_cols,
    encoder,
    feature_cols,
    high_null_cols,
    joblib,
    json,
    num_cols,
    split_summary,
    test_final,
    train_final,
    val_final,
):
    train_final.to_parquet(ARTIFACTS_DIR / "train.parquet", index=False)
    val_final.to_parquet(ARTIFACTS_DIR / "val.parquet", index=False)
    test_final.to_parquet(ARTIFACTS_DIR / "test.parquet", index=False)

    preprocessor = {
        "encoder": encoder,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "feature_cols": feature_cols,
        "dropped_high_null_cols": high_null_cols,
        "high_null_threshold": HIGH_NULL_THRESHOLD,
        "train_end": str(TRAIN_END),
        "val_end": str(VAL_END),
    }
    joblib.dump(preprocessor, ARTIFACTS_DIR / "preprocessor.joblib")

    metadata = {
        "splits": split_summary,
        "n_features": len(feature_cols),
        "n_cat_features": len(cat_cols),
        "n_num_features": len(num_cols),
        "n_dropped_cols": len(high_null_cols),
        "high_null_threshold": HIGH_NULL_THRESHOLD,
        "train_end": str(TRAIN_END),
        "val_end": str(VAL_END),
    }
    with open(ARTIFACTS_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    {
        "guardado_en": str(ARTIFACTS_DIR.resolve()),
        "archivos": [
            "train.parquet",
            "val.parquet",
            "test.parquet",
            "preprocessor.joblib",
            "metadata.json",
        ],
    }
    return f, metadata, preprocessor


@app.cell
def _(mo):
    mo.md(r"""
        ## 9. Resumen del preprocesamiento

        Resultados concretos del pipeline:

        - **Split temporal:** train 440.018 filas (74,5%), val 58.095 (9,8%), test 92.427 (15,7%).
          Proporciones cercanas al objetivo 70/12/18 con cortes el 2018-04-08 y el 2018-04-30.
        - **Tasa de fraude por split** muy estable: 3,50% / 3,52% / 3,48%. Confirma que la
          variación temporal del target es de alta frecuencia (semanal/diaria) y no de bloque,
          por lo que la evaluación temporal posterior es informativa.
        - **Drop de columnas:** solo **12 columnas** superaron el umbral de 90% de nulos en train.
          La mayoría del dataset es usable; XGBoost manejará los nulos remanentes nativamente.
        - **Features finales:** 424 (29 categóricas + 395 numéricas), tras excluir
          `TransactionID`, `isFraud`, `timestamp` y `TransactionDT`.
        - **Persistencia:** `train.parquet`, `val.parquet`, `test.parquet`,
          `preprocessor.joblib` y `metadata.json` en `artifacts/`. Los notebooks 03 y 04
          consumen estos archivos directamente sin re-procesar.

        **Próximo notebook:** `03_training.py` — Logistic Regression, Random Forest y XGBoost
        (GPU CUDA 12.8) con evaluación en val/test y por bloques semanales.
        """)
    return


if __name__ == "__main__":
    app.run()
