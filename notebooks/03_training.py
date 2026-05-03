import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Entrenamiento y Evaluación — IEEE-CIS Fraud Detection

        Este notebook entrena tres modelos sobre los datos preprocesados:

        1. **Logistic Regression** — baseline lineal (con imputación + escalado).
        2. **Random Forest** — baseline ensemble tradicional (con imputación).
        3. **XGBoost** — modelo avanzado, entrenado en **GPU (CUDA 12.8)**.

        Los modelos se evalúan sobre el conjunto de validación para selección
        y sobre test para reportar la métrica final. Adicionalmente se hace una
        **evaluación temporal por semana** sobre test para detectar degradación.

        Métricas elegidas (dataset desbalanceado): **ROC-AUC**, **PR-AUC**,
        **F1**, **balanced accuracy**, precision y recall.
        """
    )
    return


@app.cell
def _():
    import json
    import time
    from pathlib import Path

    import joblib
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        f1_score,
        balanced_accuracy_score,
        precision_score,
        recall_score,
        confusion_matrix,
    )
    import xgboost as xgb

    sns.set_theme(style="whitegrid")

    ARTIFACTS_DIR = Path("../artifacts")
    RANDOM_STATE = 42

    return (
        ARTIFACTS_DIR,
        LogisticRegression,
        Path,
        Pipeline,
        RANDOM_STATE,
        RandomForestClassifier,
        SimpleImputer,
        StandardScaler,
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        joblib,
        json,
        np,
        pd,
        plt,
        precision_score,
        recall_score,
        roc_auc_score,
        sns,
        time,
        xgb,
    )


@app.cell
def _(mo):
    mo.md(r"""## 1. Carga de datos preprocesados""")
    return


@app.cell
def _(ARTIFACTS_DIR, joblib, pd):
    train_df = pd.read_parquet(ARTIFACTS_DIR / "train.parquet")
    val_df = pd.read_parquet(ARTIFACTS_DIR / "val.parquet")
    test_df = pd.read_parquet(ARTIFACTS_DIR / "test.parquet")
    preprocessor = joblib.load(ARTIFACTS_DIR / "preprocessor.joblib")

    feature_cols = preprocessor["feature_cols"]
    cat_cols = preprocessor["cat_cols"]
    num_cols = preprocessor["num_cols"]

    {
        "train": train_df.shape,
        "val": val_df.shape,
        "test": test_df.shape,
        "n_features": len(feature_cols),
    }
    return (
        cat_cols,
        feature_cols,
        num_cols,
        preprocessor,
        test_df,
        train_df,
        val_df,
    )


@app.cell
def _(feature_cols, test_df, train_df, val_df):
    X_train = train_df[feature_cols]
    y_train = train_df["isFraud"].values

    X_val = val_df[feature_cols]
    y_val = val_df["isFraud"].values

    X_test = test_df[feature_cols]
    y_test = test_df["isFraud"].values

    {
        "X_train": X_train.shape,
        "y_train_pos_rate": float(y_train.mean()),
        "X_val": X_val.shape,
        "y_val_pos_rate": float(y_val.mean()),
        "X_test": X_test.shape,
        "y_test_pos_rate": float(y_test.mean()),
    }
    return X_test, X_train, X_val, y_test, y_train, y_val


@app.cell
def _(
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
):
    def evaluate(y_true, y_proba, threshold=0.5):
        y_pred = (y_proba >= threshold).astype(int)
        return {
            "roc_auc": roc_auc_score(y_true, y_proba),
            "pr_auc": average_precision_score(y_true, y_proba),
            "f1": f1_score(y_true, y_pred),
            "balanced_acc": balanced_accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred),
        }

    return (evaluate,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 2. Modelo 1 — Logistic Regression

        Como baseline lineal usamos un pipeline con:

        - `SimpleImputer(strategy="median")` para nulos numéricos
        - `StandardScaler` para que los coeficientes sean comparables
        - `LogisticRegression` con `class_weight="balanced"` para compensar el desbalance

        Entrenamos en CPU (es rápido para problemas convexos a esta escala).
        """
    )
    return


@app.cell
def _(
    LogisticRegression,
    Pipeline,
    RANDOM_STATE,
    SimpleImputer,
    StandardScaler,
    X_train,
    X_val,
    evaluate,
    time,
    y_train,
    y_val,
):
    logreg_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )),
    ])

    t0 = time.time()
    logreg_pipeline.fit(X_train, y_train)
    logreg_train_time = time.time() - t0

    logreg_val_proba = logreg_pipeline.predict_proba(X_val)[:, 1]
    logreg_val_metrics = evaluate(y_val, logreg_val_proba)
    logreg_val_metrics["train_time_s"] = round(logreg_train_time, 1)
    logreg_val_metrics
    return (
        logreg_pipeline,
        logreg_train_time,
        logreg_val_metrics,
        logreg_val_proba,
        t0,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 3. Modelo 2 — Random Forest

        Ensemble tradicional. Manejamos los nulos con imputación por mediana y dejamos
        `class_weight="balanced_subsample"` para el desbalance. Limitamos `max_depth`
        para evitar overfitting y mantener tiempos razonables.
        """
    )
    return


@app.cell
def _(
    Pipeline,
    RANDOM_STATE,
    RandomForestClassifier,
    SimpleImputer,
    X_train,
    X_val,
    evaluate,
    time,
    y_train,
    y_val,
):
    rf_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=16,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )),
    ])

    t1 = time.time()
    rf_pipeline.fit(X_train, y_train)
    rf_train_time = time.time() - t1

    rf_val_proba = rf_pipeline.predict_proba(X_val)[:, 1]
    rf_val_metrics = evaluate(y_val, rf_val_proba)
    rf_val_metrics["train_time_s"] = round(rf_train_time, 1)
    rf_val_metrics
    return rf_pipeline, rf_train_time, rf_val_metrics, rf_val_proba, t1


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 4. Modelo 3 — XGBoost (GPU, CUDA 12.8)

        Modelo avanzado. Maneja NaN de forma nativa, no necesita imputación ni escalado.
        Usamos:

        - `device="cuda"` y `tree_method="hist"` para aprovechar la RTX 4070
        - `scale_pos_weight` calculado como `n_neg / n_pos` para el desbalance
        - **Early stopping** sobre el set de validación para evitar overfitting

        Si XGBoost no detecta CUDA, lanzará un error explícito; verifica entonces la instalación.
        """
    )
    return


@app.cell
def _(X_train, X_val, evaluate, time, xgb, y_train, y_val):
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos

    xgb_clf = xgb.XGBClassifier(
        n_estimators=2000,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.7,
        min_child_weight=5,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        device="cuda",
        eval_metric="aucpr",
        early_stopping_rounds=100,
        random_state=42,
    )

    t2 = time.time()
    xgb_clf.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )
    xgb_train_time = time.time() - t2

    xgb_val_proba = xgb_clf.predict_proba(X_val)[:, 1]
    xgb_val_metrics = evaluate(y_val, xgb_val_proba)
    xgb_val_metrics["train_time_s"] = round(xgb_train_time, 1)
    xgb_val_metrics["best_iteration"] = int(xgb_clf.best_iteration)
    xgb_val_metrics
    return (
        n_neg,
        n_pos,
        scale_pos_weight,
        t2,
        xgb_clf,
        xgb_train_time,
        xgb_val_metrics,
        xgb_val_proba,
    )


@app.cell
def _(mo):
    mo.md(r"""## 5. Comparación en validación""")
    return


@app.cell
def _(logreg_val_metrics, pd, rf_val_metrics, xgb_val_metrics):
    val_comparison = pd.DataFrame({
        "logistic_regression": logreg_val_metrics,
        "random_forest": rf_val_metrics,
        "xgboost_gpu": xgb_val_metrics,
    }).T.round(4)
    val_comparison
    return (val_comparison,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 6. Evaluación final en test

        Reportamos métricas de los tres modelos sobre el conjunto de test (último mes).
        Es esperable cierta degradación frente a validación si hay drift en este periodo.
        """
    )
    return


@app.cell
def _(
    X_test,
    evaluate,
    logreg_pipeline,
    pd,
    rf_pipeline,
    xgb_clf,
    y_test,
):
    logreg_test_proba = logreg_pipeline.predict_proba(X_test)[:, 1]
    rf_test_proba = rf_pipeline.predict_proba(X_test)[:, 1]
    xgb_test_proba = xgb_clf.predict_proba(X_test)[:, 1]

    test_comparison = pd.DataFrame({
        "logistic_regression": evaluate(y_test, logreg_test_proba),
        "random_forest": evaluate(y_test, rf_test_proba),
        "xgboost_gpu": evaluate(y_test, xgb_test_proba),
    }).T.round(4)
    test_comparison
    return (
        logreg_test_proba,
        rf_test_proba,
        test_comparison,
        xgb_test_proba,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 7. Evaluación temporal por semana (XGBoost)

        Tomamos el set de test y lo dividimos en bloques semanales. Para cada semana
        calculamos AUC y PR-AUC. Si las métricas caen consistentemente con el tiempo,
        tenemos evidencia de **degradación por drift**.
        """
    )
    return


@app.cell
def _(
    average_precision_score,
    pd,
    roc_auc_score,
    test_df,
    xgb_test_proba,
    y_test,
):
    weekly = test_df[["timestamp", "isFraud"]].copy()
    weekly["proba"] = xgb_test_proba
    weekly["week"] = weekly["timestamp"].dt.to_period("W").dt.start_time

    def _safe_auc(g):
        if g["isFraud"].nunique() < 2:
            return None
        return roc_auc_score(g["isFraud"], g["proba"])

    def _safe_pr_auc(g):
        if g["isFraud"].nunique() < 2:
            return None
        return average_precision_score(g["isFraud"], g["proba"])

    weekly_metrics = (
        weekly.groupby("week")
        .apply(lambda g: pd.Series({
            "n": len(g),
            "fraud_rate": g["isFraud"].mean(),
            "roc_auc": _safe_auc(g),
            "pr_auc": _safe_pr_auc(g),
        }))
        .reset_index()
    )
    weekly_metrics
    return _safe_auc, _safe_pr_auc, weekly, weekly_metrics


@app.cell
def _(plt, weekly_metrics):
    fig_w, axes_w = plt.subplots(1, 2, figsize=(14, 4))

    axes_w[0].plot(weekly_metrics["week"], weekly_metrics["roc_auc"], marker="o", color="#2E86AB")
    axes_w[0].set_title("ROC-AUC semanal en test (XGBoost)")
    axes_w[0].set_ylabel("ROC-AUC")
    axes_w[0].tick_params(axis="x", rotation=30)

    axes_w[1].plot(weekly_metrics["week"], weekly_metrics["pr_auc"], marker="o", color="#E63946")
    axes_w[1].set_title("PR-AUC semanal en test (XGBoost)")
    axes_w[1].set_ylabel("PR-AUC")
    axes_w[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    fig_w
    return axes_w, fig_w


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 8. Importancia de features (XGBoost)

        Top 20 features según `gain`. Útil para entender qué señales está usando el modelo
        y para discutir interpretabilidad en el informe.
        """
    )
    return


@app.cell
def _(feature_cols, pd, plt, xgb_clf):
    importance = pd.DataFrame({
        "feature": feature_cols,
        "gain": xgb_clf.feature_importances_,
    }).sort_values("gain", ascending=False).head(20)

    fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
    ax_imp.barh(importance["feature"][::-1], importance["gain"][::-1], color="#2E86AB")
    ax_imp.set_title("Top 20 features por importancia (XGBoost)")
    ax_imp.set_xlabel("Gain")
    plt.tight_layout()
    fig_imp
    return ax_imp, fig_imp, importance


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 9. Persistencia del modelo final

        Guardamos en `artifacts/`:

        - `xgb_model.json` — modelo XGBoost en formato portable (no atado a la versión de pickle)
        - `xgb_model.joblib` — wrapper sklearn (para usar `predict_proba` directo)
        - `metrics.json` — métricas de val/test y evaluación semanal
        - `weekly_metrics.csv` — para gráficas y reporte
        """
    )
    return


@app.cell
def _(
    ARTIFACTS_DIR,
    joblib,
    json,
    test_comparison,
    val_comparison,
    weekly_metrics,
    xgb_clf,
):
    xgb_clf.save_model(ARTIFACTS_DIR / "xgb_model.json")
    joblib.dump(xgb_clf, ARTIFACTS_DIR / "xgb_model.joblib")

    weekly_metrics.to_csv(ARTIFACTS_DIR / "weekly_metrics.csv", index=False)

    metrics_summary = {
        "validation": val_comparison.to_dict(orient="index"),
        "test": test_comparison.to_dict(orient="index"),
        "best_iteration_xgb": int(xgb_clf.best_iteration),
    }
    with open(ARTIFACTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False, default=float)

    {
        "guardado": [
            "xgb_model.json",
            "xgb_model.joblib",
            "metrics.json",
            "weekly_metrics.csv",
        ],
    }
    return f, metrics_summary


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 10. Conclusiones

        Resultados obtenidos sobre validación y test:

        | Modelo | ROC-AUC val | ROC-AUC test | PR-AUC val | PR-AUC test | F1 test |
        |---|---|---|---|---|---|
        | Logistic Regression | 0,8425 | 0,8223 | 0,3490 | 0,1710 | 0,1872 |
        | Random Forest | 0,8819 | 0,8810 | 0,4299 | 0,4864 | 0,3597 |
        | **XGBoost (GPU)** | **0,9129** | **0,8962** | **0,5817** | **0,5469** | **0,5422** |

        - **XGBoost domina en todas las métricas relevantes** (ROC-AUC, PR-AUC, F1).
          Su PR-AUC en test (0,547) es 12% superior al de Random Forest (0,486) y
          3,2× superior al de Logistic Regression (0,171), confirmando la necesidad
          de capturar interacciones no lineales en este problema.
        - **Logistic Regression** logra alto recall (0,737) pero precisión muy baja (0,107),
          lo que indica que un modelo lineal no es viable para producción aquí: marcaría
          demasiados falsos positivos.
        - **Caída entre val y test** del XGBoost: ROC-AUC -1,7%, PR-AUC -6,0%. La caída
          es asimétrica y mayor en PR-AUC, primer indicio de drift que motiva el siguiente notebook.
        - **Evaluación semanal en test** (XGBoost): ROC-AUC oscila entre 0,868 (semana del
          14-may, peor) y 0,924 (semana del 30-abr, mejor). El patrón **no es monotónico**:
          hay una caída en mid-mayo y una recuperación parcial al final del periodo.
        - **Mejor iteración** de XGBoost: 1.995 árboles (early stopping). Se guarda fija
          en `metrics.json` para los reentrenamientos del siguiente notebook.

        **Modelo elegido para despliegue:** XGBoost (GPU CUDA 12.8).

        **Próximo notebook:** `04_drift_analysis.py` — KS, PSI, drift en predicciones y
        simulación estática vs adaptativa.
        """
    )
    return