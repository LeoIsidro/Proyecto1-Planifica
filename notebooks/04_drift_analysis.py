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
        # Análisis y Adaptación al Concept Drift

        Este notebook evalúa formalmente la presencia de **concept drift** en el dataset
        IEEE-CIS y compara una estrategia de despliegue **estática** (modelo fijo) contra
        una **adaptativa** (reentrenamiento periódico).

        Análisis incluidos:

        1. **Drift en el target** (`label drift`) — cómo evoluciona P(isFraud=1) en el tiempo
        2. **Drift en features numéricas** — test de Kolmogorov-Smirnov train vs test
        3. **Drift en features categóricas** — test de chi-cuadrado
        4. **PSI (Population Stability Index)** — métrica estándar en finanzas
        5. **Drift en predicciones** — distribución de probabilidades del modelo
        6. **Simulación de adaptación** — reentrenamiento incremental por semana
        7. **Estrategia de trigger** — cuándo el sistema en producción debe reentrenar
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

    from scipy.stats import ks_2samp, chi2_contingency
    from sklearn.metrics import roc_auc_score, average_precision_score
    import xgboost as xgb

    sns.set_theme(style="whitegrid")

    ARTIFACTS_DIR = Path("../artifacts")
    return (
        ARTIFACTS_DIR,
        Path,
        average_precision_score,
        chi2_contingency,
        joblib,
        json,
        ks_2samp,
        np,
        pd,
        plt,
        roc_auc_score,
        sns,
        time,
        xgb,
    )


@app.cell
def _(mo):
    mo.md(r"""## 1. Carga de datos y modelo""")
    return


@app.cell
def _(ARTIFACTS_DIR, joblib, pd):
    train_df = pd.read_parquet(ARTIFACTS_DIR / "train.parquet")
    val_df = pd.read_parquet(ARTIFACTS_DIR / "val.parquet")
    test_df = pd.read_parquet(ARTIFACTS_DIR / "test.parquet")

    preprocessor = joblib.load(ARTIFACTS_DIR / "preprocessor.joblib")
    xgb_model = joblib.load(ARTIFACTS_DIR / "xgb_model.joblib")

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
        xgb_model,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 2. Drift en el target

        Evolución de la tasa de fraude semanal a lo largo de **todo el periodo** (train + val + test).
        Si la línea es estable: drift bajo. Si tiene tendencia o picos: drift fuerte.
        """
    )
    return


@app.cell
def _(pd, plt, test_df, train_df, val_df):
    timeline = pd.concat([
        train_df[["timestamp", "isFraud"]].assign(split="train"),
        val_df[["timestamp", "isFraud"]].assign(split="val"),
        test_df[["timestamp", "isFraud"]].assign(split="test"),
    ])
    timeline["week"] = timeline["timestamp"].dt.to_period("W").dt.start_time

    weekly_target = (
        timeline.groupby(["week", "split"])
        .agg(n=("isFraud", "size"), fraud_rate=("isFraud", "mean"))
        .reset_index()
    )

    fig_t, ax_t = plt.subplots(figsize=(14, 4))
    colors = {"train": "#2E86AB", "val": "#F4A261", "test": "#E63946"}
    for sp, sub in weekly_target.groupby("split"):
        ax_t.plot(sub["week"], sub["fraud_rate"], marker="o", label=sp, color=colors[sp])
    ax_t.set_title("Tasa de fraude semanal (train + val + test)")
    ax_t.set_ylabel("Tasa de fraude")
    ax_t.set_xlabel("Semana")
    ax_t.legend()
    fig_t
    return ax_t, colors, fig_t, timeline, weekly_target


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 3. Drift en features numéricas — Kolmogorov-Smirnov

        Para cada feature numérica importante comparamos su distribución en train vs test.
        El test KS devuelve un estadístico (0=idénticas, 1=disjuntas) y un p-value.

        Limitamos a las **top 30 features por importancia** del XGBoost para mantener el
        análisis enfocado en lo que el modelo realmente usa.
        """
    )
    return


@app.cell
def _(feature_cols, num_cols, pd, xgb_model):
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "gain": xgb_model.feature_importances_,
    }).sort_values("gain", ascending=False)

    top_num = [f for f in importance_df["feature"] if f in num_cols][:30]
    top_num
    return importance_df, top_num


@app.cell
def _(ks_2samp, np, pd, test_df, top_num, train_df):
    ks_results = []
    for feat in top_num:
        a = train_df[feat].dropna().values
        b = test_df[feat].dropna().values
        if len(a) == 0 or len(b) == 0:
            continue
        if len(a) > 50000:
            a = np.random.default_rng(42).choice(a, 50000, replace=False)
        if len(b) > 50000:
            b = np.random.default_rng(42).choice(b, 50000, replace=False)
        stat, pval = ks_2samp(a, b)
        ks_results.append({"feature": feat, "ks_stat": stat, "p_value": pval})

    ks_df = pd.DataFrame(ks_results).sort_values("ks_stat", ascending=False)
    ks_df["drift"] = ks_df["p_value"] < 0.01
    ks_df
    return feat, ks_df, ks_results, pval, stat


@app.cell
def _(ks_df):
    {
        "features_evaluadas": len(ks_df),
        "con_drift_significativo (p<0.01)": int(ks_df["drift"].sum()),
        "ks_stat_max": float(ks_df["ks_stat"].max()),
        "feature_mas_drift": ks_df.iloc[0]["feature"],
    }
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 4. Drift en features categóricas — Chi-cuadrado

        Para cada categórica importante comparamos la distribución de categorías
        entre train y test con un test de chi-cuadrado.
        """
    )
    return


@app.cell
def _(cat_cols, feature_cols, importance_df):
    top_cat = [f for f in importance_df["feature"] if f in cat_cols][:15]
    top_cat
    return (top_cat,)


@app.cell
def _(chi2_contingency, pd, test_df, top_cat, train_df):
    chi_results = []
    for fc in top_cat:
        train_counts = train_df[fc].value_counts()
        test_counts = test_df[fc].value_counts()
        all_categories = sorted(set(train_counts.index) | set(test_counts.index))
        contingency = pd.DataFrame({
            "train": [train_counts.get(c, 0) for c in all_categories],
            "test": [test_counts.get(c, 0) for c in all_categories],
        })
        contingency = contingency[contingency.sum(axis=1) > 0]
        chi2, pval_c, _, _ = chi2_contingency(contingency.T)
        chi_results.append({
            "feature": fc,
            "chi2": chi2,
            "p_value": pval_c,
            "n_categories": len(contingency),
        })

    chi_df = pd.DataFrame(chi_results).sort_values("chi2", ascending=False)
    chi_df["drift"] = chi_df["p_value"] < 0.01
    chi_df
    return all_categories, chi2, chi_df, chi_results, contingency, fc, pval_c, test_counts, train_counts


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 5. PSI (Population Stability Index)

        Métrica estándar en banca y fraude. Mide qué tan distinta es una distribución
        respecto de un baseline.

        - **PSI < 0.1** — sin cambios significativos
        - **0.1 ≤ PSI < 0.2** — cambio moderado, monitorear
        - **PSI ≥ 0.2** — cambio significativo, considerar reentrenar

        Calculamos PSI sobre las top features usando train como baseline y test como periodo actual.
        """
    )
    return


@app.cell
def _(np, pd, test_df, top_num, train_df):
    def psi(expected, actual, bins=10):
        expected = expected[~np.isnan(expected)]
        actual = actual[~np.isnan(actual)]
        if len(expected) == 0 or len(actual) == 0:
            return np.nan
        cuts = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        if len(cuts) < 3:
            return np.nan
        e_counts, _ = np.histogram(expected, bins=cuts)
        a_counts, _ = np.histogram(actual, bins=cuts)
        e_pct = np.where(e_counts == 0, 1e-6, e_counts) / e_counts.sum()
        a_pct = np.where(a_counts == 0, 1e-6, a_counts) / a_counts.sum()
        return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))

    psi_results = []
    for f_psi in top_num:
        v = psi(train_df[f_psi].values, test_df[f_psi].values)
        psi_results.append({"feature": f_psi, "psi": v})

    psi_df = pd.DataFrame(psi_results).sort_values("psi", ascending=False)
    psi_df["bucket"] = pd.cut(
        psi_df["psi"],
        bins=[-1, 0.1, 0.2, 10],
        labels=["estable (<0.1)", "moderado (0.1-0.2)", "significativo (>0.2)"],
    )
    psi_df
    return f_psi, psi, psi_df, psi_results, v


@app.cell
def _(plt, psi_df):
    fig_psi, ax_psi = plt.subplots(figsize=(10, 8))
    top_psi = psi_df.head(20)
    bar_colors = ["#E63946" if x > 0.2 else "#F4A261" if x > 0.1 else "#2E86AB" for x in top_psi["psi"]]
    ax_psi.barh(top_psi["feature"][::-1], top_psi["psi"][::-1], color=bar_colors[::-1])
    ax_psi.axvline(0.1, color="orange", linestyle="--", label="moderado")
    ax_psi.axvline(0.2, color="red", linestyle="--", label="significativo")
    ax_psi.set_title("PSI por feature (train vs test)")
    ax_psi.set_xlabel("PSI")
    ax_psi.legend()
    plt.tight_layout()
    fig_psi
    return ax_psi, bar_colors, fig_psi, top_psi


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 6. Drift en las predicciones

        Distribución de probabilidades que el modelo asigna en train vs test.
        Si la distribución cambia, el modelo está siendo "empujado" a una región
        diferente del espacio de probabilidades — síntoma de drift en los inputs.
        """
    )
    return


@app.cell
def _(feature_cols, np, plt, test_df, train_df, xgb_model):
    rng = np.random.default_rng(42)
    train_sample = train_df.sample(n=min(80000, len(train_df)), random_state=42)

    proba_train = xgb_model.predict_proba(train_sample[feature_cols])[:, 1]
    proba_test = xgb_model.predict_proba(test_df[feature_cols])[:, 1]

    fig_pd, ax_pd = plt.subplots(figsize=(10, 4))
    ax_pd.hist(proba_train, bins=80, alpha=0.5, label="train", color="#2E86AB", density=True)
    ax_pd.hist(proba_test, bins=80, alpha=0.5, label="test", color="#E63946", density=True)
    ax_pd.set_yscale("log")
    ax_pd.set_title("Distribución de probabilidades del modelo")
    ax_pd.set_xlabel("P(isFraud=1)")
    ax_pd.legend()
    fig_pd
    return ax_pd, fig_pd, proba_test, proba_train, rng, train_sample


@app.cell
def _(ks_2samp, proba_test, proba_train):
    pred_drift_stat, pred_drift_pval = ks_2samp(proba_train, proba_test)
    {
        "ks_pred_drift": float(pred_drift_stat),
        "p_value": float(pred_drift_pval),
        "significativo (p<0.01)": bool(pred_drift_pval < 0.01),
    }
    return pred_drift_pval, pred_drift_stat


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 7. Simulación de adaptación: estática vs adaptativa

        Comparamos dos estrategias de despliegue sobre las semanas de test:

        - **Estática:** el modelo entrenado una sola vez (notebook 03) predice todas las semanas.
        - **Adaptativa:** después de cada semana, el modelo se reentrena incorporando la
          semana recién observada (ventana expansiva). Simula el escenario donde tenemos
          ground truth con un día de delay.

        Reentrenamientos en GPU (CUDA), `n_estimators=500` fijo (sin early stopping)
        para que cada semana sea comparable y la simulación termine en pocos minutos.
        """
    )
    return


@app.cell
def _(
    average_precision_score,
    feature_cols,
    pd,
    roc_auc_score,
    test_df,
    time,
    train_df,
    xgb,
    xgb_model,
):
    test_with_week = test_df.copy()
    test_with_week["week"] = test_with_week["timestamp"].dt.to_period("W").dt.start_time
    weeks = sorted(test_with_week["week"].unique())

    base_train_X = train_df[feature_cols]
    base_train_y = train_df["isFraud"].values

    static_results = []
    for w in weeks:
        week_df = test_with_week[test_with_week["week"] == w]
        if week_df["isFraud"].nunique() < 2:
            continue
        proba = xgb_model.predict_proba(week_df[feature_cols])[:, 1]
        static_results.append({
            "week": w,
            "n": len(week_df),
            "fraud_rate": float(week_df["isFraud"].mean()),
            "roc_auc": float(roc_auc_score(week_df["isFraud"], proba)),
            "pr_auc": float(average_precision_score(week_df["isFraud"], proba)),
        })

    adaptive_results = []
    cum_X = base_train_X.copy()
    cum_y = base_train_y.copy()
    current_model = xgb_model

    common_params = dict(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.7,
        min_child_weight=5,
        reg_lambda=1.0,
        tree_method="hist",
        device="cuda",
        random_state=42,
    )
    pos = (cum_y == 1).sum()
    neg = (cum_y == 0).sum()
    common_params["scale_pos_weight"] = neg / pos

    for i, w in enumerate(weeks):
        week_df = test_with_week[test_with_week["week"] == w]
        if week_df["isFraud"].nunique() < 2:
            continue

        proba_a = current_model.predict_proba(week_df[feature_cols])[:, 1]
        adaptive_results.append({
            "week": w,
            "n": len(week_df),
            "roc_auc": float(roc_auc_score(week_df["isFraud"], proba_a)),
            "pr_auc": float(average_precision_score(week_df["isFraud"], proba_a)),
        })

        if i < len(weeks) - 1:
            cum_X = pd.concat([cum_X, week_df[feature_cols]], ignore_index=True)
            cum_y = list(cum_y) + list(week_df["isFraud"].values)
            cum_y_arr = pd.Series(cum_y).values
            new_pos = (cum_y_arr == 1).sum()
            new_neg = (cum_y_arr == 0).sum()
            common_params["scale_pos_weight"] = new_neg / new_pos

            t = time.time()
            new_model = xgb.XGBClassifier(**common_params)
            new_model.fit(cum_X, cum_y_arr, verbose=False)
            print(f"retrain tras semana {w.date()} → {time.time()-t:.1f}s, train_size={len(cum_X)}")
            current_model = new_model
            cum_y = cum_y_arr

    static_df = pd.DataFrame(static_results)
    adaptive_df = pd.DataFrame(adaptive_results)
    static_df, adaptive_df
    return (
        adaptive_df,
        adaptive_results,
        base_train_X,
        base_train_y,
        common_params,
        cum_X,
        cum_y,
        cum_y_arr,
        current_model,
        i,
        neg,
        new_model,
        new_neg,
        new_pos,
        pos,
        proba,
        proba_a,
        static_df,
        static_results,
        t,
        test_with_week,
        w,
        week_df,
        weeks,
    )


@app.cell
def _(adaptive_df, plt, static_df):
    fig_cmp, axes_cmp = plt.subplots(1, 2, figsize=(14, 4))

    axes_cmp[0].plot(static_df["week"], static_df["roc_auc"], marker="o", label="estática", color="#2E86AB")
    axes_cmp[0].plot(adaptive_df["week"], adaptive_df["roc_auc"], marker="s", label="adaptativa", color="#E63946")
    axes_cmp[0].set_title("ROC-AUC semanal")
    axes_cmp[0].set_ylabel("ROC-AUC")
    axes_cmp[0].tick_params(axis="x", rotation=30)
    axes_cmp[0].legend()

    axes_cmp[1].plot(static_df["week"], static_df["pr_auc"], marker="o", label="estática", color="#2E86AB")
    axes_cmp[1].plot(adaptive_df["week"], adaptive_df["pr_auc"], marker="s", label="adaptativa", color="#E63946")
    axes_cmp[1].set_title("PR-AUC semanal")
    axes_cmp[1].set_ylabel("PR-AUC")
    axes_cmp[1].tick_params(axis="x", rotation=30)
    axes_cmp[1].legend()

    plt.tight_layout()
    fig_cmp
    return axes_cmp, fig_cmp


@app.cell
def _(adaptive_df, static_df):
    comparison_summary = {
        "estatica_roc_auc_promedio": float(static_df["roc_auc"].mean()),
        "adaptativa_roc_auc_promedio": float(adaptive_df["roc_auc"].mean()),
        "delta_roc_auc": float(adaptive_df["roc_auc"].mean() - static_df["roc_auc"].mean()),
        "estatica_pr_auc_promedio": float(static_df["pr_auc"].mean()),
        "adaptativa_pr_auc_promedio": float(adaptive_df["pr_auc"].mean()),
        "delta_pr_auc": float(adaptive_df["pr_auc"].mean() - static_df["pr_auc"].mean()),
    }
    comparison_summary
    return (comparison_summary,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 8. Estrategia de trigger para reentrenamiento

        En producción, el sistema debería decidir **automáticamente** cuándo reentrenar.
        Definimos los siguientes umbrales:

        - **Trigger 1 (data drift):** PSI promedio sobre top features > 0.2
        - **Trigger 2 (performance drop):** caída de ROC-AUC semanal > 5% respecto al baseline
        - **Trigger 3 (target drift):** cambio absoluto en `fraud_rate` semanal > 1pp
        - **Cadencia mínima:** reentrenar al menos cada 4 semanas aunque no haya señales

        Aplicamos estas reglas retrospectivamente al periodo de test para ver cuándo
        se hubiera disparado un reentrenamiento.
        """
    )
    return


@app.cell
def _(pd, psi_df, static_df, train_df):
    baseline_fraud = float(train_df["isFraud"].mean())
    baseline_auc = float(static_df["roc_auc"].iloc[0])

    triggers = static_df.copy()
    triggers["fraud_rate_delta"] = (triggers["fraud_rate"] - baseline_fraud).abs()
    triggers["auc_drop_pct"] = (baseline_auc - triggers["roc_auc"]) / baseline_auc * 100
    triggers["trigger_perf"] = triggers["auc_drop_pct"] > 5.0
    triggers["trigger_target"] = triggers["fraud_rate_delta"] > 0.01
    triggers["psi_global"] = float(psi_df["psi"].mean())
    triggers["trigger_psi"] = triggers["psi_global"] > 0.2
    triggers["should_retrain"] = (
        triggers["trigger_perf"] | triggers["trigger_target"] | triggers["trigger_psi"]
    )
    triggers[["week", "roc_auc", "auc_drop_pct", "fraud_rate", "fraud_rate_delta",
              "psi_global", "trigger_perf", "trigger_target", "trigger_psi", "should_retrain"]]
    return baseline_auc, baseline_fraud, triggers


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 9. Persistencia del análisis de drift
        """
    )
    return


@app.cell
def _(
    ARTIFACTS_DIR,
    adaptive_df,
    chi_df,
    comparison_summary,
    json,
    ks_df,
    pred_drift_pval,
    pred_drift_stat,
    psi_df,
    static_df,
    triggers,
):
    drift_summary = {
        "ks_test": {
            "n_features_evaluadas": len(ks_df),
            "n_con_drift": int(ks_df["drift"].sum()),
            "ks_max": float(ks_df["ks_stat"].max()),
            "feature_top": ks_df.iloc[0]["feature"],
        },
        "chi_squared": {
            "n_features_evaluadas": len(chi_df),
            "n_con_drift": int(chi_df["drift"].sum()),
        },
        "psi": {
            "psi_promedio": float(psi_df["psi"].mean()),
            "psi_max": float(psi_df["psi"].max()),
            "n_features_significativas": int((psi_df["psi"] > 0.2).sum()),
        },
        "prediction_drift": {
            "ks_stat": float(pred_drift_stat),
            "p_value": float(pred_drift_pval),
        },
        "adaptive_vs_static": comparison_summary,
        "trigger_weeks_recommend_retrain": [
            str(w.date()) for w in triggers[triggers["should_retrain"]]["week"]
        ],
    }

    static_df.to_csv(ARTIFACTS_DIR / "static_weekly.csv", index=False)
    adaptive_df.to_csv(ARTIFACTS_DIR / "adaptive_weekly.csv", index=False)
    psi_df.to_csv(ARTIFACTS_DIR / "psi_features.csv", index=False)
    ks_df.to_csv(ARTIFACTS_DIR / "ks_features.csv", index=False)
    triggers.to_csv(ARTIFACTS_DIR / "trigger_simulation.csv", index=False)

    with open(ARTIFACTS_DIR / "drift_summary.json", "w", encoding="utf-8") as fdrift:
        json.dump(drift_summary, fdrift, indent=2, ensure_ascii=False, default=float)

    drift_summary
    return drift_summary, fdrift


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 10. Conclusiones

        ### Resumen cuantitativo del drift detectado

        | Test | Resultado | Interpretación |
        |---|---|---|
        | KS (numéricas, top 30) | 21/30 con drift (p<0,01), KS_max=0,18 (V326) | Drift estadísticamente detectable pero de **magnitud baja-moderada** |
        | Chi-cuadrado (categóricas, top 15) | 15/15 con drift | Esperable: chi² es muy sensible con N grande |
        | **PSI** | promedio **0,009**, máximo **0,030**, **0 features > 0,2** | Por la métrica estándar de la industria, **no hay drift estructural significativo** |
        | Drift en predicciones | KS=0,065 (p≈0) | Cambio detectable pero pequeño en la distribución de probas |

        ### Por qué KS y PSI dan lecturas distintas

        Con 440k vs 92k muestras, el test KS detecta diferencias estadísticamente
        significativas aunque sean diminutas en magnitud (alta potencia estadística).
        El **PSI mide magnitud absoluta** y por eso resulta más informativo en este caso:
        las distribuciones cambian un poco pero no lo suficiente como para activar
        una alerta operativa por sí solas.

        ### Valor de la adaptación (resultado central del proyecto)

        | Estrategia | ROC-AUC promedio | PR-AUC promedio |
        |---|---|---|
        | Estática (modelo fijo) | 0,8967 | 0,5475 |
        | **Adaptativa (reentrenamiento semanal)** | **0,9256** | **0,6112** |
        | **Δ adaptativa - estática** | **+0,0289 (+3,2%)** | **+0,0637 (+11,6%)** |

        El reentrenamiento incremental gana **+11,6% en PR-AUC promedio**, que es la
        métrica que más importa para fraude (clase minoritaria). Aunque el drift por PSI
        no parece "fuerte", el modelo sí se beneficia significativamente de actualizarse,
        lo que confirma que existen cambios sutiles en las relaciones feature→target
        (concept drift) que las métricas de distribución por sí solas no capturan.

        ### Triggers de reentrenamiento

        Aplicando las reglas (PSI>0,2, caída de AUC>5% o Δfraud_rate>1pp) sobre las 5
        semanas de test, **solo se dispara una alerta**: la semana del **2018-05-14**,
        coincidiendo con el peor desempeño del modelo estático (AUC 0,868).

        ### Implicancias para el despliegue

        - El modelo estático es **suficientemente robusto** como para servir en producción
          sin requerir reentrenamientos diarios.
        - La estrategia recomendada es **híbrida**: reentrenamiento programado cada
          2-4 semanas + trigger por caída de performance como red de seguridad.
        - El monitoreo en producción debe incluir **PR-AUC** (no solo ROC-AUC) y la
          tasa de fraude observada por ventana, no solo PSI.

        **Siguiente paso:** servicio FastAPI en `src/api/` que carga `xgb_model.joblib` +
        `preprocessor.joblib` y expone `/predict`, `/health` y `/version`. Un job
        separado ejecuta este notebook periódicamente y dispara reentrenamiento cuando
        algún trigger se activa.
        """
    )
    return


if __name__ == "__main__":
    app.run()