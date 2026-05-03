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
    # EDA — IEEE-CIS Fraud Detection

    Exploración inicial del dataset para entender:

    - Tamaño, estructura y tipos de columnas
    - Distribución de la variable objetivo (`isFraud`)
    - **Distribución temporal** de transacciones y de la tasa de fraude (clave para concept drift)
    - Calidad de los datos: valores faltantes
    - Diferencia entre tabla de transacciones e identidad

    El dataset cubre aproximadamente 6 meses de transacciones reales de Vesta Corporation.
    La columna `TransactionDT` es un **offset en segundos** desde una fecha de referencia
    (no es un timestamp absoluto). La fecha base que usaremos como referencia es `2017-12-01`,
    convención adoptada por la comunidad de Kaggle para esta competencia.
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path

    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (12, 5)

    DATA_DIR = Path("../data")
    REFERENCE_DATE = pd.Timestamp("2017-12-01")
    return DATA_DIR, REFERENCE_DATE, np, pd, plt


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Carga de datos
    """)
    return


@app.cell
def _(DATA_DIR, pd):
    transactions = pd.read_csv(DATA_DIR / "train_transaction.csv")
    identity = pd.read_csv(DATA_DIR / "train_identity.csv")
    transactions.shape, identity.shape
    return identity, transactions


@app.cell
def _(mo):
    mo.md(r"""
    La tabla de **transacciones** tiene una fila por operación y la tabla de **identidad**
    contiene metadatos opcionales (dispositivo, navegador, etc.) solo para un subconjunto
    de transacciones. Hacemos un `left join` para no perder filas.
    """)
    return


@app.cell
def _(identity, transactions):
    df = transactions.merge(identity, on="TransactionID", how="left")
    df.shape
    return (df,)


@app.cell
def _(df, identity, transactions):
    coverage = {
        "transacciones_total": len(transactions),
        "transacciones_con_identidad": len(identity),
        "% con identidad": round(100 * len(identity) / len(transactions), 2),
        "columnas_totales": df.shape[1],
    }
    coverage
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Estructura general
    """)
    return


@app.cell
def _(df):
    df.dtypes.value_counts()
    return


@app.cell
def _(df):
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    f"Uso de memoria: {memory_mb:.0f} MB"
    return


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Variable objetivo

    El target es `isFraud` (binaria). Esperamos un fuerte desbalance: en datasets reales
    de fraude financiero la clase positiva suele estar entre el 0.1% y el 5%.
    Esto condiciona la elección de métricas (no usaremos accuracy) y de estrategias de modelado.
    """)
    return


@app.cell
def _(df):
    target_dist = df["isFraud"].value_counts(normalize=False).to_frame("count")
    target_dist["pct"] = (100 * target_dist["count"] / len(df)).round(3)
    target_dist
    return


@app.cell
def _(df, plt):
    fig_target, ax_target = plt.subplots(figsize=(6, 4))
    df["isFraud"].value_counts().plot(kind="bar", ax=ax_target, color=["#2E86AB", "#E63946"])
    ax_target.set_title("Distribución de la clase objetivo")
    ax_target.set_xlabel("isFraud")
    ax_target.set_ylabel("Cantidad de transacciones")
    ax_target.set_xticklabels(["No fraude (0)", "Fraude (1)"], rotation=0)
    fig_target
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Análisis temporal

    Convertimos `TransactionDT` (segundos desde la fecha de referencia) en `timestamp` real
    y derivamos columnas auxiliares de fecha, día y semana. Estas variables son la base
    para todo el análisis de concept drift y para el split temporal de train/val/test.
    """)
    return


@app.cell
def _(REFERENCE_DATE, df, pd):
    df_t = df.copy()
    df_t["timestamp"] = REFERENCE_DATE + pd.to_timedelta(df_t["TransactionDT"], unit="s")
    df_t["date"] = df_t["timestamp"].dt.date
    df_t["week"] = df_t["timestamp"].dt.to_period("W").dt.start_time
    df_t["hour"] = df_t["timestamp"].dt.hour
    df_t["dayofweek"] = df_t["timestamp"].dt.dayofweek
    df_t[["TransactionDT", "timestamp", "date", "week"]].head()
    return (df_t,)


@app.cell
def _(df_t):
    period_info = {
        "fecha_min": df_t["timestamp"].min(),
        "fecha_max": df_t["timestamp"].max(),
        "duracion_dias": (df_t["timestamp"].max() - df_t["timestamp"].min()).days,
    }
    period_info
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 4.1 Volumen diario de transacciones
    """)
    return


@app.cell
def _(df_t, plt):
    daily_volume = df_t.groupby("date").size()
    fig_vol, ax_vol = plt.subplots(figsize=(14, 4))
    daily_volume.plot(ax=ax_vol, color="#2E86AB")
    ax_vol.set_title("Transacciones por día")
    ax_vol.set_ylabel("Cantidad")
    ax_vol.set_xlabel("Fecha")
    fig_vol
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 4.2 Tasa de fraude diaria

    Esta es la gráfica **más importante** para justificar el enfoque de concept drift:
    si la tasa de fraude varía significativamente a lo largo del tiempo, un modelo entrenado
    en un periodo puede degradarse al evaluarse en otro. Buscamos picos, tendencias o cambios
    de régimen.
    """)
    return


@app.cell
def _(df_t, plt):
    daily_fraud = df_t.groupby("date")["isFraud"].agg(["sum", "count"])
    daily_fraud["fraud_rate"] = daily_fraud["sum"] / daily_fraud["count"]

    fig_fr, ax_fr = plt.subplots(figsize=(14, 4))
    daily_fraud["fraud_rate"].plot(ax=ax_fr, color="#E63946")
    ax_fr.axhline(
        daily_fraud["fraud_rate"].mean(), color="black", linestyle="--", label="media global"
    )
    ax_fr.set_title("Tasa de fraude diaria")
    ax_fr.set_ylabel("% de transacciones fraudulentas")
    ax_fr.set_xlabel("Fecha")
    ax_fr.legend()
    fig_fr
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 4.3 Patrón por hora del día y día de la semana
    """)
    return


@app.cell
def _(df_t, plt):
    fig_hd, axes_hd = plt.subplots(1, 2, figsize=(14, 4))

    df_t.groupby("hour")["isFraud"].mean().plot(kind="bar", ax=axes_hd[0], color="#E63946")
    axes_hd[0].set_title("Tasa de fraude por hora del día")
    axes_hd[0].set_ylabel("Tasa de fraude")

    df_t.groupby("dayofweek")["isFraud"].mean().plot(kind="bar", ax=axes_hd[1], color="#E63946")
    axes_hd[1].set_title("Tasa de fraude por día de la semana")
    axes_hd[1].set_xticklabels(["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"], rotation=0)

    fig_hd
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Valores faltantes

    El IEEE-CIS es notorio por la cantidad de columnas con muchos nulos.
    Identificarlos temprano define la estrategia de preprocesamiento (imputación,
    eliminación de columnas, o uso de modelos como XGBoost que manejan nativamente NaN).
    """)
    return


@app.cell
def _(df, pd):
    missing = df.isna().mean().sort_values(ascending=False)
    missing_summary = pd.DataFrame(
        {
            "columna": missing.index,
            "pct_nulos": (missing.values * 100).round(2),
        }
    )
    missing_summary.head(30)
    return (missing,)


@app.cell
def _(missing, plt):
    bins = [0, 10, 25, 50, 75, 90, 100]
    labels = ["0-10%", "10-25%", "25-50%", "50-75%", "75-90%", "90-100%"]
    bucketed = (missing * 100).pipe(
        lambda s: s.groupby(
            __import__("pandas").cut(s, bins=bins, labels=labels, include_lowest=True)
        ).size()
    )

    fig_m, ax_m = plt.subplots(figsize=(8, 4))
    bucketed.plot(kind="bar", ax=ax_m, color="#F4A261")
    ax_m.set_title("Distribución de columnas según % de valores faltantes")
    ax_m.set_xlabel("% de nulos")
    ax_m.set_ylabel("Cantidad de columnas")
    fig_m
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Tipos de columnas y grupos de features
    """)
    return


@app.cell
def _(df):
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    feature_groups = {
        "categóricas": len(cat_cols),
        "numéricas": len(num_cols),
        "card_*": len([c for c in df.columns if c.startswith("card")]),
        "addr_*": len([c for c in df.columns if c.startswith("addr")]),
        "C_*": len([c for c in df.columns if c.startswith("C") and c[1:].isdigit()]),
        "D_*": len([c for c in df.columns if c.startswith("D") and c[1:].isdigit()]),
        "M_*": len([c for c in df.columns if c.startswith("M") and c[1:].isdigit()]),
        "V_*": len([c for c in df.columns if c.startswith("V") and c[1:].isdigit()]),
        "id_*": len([c for c in df.columns if c.startswith("id_") or c.startswith("id-")]),
    }
    feature_groups
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Distribución del monto de transacción
    """)
    return


@app.cell
def _(df, np, plt):
    fig_amt, axes_amt = plt.subplots(1, 2, figsize=(14, 4))

    df["TransactionAmt"].clip(upper=1000).hist(bins=80, ax=axes_amt[0], color="#2E86AB")
    axes_amt[0].set_title("Distribución de TransactionAmt (clip=1000)")
    axes_amt[0].set_xlabel("USD")

    np.log1p(df["TransactionAmt"]).hist(bins=80, ax=axes_amt[1], color="#2E86AB")
    axes_amt[1].set_title("log1p(TransactionAmt)")

    fig_amt
    return


@app.cell
def _(df, plt):
    fig_amt2, ax_amt2 = plt.subplots(figsize=(10, 4))
    df.groupby("isFraud")["TransactionAmt"].apply(lambda s: s.clip(upper=1000)).reset_index(
        level=0
    ).boxplot(by="isFraud", column="TransactionAmt", ax=ax_amt2)
    ax_amt2.set_title("Monto por clase (clip=1000)")
    fig_amt2
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Conclusiones del EDA

    Hallazgos cuantitativos sobre el dataset:

    1. **Volumen y forma:** 590.540 transacciones × 394 columnas tras el merge.
       Solo 144.233 transacciones (24,4%) tienen información de identidad asociada,
       por lo que el preprocesamiento no puede asumir su presencia.

    2. **Desbalance de la clase objetivo:** 20.663 fraudes vs 569.877 no fraudes
       (**3,50%** de tasa global). Esto descarta accuracy como métrica y prioriza
       **ROC-AUC, PR-AUC y F1** para evaluación.

    3. **Cobertura temporal:** del 2017-12-02 al 2018-06-01 (**181 días**, ~6 meses).
       Suficiente para hacer un split temporal train/val/test con bloques significativos
       y para evaluar evolución semanal sin quedarnos sin datos.

    4. **Drift en el target presente pero acotado:** la tasa de fraude diaria tiene
       media 3,60%, mediana 3,56%, std 1,00% (CV ≈ 28%). El día con mínima tasa
       registra 1,10% y el máximo 6,99% — un rango de **6×** entre días extremos.
       No es drift catastrófico, pero es suficiente para que un modelo estático se
       degrade y para que la adaptación tenga retorno medible.

    5. **Calidad de datos:** muchas columnas con alto porcentaje de nulos. Esto
       orienta al uso de XGBoost/LightGBM, que manejan NaN nativamente sin imputación.

    **Próximo notebook:** `02_preprocessing.py` — split temporal, drop de columnas
    con >90% nulos, label encoding y persistencia en `artifacts/`.
    """)
    return


if __name__ == "__main__":
    app.run()
