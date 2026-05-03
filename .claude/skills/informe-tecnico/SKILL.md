---
name: informe-tecnico
description: Genera el informe técnico en LaTeX (máximo 5 páginas) del Proyecto 1 del curso "Planificación y Toma de Decisiones en IA". Lee los artifacts/ del repo para usar números reales del entrenamiento y del análisis de drift, y sigue la rúbrica del curso. Output compilable con pdflatex en docs/informe/main.tex.
---

# Skill: Informe técnico del Proyecto 1

## Propósito

Generar el informe técnico de máximo 5 páginas del proyecto de detección de fraude
con concept drift sobre IEEE-CIS, en LaTeX, listo para compilar con pdflatex.

El informe debe usar los **datos reales** de los artifacts del repo, nunca inventar
métricas. Si un artifact necesario no existe, indicar al usuario qué notebook debe
ejecutar primero.

## Inputs obligatorios (leer en este orden antes de escribir)

1. `README.md` — visión general del proyecto y resultados clave
2. `artifacts/metadata.json` — splits temporales, número de features, columnas droppeadas
3. `artifacts/metrics.json` — métricas val/test de los 3 modelos
4. `artifacts/drift_summary.json` — KS, chi-cuadrado, PSI, comparación adaptativa, triggers
5. `artifacts/weekly_metrics.csv` — degradación semanal del modelo estático

Inputs opcionales si están disponibles:
- `artifacts/static_weekly.csv`, `artifacts/adaptive_weekly.csv`
- `artifacts/psi_features.csv`, `artifacts/ks_features.csv`
- `artifacts/trigger_simulation.csv`
- `notebooks/04_drift_analysis.py` — para citar análisis específicos

## Output

Crear los siguientes archivos:

```
docs/informe/
├── main.tex          # documento principal compilable
├── refs.bib          # bibliografía (vacía o con refs reales)
└── figs/             # carpeta para figuras (puede quedar vacía)
```

Y al final, imprimir un resumen indicando:
- Path del archivo generado
- Comando exacto para compilar (`pdflatex docs/informe/main.tex`, correr 2× para refs)
- Si faltó algún artifact, listarlo

## Estructura del informe (sigue la rúbrica del curso)

El informe tiene **5 secciones obligatorias**, cada una de ~1 página. La rúbrica es:

| Sección | Contenido | Peso |
|---|---|---|
| 1 | Definición del problema, objetivos, restricciones y métricas | 4 pts |
| 2 | Análisis y preparación de datos | 4 pts |
| 3 | Modelado y evaluación temporal | 4 pts |
| 4 | Diseño del sistema y estrategia de adaptación | 4 pts |
| 5 | Calidad del producto entregado | 4 pts |

### Sección 1: Definición del problema, objetivos, restricciones y métricas

Debe cubrir:
- **Problema**: detección de fraude en transacciones financieras con datos no estacionarios
- **Objetivos**: clasificar correctamente, manejar drift, decidir cuándo reentrenar
- **Restricciones**: dataset histórico limitado a 6 meses, sin labels de los `test_*.csv`,
  desbalance fuerte (3,5%), latencia objetivo (~10-20 ms por inferencia)
- **Métricas técnicas**: ROC-AUC, PR-AUC, F1 (justificar por qué NO accuracy)
- **Métricas de decisión**: tiempo de inferencia, costo de falso positivo vs falso negativo
- **Métricas sociales**: equidad (no podemos analizar fairness sin atributos sensibles
  en este dataset, pero mencionarlo como limitación)

### Sección 2: Análisis y preparación de datos

Debe cubrir:
- Tamaño del dataset (590.540 transacciones × 394 columnas tras merge)
- Cobertura temporal: del 2017-12-02 al 2018-06-01 (181 días)
- Tasa de fraude global: 3,50%
- Variación temporal: media 3,60%, std 1,00%, rango [1,10%, 6,99%]
- Decisiones de preprocesamiento:
  - Drop de 12 columnas con >90% nulos (calculado solo sobre train para evitar leakage)
  - OrdinalEncoder ajustado solo en train, unknown_value=-1
  - Feature engineering temporal (hour, dayofweek, days_since_start, etc.)
- Split temporal estricto: train (440.018, 74,5%) / val (58.095, 9,8%) / test (92.427, 15,7%)
- Justificar por qué split temporal y no aleatorio

### Sección 3: Modelado y evaluación temporal

Tabla obligatoria con resultados reales (sacar de `metrics.json`):

| Modelo | ROC-AUC val | ROC-AUC test | PR-AUC val | PR-AUC test |
|---|---|---|---|---|
| Logistic Regression | 0,8425 | 0,8223 | 0,3490 | 0,1710 |
| Random Forest | 0,8819 | 0,8810 | 0,4299 | 0,4864 |
| **XGBoost (GPU)** | **0,9129** | **0,8962** | **0,5817** | **0,5469** |

Comentar:
- Por qué XGBoost gana (interacciones no lineales)
- Por qué LogReg no es viable (precisión 0,107)
- Caída entre val y test (drift inicial)
- Evaluación semanal: AUC oscila entre 0,868 (peor, semana 14-may) y 0,924 (mejor, 30-abr)
- Patrón no monotónico

### Sección 4: Diseño del sistema y estrategia de adaptación

Debe cubrir:
- Arquitectura: ingesta → preprocessor → XGBoost → decisión (allow/review/block) → API REST
- Despliegue: FastAPI containerizado en Docker, modelo serializado en joblib, volumen
  read-only para hot-swap del modelo
- Detección de drift: KS, chi-cuadrado, PSI sobre features + KS sobre predicciones
- **Hallazgo clave**: PSI máximo 0,030 (bajo), pero modelo adaptativo gana **+11,6% PR-AUC**
  → existe concept drift no capturado por métricas de distribution drift
- Estrategia adoptada: reentrenamiento programado cada 2-4 semanas + trigger por
  caída de performance (>5% AUC) o cambio en fraud_rate (>1pp)
- Tabla comparativa estática vs adaptativa (de `drift_summary.json`)

### Sección 5: Calidad del producto entregado

Debe cubrir:
- Estructura del repo (notebooks marimo, src/api, src/retrain, Dockerfile)
- Reproducibilidad: requirements.txt + requirements-api.txt, artifacts versionables,
  todo en git
- Cómo correr de cero (referenciar README)
- Diagrama de arquitectura (referenciar a la figura generada por `/diagrama-arquitectura`)
- Limitaciones honestas (periodo corto, fecha base es convención, single worker)

## Plantilla LaTeX a usar

Usar exactamente esta plantilla como base de `main.tex`. Reemplazar los placeholders
`{{...}}` con datos reales leídos de los artifacts:

```latex
\documentclass[10pt,a4paper]{article}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[spanish]{babel}
\usepackage[margin=2cm,top=2cm,bottom=2cm]{geometry}
\usepackage{libertine}
\usepackage[libertine]{newtxmath}
\usepackage{microtype}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage[colorlinks=true,linkcolor=accent,urlcolor=accent,citecolor=accent]{hyperref}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{titlesec}
\usepackage{enumitem}
\usepackage{caption}

\definecolor{accent}{HTML}{2E5AAC}
\definecolor{muted}{HTML}{6C757D}

\titleformat{\section}{\large\bfseries\color{accent}}{\thesection.}{0.5em}{}
\titleformat{\subsection}{\normalsize\bfseries}{\thesubsection}{0.5em}{}
\titlespacing*{\section}{0pt}{1.2ex}{0.6ex}
\titlespacing*{\subsection}{0pt}{0.8ex}{0.4ex}

\setlist{nosep,leftmargin=1.2em}
\captionsetup{font=small,labelfont=bf,labelsep=period}
\renewcommand{\arraystretch}{1.15}

\title{\vspace{-1.5em}\textbf{Sistema Inteligente Adaptativo para Detección de Fraude}\\
\large\color{muted}Proyecto 1 — Planificación y Toma de Decisiones en IA}
\author{{{NOMBRES_DEL_EQUIPO}}}
\date{\today}

\begin{document}
\maketitle
\vspace{-2em}

\section{Definición del problema, objetivos y métricas}
{{...}}

\section{Análisis y preparación de los datos}
{{...}}

\section{Modelado y evaluación temporal}
{{...}}

\section{Diseño del sistema y estrategia de adaptación}
{{...}}

\section{Calidad del producto y reproducibilidad}
{{...}}

\end{document}
```

## Reglas de estilo

- **Idioma**: español, formal pero claro. Evitar muletillas y palabras vacías.
- **Diseño**: minimalista, paleta neutra con UN color de acento (azul `#2E5AAC`).
- **Tablas**: usar `booktabs` (`\toprule`, `\midrule`, `\bottomrule`). Sin líneas verticales.
- **Listas**: máximo 4 ítems por lista para no romper el límite de páginas.
- **Código**: NO incluir bloques largos de código. Si se cita, usar `\texttt{...}` inline.
- **Figuras**: referenciar como `\ref{fig:nombre}`, ubicar en `figs/` y comentar
  brevemente. Si las figuras no existen aún, dejar el `\includegraphics` comentado
  con un TODO claro.
- **Decimales**: usar coma decimal (estilo español): `0,8962` no `0.8962`.
- **Porcentajes**: con espacio fino antes del símbolo: `11,6\,\%`.
- **Citas a archivos del repo**: usar `\texttt{notebooks/04\_drift\_analysis.py}`.

## Reglas duras

1. **Cada número que aparezca en el informe debe venir de los artifacts/**, no de
   estimaciones. Si un artifact no existe, decir al usuario qué notebook ejecutar.
2. **Máximo 5 páginas**. Si al primer borrador supera ese límite, recortar prosa
   antes de recortar tablas o conclusiones.
3. **No usar emojis** en el LaTeX bajo ninguna circunstancia.
4. **No inventar referencias bibliográficas**. Si se cita algo (PSI estándar de banca,
   KS test) usar refs académicas reales o no citar.
5. **No agregar disclaimers tipo "este informe fue generado con IA"**.

## Flujo del agente

1. Verificar que existen los artifacts obligatorios. Si falta alguno, abortar y
   pedir al usuario que ejecute el notebook correspondiente.
2. Leer los JSONs y extraer todos los números necesarios.
3. Crear `docs/informe/` con `main.tex` y `refs.bib` vacío.
4. Generar el `.tex` siguiendo la plantilla y la estructura de las 5 secciones.
5. Verificar que el documento NO supera ~280 líneas (proxy de 5 páginas con esta
   plantilla; ajustar si hace falta).
6. Imprimir resumen final al usuario con el path y el comando de compilación.
