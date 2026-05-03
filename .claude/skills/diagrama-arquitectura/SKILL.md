---
name: diagrama-arquitectura
description: Genera los diagramas del sistema del Proyecto 1 (flujo de datos, despliegue y loop adaptativo) en Mermaid (para edición y vista en GitHub) y en TikZ (para inclusión en el informe LaTeX). Diseño minimalista y elegante. Output en docs/diagramas/.
---

# Skill: Diagramas de arquitectura

## Propósito

Generar tres diagramas que documentan el sistema:

1. **Pipeline de datos y modelado** — desde los CSVs hasta el modelo entrenado.
2. **Arquitectura de despliegue** — Docker compose, API REST, volumen de artifacts.
3. **Loop adaptativo** — monitoreo de drift, trigger, reentrenamiento, hot-swap del modelo.

Cada diagrama se entrega en **dos formatos**:

- **Mermaid** (`.md`): para edición rápida y visualización en GitHub/GitLab
- **TikZ** (`.tex`): para inclusión directa en el informe LaTeX

## Inputs (leer si están disponibles)

- `README.md` — para entender el flujo actual
- `docker-compose.yml` — componentes del despliegue
- `src/api/main.py` — endpoints expuestos
- `src/retrain.py` — pipeline de reentrenamiento
- `notebooks/04_drift_analysis.py` — triggers definidos

## Output

```
docs/diagramas/
├── 01-pipeline-datos.md       # Mermaid
├── 01-pipeline-datos.tex      # TikZ
├── 02-despliegue.md
├── 02-despliegue.tex
├── 03-loop-adaptativo.md
└── 03-loop-adaptativo.tex
```

## Diagrama 1: Pipeline de datos y modelado

**Mermaid** (en `01-pipeline-datos.md`):

```mermaid
flowchart LR
    A[(IEEE-CIS<br/>CSVs)] --> B[Merge<br/>transaction + identity]
    B --> C[Feature engineering<br/>temporal]
    C --> D[Split temporal<br/>74% / 10% / 16%]
    D --> E[Drop columnas<br/>>90% nulos]
    E --> F[OrdinalEncoder<br/>fit en train]
    F --> G[Parquets<br/>artifacts/]
    G --> H{XGBoost<br/>GPU CUDA 12.8}
    H --> I[Modelo + Preprocessor<br/>joblib]

    classDef data fill:#EAF2FB,stroke:#2E5AAC,color:#1A1A1A;
    classDef step fill:#FFFFFF,stroke:#6C757D,color:#1A1A1A;
    classDef model fill:#2E5AAC,stroke:#2E5AAC,color:#FFFFFF;
    class A,G,I data;
    class B,C,D,E,F step;
    class H model;
```

**TikZ** (en `01-pipeline-datos.tex`): generar versión equivalente con `\usepackage{tikz}`
y `\usetikzlibrary{positioning,shapes.geometric,arrows.meta}`. Usar nodos rectangulares
con esquinas redondeadas (`rounded corners=2pt`), líneas finas, paleta de 3 colores
(neutro, azul acento, gris).

## Diagrama 2: Arquitectura de despliegue

**Mermaid**:

```mermaid
flowchart TB
    Client[Cliente / Sistema<br/>transaccional] -->|HTTP POST<br/>/predict| API
    subgraph Docker[Docker Compose]
        API[FastAPI<br/>uvicorn :8000]
        Volume[(artifacts/<br/>volumen :ro)]
        API <-->|carga al inicio| Volume
    end
    Volume <-.->|hot-swap<br/>al reentrenar| Local[artifacts/ local<br/>en host]
    API -->|allow / review / block| Client

    classDef ext fill:#FFFFFF,stroke:#1A1A1A,color:#1A1A1A;
    classDef svc fill:#2E5AAC,stroke:#2E5AAC,color:#FFFFFF;
    classDef vol fill:#EAF2FB,stroke:#2E5AAC,color:#1A1A1A;
    class Client,Local ext;
    class API svc;
    class Volume vol;
```

**TikZ**: equivalente en LaTeX, manteniendo la separación visual del subgraph
(rectángulo con borde punteado y label "Docker Compose").

## Diagrama 3: Loop adaptativo

**Mermaid**:

```mermaid
flowchart LR
    A[API en producción<br/>predicciones diarias] --> B[Logs de predicciones<br/>+ ground truth]
    B --> C{Monitor<br/>weekly}
    C -->|PSI > 0.20| D[Trigger drift]
    C -->|ΔAUC > 5%| D
    C -->|Δfraud_rate > 1pp| D
    C -->|Cadencia 4 sem.| D
    D --> E[src/retrain.py<br/>--device cuda]
    E --> F[Nuevos artifacts<br/>xgb_model.joblib]
    F --> G[docker compose<br/>restart fraud-api]
    G --> A

    classDef live fill:#2E5AAC,stroke:#2E5AAC,color:#FFFFFF;
    classDef monitor fill:#FFFFFF,stroke:#6C757D,color:#1A1A1A;
    classDef trigger fill:#E07A5F,stroke:#E07A5F,color:#FFFFFF;
    classDef action fill:#EAF2FB,stroke:#2E5AAC,color:#1A1A1A;
    class A live;
    class B,C monitor;
    class D trigger;
    class E,F,G action;
```

**TikZ**: equivalente, mantener el ciclo cerrado visualmente.

## Reglas de diseño

- **Paleta limitada** a 3-4 colores: neutro (`#FFFFFF`/`#EAF2FB`), azul acento (`#2E5AAC`),
  gris (`#6C757D`), y un naranja para alertas/triggers (`#E07A5F`).
- **Tipografía**: sin etiquetas en mayúsculas, máximo 3 palabras por nodo cuando se pueda.
- **Sin emojis** en ningún diagrama.
- **Líneas**: finas, sin sombras, sin gradientes.
- **Etiquetas de aristas**: solo cuando aporten información (umbrales, condiciones).
  Nunca solo "yes/no".
- **TikZ**: usar `node distance=1.5cm`, fuente `\small`, sin colores fluo.

## Cómo incluir en el informe LaTeX

Indicar al usuario que en `main.tex` puede usar:

```latex
\begin{figure}[h]
  \centering
  \input{../diagramas/02-despliegue.tex}
  \caption{Arquitectura de despliegue del servicio de inferencia.}
  \label{fig:despliegue}
\end{figure}
```

## Flujo del agente

1. Crear `docs/diagramas/` si no existe.
2. Generar los 6 archivos (3 Mermaid + 3 TikZ).
3. Verificar que cada Mermaid es renderizable (sintaxis válida).
4. Verificar que cada TikZ es compilable (al menos sintácticamente).
5. Imprimir al usuario:
   - Lista de archivos creados
   - Cómo ver los Mermaid (GitHub renderiza inline, o usar https://mermaid.live)
   - Cómo incluir los TikZ en el informe (ejemplo `\input{}`)
