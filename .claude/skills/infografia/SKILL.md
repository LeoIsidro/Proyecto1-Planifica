---
name: infografia
description: Genera la infografía final del Proyecto 1 que sintetiza el ciclo de vida completo del sistema de IA. Pregunta al usuario si la quiere en HTML autocontenido (más editable, más rápido) o React (más componible, mejor para exportar a PDF/PNG con html-to-image). Diseño minimalista y elegante. Usa números reales del proyecto.
---

# Skill: Infografía final del Proyecto 1

## Propósito

Generar una infografía visual de **una sola página** que sintetice el ciclo de vida
completo del sistema de detección de fraude adaptativo, destacando las etapas clave
y su interrelación.

Es uno de los entregables explícitos de la rúbrica (4. Producto esperado → "Infografía
final: representación visual que sintetice el ciclo de vida completo del sistema de IA").

## Pregunta inicial al usuario

Antes de generar nada, preguntar al usuario:

> ¿En qué formato querés la infografía?
> 1. **HTML autocontenido** — un archivo `.html` con CSS embebido. Se abre en cualquier
>    browser y se exporta a PDF con Ctrl+P. Más simple, sin dependencias.
> 2. **React (Vite)** — proyecto npm con componentes. Más editable a futuro, mejor para
>    exportar a PNG/SVG con `html-to-image`. Tarda más en setup.

Si responde 1: seguir el flujo HTML.
Si responde 2: seguir el flujo React.
Si no responde claramente: usar HTML por defecto.

## Inputs

- `README.md` — para extraer los resultados clave
- `artifacts/metadata.json` — splits, número de features
- `artifacts/metrics.json` — métricas de modelos
- `artifacts/drift_summary.json` — comparación adaptativa, triggers

## Output

### Si HTML

```
docs/infografia/
└── index.html        # archivo único, autocontenido
```

### Si React

```
docs/infografia/
├── package.json
├── vite.config.ts
├── tsconfig.json
├── index.html
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   ├── styles.css
│   └── components/
│       ├── Hero.tsx
│       ├── PipelineStage.tsx
│       ├── MetricCard.tsx
│       └── AdaptiveLoop.tsx
└── README.md          # cómo correr y exportar a PNG/PDF
```

## Estructura visual de la infografía

Distribución vertical, una página A4 en formato apaisado o vertical (preguntar si no
está claro; default: vertical A3 para tener más aire).

### Bloques (de arriba a abajo)

1. **Header**
   - Título: "Sistema Inteligente Adaptativo de Detección de Fraude"
   - Subtítulo: "IEEE-CIS · 590.540 transacciones · 6 meses · 3,5 % fraude"
   - Logo o marca minimalista (texto)

2. **Pipeline horizontal de 5 etapas** (iconografía simple, números arriba)
   - 01 Datos · 02 Preparación · 03 Modelado · 04 Despliegue · 05 Adaptación
   - Cada etapa con un ícono SVG inline (sin librerías) y 1-2 líneas de descripción

3. **Bloque de métricas** (3-4 cards en grid)
   - ROC-AUC test: 0,896
   - PR-AUC test: 0,547
   - Δ adaptativa vs estática: +11,6 % PR-AUC
   - Latencia promedio: ~12 ms

4. **Diagrama del loop adaptativo** (centro de la infografía)
   - Visual circular o cíclico mostrando: producción → monitoreo → trigger → reentrenamiento → producción
   - Marcadores con los 3 triggers (PSI, ΔAUC, Δfraud_rate)

5. **Stack tecnológico** (línea de logos/texto)
   - Python · marimo · XGBoost (CUDA) · FastAPI · Docker

6. **Footer**
   - Curso, equipo, año
   - Link al repo (placeholder)

## Reglas de diseño (CRÍTICAS)

- **Paleta limitada**: blanco/crema de fondo (`#FAFAFA`), texto negro suave (`#1A1A1A`),
  un acento azul (`#2E5AAC`), un acento de alerta opcional (`#E07A5F`). NUNCA usar
  más de 4 colores totales.
- **Tipografía**:
  - Sans-serif moderna: usar Inter, IBM Plex Sans o system-ui.
  - Pesos: 400 para body, 600 para títulos, 700 reservado para números/datos.
  - Tamaños generosos: hero 48px+, sección 24-32px, body 14-16px.
- **Espaciado**: respiración generosa. Padding mínimo de 64px en bordes.
- **Bordes**: sutiles (`1px solid #EEEEEE`) o sin borde, separar bloques con whitespace.
- **Sombras**: NINGUNA (o sombras muy sutiles `0 1px 2px rgba(0,0,0,0.05)`).
- **Esquinas**: ligeramente redondeadas (`border-radius: 8px`) o cuadradas.
- **Sin emojis**. Sin iconos de FontAwesome. Solo SVG inline minimalista.
- **Sin gradientes** salvo el del bloque del loop adaptativo si es estrictamente necesario.
- **Mobile responsive**: en HTML, que se vea bien también en pantallas chicas.

## Plantilla de referencia HTML (esqueleto)

```html
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>Sistema de Detección de Fraude Adaptativo · Proyecto 1</title>
  <style>
    :root {
      --bg: #FAFAFA;
      --surface: #FFFFFF;
      --text: #1A1A1A;
      --muted: #6C757D;
      --accent: #2E5AAC;
      --alert: #E07A5F;
      --border: #EEEEEE;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
      font-size: 16px;
      line-height: 1.5;
    }
    .page { max-width: 1100px; margin: 0 auto; padding: 64px 48px; }
    /* ... */
  </style>
</head>
<body>
  <main class="page">
    <!-- bloques aquí -->
  </main>
</body>
</html>
```

## Plantilla de referencia React (Vite + TS)

`package.json`:

```json
{
  "name": "infografia-fraude",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "export-png": "node scripts/export.js"
  },
  "dependencies": {
    "react": "^18.3.0",
    "react-dom": "^18.3.0"
  },
  "devDependencies": {
    "@types/react": "^18.3.0",
    "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.0",
    "typescript": "^5.5.0",
    "vite": "^5.4.0",
    "html-to-image": "^1.11.0"
  }
}
```

Componentes pequeños, props tipados, CSS module o styled-components mínimo.
Incluir un script `scripts/export.js` que use `puppeteer` o `html-to-image` para
exportar a PNG en alta resolución (recomendar 2× para print).

## Reglas duras

1. **Todos los números** que aparezcan en la infografía deben venir de los artifacts/.
   Si falta un artifact, abortar y pedir al usuario que ejecute el notebook
   correspondiente.
2. **Una sola página visual**. La infografía no se desborda en scroll infinito.
3. **No usar imágenes externas** ni íconos de CDN. Solo SVG inline.
4. **No usar Tailwind ni librerías de UI**. CSS puro o CSS-in-JS mínimo.
5. **No agregar marca personal ni firmas de Claude**.

## Flujo del agente

1. Preguntar HTML vs React.
2. Verificar que existen los artifacts. Si falta alguno, abortar.
3. Crear `docs/infografia/` con la estructura correspondiente.
4. Generar los archivos siguiendo la estructura visual y las reglas de diseño.
5. Imprimir al usuario:
   - Path del archivo principal
   - Cómo abrirlo (browser para HTML, `npm install && npm run dev` para React)
   - Cómo exportar a PDF (Ctrl+P → guardar como PDF) o PNG
