# Sargazo Detección

Herramienta de procesamiento digital de imágenes (PDI) para detectar y estimar el área de sargazo en imágenes aéreas/satelitales, con cálculo de logística de recolección.

## ¿Qué hace?

- Segmenta sargazo usando segmentación multi-condición en espacio HSV + índices de color
- Descarta falsos positivos (agua, vegetación, arena, techos)
- Aplica morfología para limpiar la máscara resultante
- Calcula el área real en m² usando fotogrametría (GSD)
- Estima el número de camiones necesarios para recolectar la biomasa

## Requisitos

```bash
pip install opencv-python numpy matplotlib
```

## Uso

```python
from actividad_final import analizar_sargazo_fotogrametria

resultado = analizar_sargazo_fotogrametria(
    ruta_imagen="tu_imagen.jpeg",
    altura_vuelo_m=50       # altitud en metros
)
```

O bien ejecuta directamente:

```bash
python actividad_final.py
```

> Cambia `ruta_imagen` y `altura_vuelo_m` en el bloque `__main__` según tu caso.

## Parámetros

| Parámetro | Descripción | Default |
|---|---|---|
| `ruta_imagen` | Ruta a la imagen aérea/satelital | — |
| `altura_vuelo_m` | Altitud de toma en metros | — |
| `fov_grados` | Campo de visión horizontal de la cámara | `82.0°` |

## Salida

- Visualización con 6 paneles del pipeline de segmentación
- Reporte con GSD, área estimada, volumen y camiones requeridos
- Diccionario con `gsd_m`, `area_m2` y `camiones`