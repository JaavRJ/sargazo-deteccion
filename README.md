# Sargazo Deteccion

Herramienta de procesamiento digital de imagenes (PDI) para detectar y estimar
el area de sargazo en imagenes aereas/satelitales, con calculo basico de
logistica de recoleccion.

## Que hace

- Segmenta candidatos de sargazo con umbrales HSV/RGB adaptativos por imagen.
- Estima contexto de agua/costa para reducir falsos positivos tierra adentro.
- Descarta vegetacion, arena clara, techos, sombrillas y objetos calidos muy
  saturados.
- Filtra blobs por area, distancia al agua y forma geometrica.
- Calcula area real en m2 usando fotogrametria (GSD).
- Estima volumen y camiones requeridos con un espesor de biomasa asumido.

## Requisitos

```bash
pip install -r requirements.txt
```

## Uso rapido

```python
from actividad_final import analizar_sargazo_fotogrametria

resultado = analizar_sargazo_fotogrametria(
    ruta_imagen="prueba1.jpeg",
    altura_vuelo_m=50,
)

print(resultado)
```

Tambien se puede ejecutar directamente:

```bash
python actividad_final.py
```

## Guardar diagnosticos

Para evaluar la mascara sin abrir ventanas:

```python
resultado = analizar_sargazo_fotogrametria(
    ruta_imagen="prueba1.jpeg",
    altura_vuelo_m=50,
    mostrar=False,
    guardar_figura="salidas/prueba1_segmentacion.png",
)
```

Esto genera la figura del pipeline y un reporte con sufijo `_reporte`.

## Parametros

| Parametro | Descripcion | Default |
|---|---|---|
| `ruta_imagen` | Ruta a la imagen aerea/satelital | requerido |
| `altura_vuelo_m` | Altitud de toma en metros | requerido |
| `fov_grados` | Campo de vision horizontal de la camara | `82.0` |
| `mostrar` | Muestra figuras con matplotlib | `True` |
| `guardar_figura` | Guarda la figura diagnostica en disco | `None` |
| `devolver_mascaras` | Devuelve mascaras intermedias para depuracion | `False` |

## Nota

El algoritmo sigue siendo PDI clasico, no un modelo entrenado. Por eso es mas
flexible que un umbral HSV fijo, pero en imagenes muy distintas puede requerir
ajustar reglas o migrar a un modelo supervisado con ejemplos anotados.
