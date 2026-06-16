# zargx

zargx es una herramienta de procesamiento digital de imagenes (PDI) para
detectar y estimar el area de sargazo en imagenes aereas/satelitales, con
calculo basico de logistica de recoleccion.

## Que hace

- Segmenta candidatos de sargazo con umbrales HSV/RGB adaptativos por imagen.
- Estima contexto de agua/costa para reducir falsos positivos tierra adentro.
- Descarta vegetacion, arena clara, techos, sombrillas y objetos calidos muy
  saturados.
- Filtra blobs por area, distancia al agua y forma geometrica.
- Calcula area real en m2 usando fotogrametria (GSD).
- Estima volumen y camiones requeridos con un espesor de biomasa asumido.

## Requisitos

Python 3.10+ y Node.js 18+.

Instala dependencias de Python:

```bash
pip install -r requirements.txt
```

Instala dependencias de React:

```bash
npm install
```

## Como correr la app web

Usa dos terminales abiertas en la carpeta del proyecto.

Terminal 1 - backend Python:

```bash
.venv\Scripts\python.exe server.py
```

Terminal 2 - frontend React:

```bash
npm run dev
```

Luego abre:

```text
http://127.0.0.1:5173
```

La web permite seleccionar una imagen, capturar altitud/FOV y ajustar los datos
de recoleccion. Al terminar muestra area, volumen, camiones requeridos y una
vista segmentada lista para descargar.

El backend queda en `http://127.0.0.1:5000`. Los resultados temporales se guardan
en `salidas/`, carpeta ignorada por Git.

Para apagar la app, detén ambos procesos con `Ctrl+C` en sus terminales.

## Deploy en Render

El proyecto esta preparado para desplegarse como un solo Web Service en Render.
Render construye React en `dist/` y Flask sirve tanto la app como la API desde
el mismo dominio.

Configuracion recomendada:

```text
Build command: npm ci && npm run build && pip install -r requirements.txt
Start command: gunicorn server:app --bind 0.0.0.0:$PORT
```

Tambien se incluye `render.yaml` para crear el servicio desde Blueprint. En
produccion la app usa rutas relativas, por lo que `/api/analyze` y `/outputs/`
funcionan en el mismo dominio publico de Render.

## Uso desde Python

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
| `espesor_biomasa_m` | Espesor promedio para estimar volumen | `0.05` |
| `capacidad_camion_m3` | Capacidad usada para calcular camiones | `14.0` |
| `mostrar` | Muestra figuras con matplotlib | `True` |
| `guardar_figura` | Guarda la figura diagnostica en disco | `None` |
| `devolver_mascaras` | Devuelve mascaras intermedias para depuracion | `False` |

## Nota

El algoritmo sigue siendo PDI clasico, no un modelo entrenado. Por eso es mas
flexible que un umbral HSV fijo, pero en imagenes muy distintas puede requerir
ajustar reglas o migrar a un modelo supervisado con ejemplos anotados.
