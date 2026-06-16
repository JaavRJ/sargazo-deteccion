"""
analizar_sargazo_riviera.py
===========================
Módulo de estimación logística de recolección de sargazo costero.
Dominio geográfico: corredor Akumal – Bahía Soliman – Tulum.
Caracterización del dominio: arena coralina blanca (alta reflectancia),
agua de baja turbidez (cian claro), iluminación tropical directa.

Restricciones de implementación:
  - Álgebra de matrices espaciales pura (OpenCV + NumPy + math).
  - Sin ML, sin cv2.findContours, sin cv2.connectedComponentsWithStats.
  - Todas las funciones son puras, sin efectos secundarios.
"""

import math
import cv2
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# BLOQUE I — FUNCIONES PRIMITIVAS DE PDI
# =============================================================================

def convertir_a_hsv_canal(img_bgr, canal):
    """
    Convierte una imagen BGR al espacio HSV y extrae un canal individual.

    canal: 0 = Tono (H)   [0 – 179]
           1 = Saturación (S) [0 – 255]
           2 = Valor / Brillo (V) [0 – 255]
    Retorna: ndarray float32 del canal solicitado.
    """
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return img_hsv[:, :, canal].astype(np.float32)


def segmentar_umbral(canal_f, tipo, t1=None, t2=None):
    """
    Binariza un canal con la estrategia indicada.

    tipo="banda"    → 255 donde t1 ≤ canal ≤ t2   (umbralización de banda)
    tipo="fija"     → 255 donde canal  > t1         (umbral fijo manual)
    tipo="otsu"     → umbral global automático (Otsu)
    tipo="media"    → umbral = media del canal
    tipo="inversa"  → NOT del umbral fijo (complementaria)
    tipo="kapur"    → búsqueda de entropía máxima (Kapur) sobre el histograma

    Retorna: máscara binaria uint8 (0 / 255).
    """
    c = np.clip(canal_f, 0, 255).astype(np.uint8)

    if tipo == "banda":
        return cv2.inRange(c,
                           np.array(int(t1), dtype=np.uint8),
                           np.array(int(t2), dtype=np.uint8))

    if tipo == "fija":
        _, m = cv2.threshold(c, int(t1), 255, cv2.THRESH_BINARY)
        return m

    if tipo == "otsu":
        _, m = cv2.threshold(c, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return m

    if tipo == "media":
        t_media = float(c.mean())
        _, m = cv2.threshold(c, t_media, 255, cv2.THRESH_BINARY)
        return m

    if tipo == "inversa":
        _, m = cv2.threshold(c, int(t1), 255, cv2.THRESH_BINARY_INV)
        return m

    if tipo == "kapur":
        hist = cv2.calcHist([c], [0], None, [256], [0, 256]).flatten()
        hist_norm = hist / (hist.sum() + 1e-10)
        mejor_t, mejor_H = 0, -np.inf
        for t in range(1, 255):
            p0 = hist_norm[:t];  p1 = hist_norm[t:]
            s0 = p0.sum();       s1 = p1.sum()
            if s0 < 1e-10 or s1 < 1e-10:
                continue
            e0 = -np.sum((p0 / s0) * np.log2(p0 / s0 + 1e-10))
            e1 = -np.sum((p1 / s1) * np.log2(p1 / s1 + 1e-10))
            H_total = e0 + e1
            if H_total > mejor_H:
                mejor_H = H_total
                mejor_t = t
        _, m = cv2.threshold(c, mejor_t, 255, cv2.THRESH_BINARY)
        return m

    raise ValueError(f"Tipo de umbralización desconocido: '{tipo}'")


def morfologia(mask, op, k):
    """
    Aplica una operación morfológica sobre una máscara binaria.

    op: "apertura"   → erosión → dilatación   (elimina ruido pequeño)
        "cierre"     → dilatación → erosión   (consolida regiones)
        "dilatacion" → dilatación simple
        "erosion"    → erosión simple
        "gradiente"  → diferencia dilatación − erosión (realza bordes)
        "tophat"     → imagen original − apertura (resalta picos claros)
        "blackhat"   → cierre − imagen original (resalta valles oscuros)

    k: lado del elemento estructurante elíptico (píxeles).
    Retorna: máscara morfológicamente transformada (uint8).
    """
    elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(k), int(k)))
    tabla = {
        "apertura"  : cv2.MORPH_OPEN,
        "cierre"    : cv2.MORPH_CLOSE,
        "dilatacion": cv2.MORPH_DILATE,
        "erosion"   : cv2.MORPH_ERODE,
        "gradiente" : cv2.MORPH_GRADIENT,
        "tophat"    : cv2.MORPH_TOPHAT,
        "blackhat"  : cv2.MORPH_BLACKHAT,
    }
    if op not in tabla:
        raise ValueError(f"Operación morfológica desconocida: '{op}'")
    return cv2.morphologyEx(mask, tabla[op], elem)


def expansion_histograma(canal_f):
    """
    Estiramiento lineal del histograma al rango [0, 255].

    Normaliza el canal para que su mínimo → 0 y su máximo → 255,
    mejorando el contraste global antes de la segmentación.
    Retorna: canal float32 en [0, 255].
    """
    c_min = canal_f.min()
    c_max = canal_f.max()
    rango = c_max - c_min + 1e-8
    return ((canal_f - c_min) / rango * 255.0).astype(np.float32)


def filtro_bilateral(img_bgr, d=7, sigma_color=35, sigma_space=35):
    """
    Filtro bilateral: suaviza preservando los bordes del sargazo.

    A diferencia del Gaussiano puro, el bilateral pondera cada píxel
    vecino por su similitud de color, conservando las fronteras entre
    clases espectralmente distintas (sargazo / arena / agua).

    Retorna: imagen BGR filtrada (uint8).
    """
    return cv2.bilateralFilter(img_bgr, d=int(d),
                               sigmaColor=float(sigma_color),
                               sigmaSpace=float(sigma_space))


def realce_blackhat(canal_v_f, k=15):
    """
    Black-hat morfológico sobre el canal de Valor (V).

    Calcula: cierre(V) − V, que resalta las zonas OSCURAS sobre un
    fondo claro. En el dominio Akumal-Tulum (arena coralina muy blanca),
    el sargazo marrón aparece como un valle oscuro → el Black-hat lo
    amplifica y facilita su binarización posterior.

    Retorna: imagen de realce uint8 en [0, 255].
    """
    v_uint8 = np.clip(canal_v_f, 0, 255).astype(np.uint8)
    elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(k), int(k)))
    return cv2.morphologyEx(v_uint8, cv2.MORPH_BLACKHAT, elem)


# =============================================================================
# BLOQUE II — FUNCIÓN PRINCIPAL
# =============================================================================

def analizar_sargazo_riviera(ruta_imagen, altura_vuelo_m, fov_grados=82.0):
    """
    Pipeline completo de segmentación de sargazo y estimación logística.

    Parámetros
    ----------
    ruta_imagen    : str   Ruta a la imagen satelital (JPG / PNG).
    altura_vuelo_m : float Altitud de la toma en metros.
    fov_grados     : float Campo de visión horizontal de la cámara (default 82°).

    Retorna
    -------
    dict  {gsd_m, area_m2, volumen_m3, camiones, pixeles_sargazo}
    """

    print("\n" + "=" * 62)
    print("  ANÁLISIS SARGAZO — CORREDOR AKUMAL · SOLIMAN · TULUM")
    print("=" * 62)

    # ─────────────────────────────────────────────────────────────────
    # PASO 1 — CARGA Y VALIDACIÓN
    # ─────────────────────────────────────────────────────────────────
    img_bgr = cv2.imread(ruta_imagen)
    if img_bgr is None:
        raise ValueError(f"No se pudo cargar la imagen: '{ruta_imagen}'")
    img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    alto_px, ancho_px = img_bgr.shape[:2]
    print(f"  Resolución         : {ancho_px} × {alto_px} px")

    # ─────────────────────────────────────────────────────────────────
    # PASO 2 — FOTOGRAMETRÍA DINÁMICA (GSD)
    # ─────────────────────────────────────────────────────────────────
    fov_rad       = math.radians(fov_grados)
    ancho_fisico  = 2.0 * altura_vuelo_m * math.tan(fov_rad / 2.0)
    gsd_m_px      = ancho_fisico / ancho_px
    area_px_m2    = gsd_m_px ** 2
    print(f"  Altitud / FOV      : {altura_vuelo_m} m  /  {fov_grados}°")
    print(f"  GSD                : {gsd_m_px:.5f} m/px")
    print(f"  Área por píxel     : {area_px_m2:.7f} m²")

    # ─────────────────────────────────────────────────────────────────
    # PASO 3 — PREPROCESAMIENTO ESPECTRAL
    #
    # 3a. Filtro bilateral: elimina ruido de texturas del mar y de
    #     la selva preservando los bordes espectrales del sargazo.
    # 3b. Black-hat morfológico sobre V: amplifica el sargazo oscuro
    #     frente a la arena coralina blanca característica del dominio.
    # 3c. Expansión de histograma sobre S: maximiza el contraste entre
    #     el sargazo (saturación media) y la espuma blanca (S≈0) o el
    #     agua turquesa (S alta) antes de binarizar.
    # ─────────────────────────────────────────────────────────────────
    img_pre  = filtro_bilateral(img_bgr, d=7, sigma_color=35, sigma_space=35)

    canal_H  = convertir_a_hsv_canal(img_pre, 0)
    canal_S  = convertir_a_hsv_canal(img_pre, 1)
    canal_V  = convertir_a_hsv_canal(img_pre, 2)

    canal_S_exp = expansion_histograma(canal_S)

    blackhat_V  = realce_blackhat(canal_V, k=15).astype(np.float32)

    # ─────────────────────────────────────────────────────────────────
    # PASO 4 — TRIPLE CONDICIÓN HSV (arquitectura obligatoria)
    #
    # Umbrales calibrados sobre el corredor Akumal-Tulum:
    #
    # H ∈ [5, 28]:
    #   Retiene solo tonos rojo-naranja-marrón del sargazo seco.
    #   En el espacio HSV, el marrón cae entre H=5 y H=28.
    #   El agua cian del Caribe tiene H > 85 → queda fuera.
    #   La arena coralina tiene H variable; la condición S la excluye.
    #
    # S ∈ [40, 255] (sobre el canal expandido):
    #   La arena blanca y la espuma tienen S ≈ 0–25 → excluidas.
    #   El agua tiene S alta pero es filtrada por H y V.
    #   El sargazo posee saturación moderada → sobrevive.
    #
    # V ∈ [20, 140]:
    #   Excluye sombras de la selva (V < 20) y brillos de sol
    #   sobre arena o espuma (V > 140).
    #   El sargazo marrón oscuro ocupa V ≈ 30–135 en este dominio.
    # ─────────────────────────────────────────────────────────────────
    m_H      = segmentar_umbral(canal_H,     tipo="banda", t1=5,  t2=28)
    m_S      = segmentar_umbral(canal_S_exp, tipo="banda", t1=40, t2=255)
    m_V      = segmentar_umbral(canal_V,     tipo="banda", t1=20, t2=140)

    m_triple = cv2.bitwise_and(cv2.bitwise_and(m_H, m_S), m_V)

    # ─────────────────────────────────────────────────────────────────
    # PASO 5 — CONDICIÓN ESPECTRAL COMPLEMENTARIA (sugerencia PDI)
    #
    # 5a. Binarización del Black-hat:
    #     Umbral de Otsu sobre el mapa black-hat de V para detectar
    #     las zonas oscuras que el Black-hat resaltó (sargazo en arena).
    #     Se usa Otsu porque la distribución bimodal (arena vs sargazo)
    #     hace que Otsu encuentre el umbral óptimo automáticamente.
    #
    # 5b. Exclusión de agua pura por H:
    #     El agua del Caribe tiene siempre H > 85 en este dominio.
    #     Una binarización inversa sobre H genera una máscara de
    #     "no-agua" que elimina falsos positivos en zonas submarinas.
    #     Se aplica como AND lógico sobre m_triple.
    #
    # 5c. Combinación OR lógica:
    #     m_reforzada = triple AND blackhat_otsu (sargazo sobre arena)
    #     m_noagua    = triple AND no-agua        (sargazo en orilla)
    #     La unión (OR) maximiza la sensibilidad sin perder precisión.
    # ─────────────────────────────────────────────────────────────────
    m_blackhat_bin = segmentar_umbral(blackhat_V, tipo="otsu")
    m_no_agua      = segmentar_umbral(canal_H, tipo="inversa", t1=84)

    m_reforzada = cv2.bitwise_and(m_triple, m_blackhat_bin)
    m_costa     = cv2.bitwise_and(m_triple, m_no_agua)
    m_candidatos = cv2.bitwise_or(m_reforzada, m_costa)

    # ─────────────────────────────────────────────────────────────────
    # PASO 6 — MORFOLOGÍA DE LATICES
    #
    # 6a. Filtro de mediana (3×3):
    #     Elimina ruido sal-pimienta (píxeles aislados) antes de
    #     la morfología estructural, sin desplazar los bordes.
    #
    # 6b. Apertura (k=3):
    #     Erosión seguida de dilatación con kernel pequeño.
    #     Borra objetos más pequeños que el kernel (ruido impulsivo,
    #     artefactos de espuma aislados) conservando el sargazo.
    #
    # 6c. Cierre (k=5):
    #     Dilatación seguida de erosión con kernel mayor.
    #     Conecta fragmentos de biomasa próximos (el sargazo se
    #     deposita en manchas irregulares con huecos internos)
    #     y rellena agujeros menores al kernel.
    # ─────────────────────────────────────────────────────────────────
    m_median   = cv2.medianBlur(m_candidatos, 3)
    m_apertura = morfologia(m_median,   op="apertura", k=3)
    m_final    = morfologia(m_apertura, op="cierre",   k=5)

    # ─────────────────────────────────────────────────────────────────
    # PASO 7 — CÁLCULO LOGÍSTICO MATRICIAL
    # ─────────────────────────────────────────────────────────────────
    pixeles_sargazo = cv2.countNonZero(m_final)
    area_total_m2   = pixeles_sargazo * area_px_m2
    volumen_m3      = area_total_m2 * 0.05
    cap_camion_m3   = 14.0
    camiones        = int(math.ceil(volumen_m3 / cap_camion_m3)) if volumen_m3 > 0 else 0

    print(f"\n  {'─' * 48}")
    print(f"  Píxeles de sargazo   : {pixeles_sargazo:>12,} px")
    print(f"  Área estimada        : {area_total_m2:>12,.4f} m²")
    print(f"  Volumen de biomasa   : {volumen_m3:>12,.4f} m³")
    print(f"  Camiones requeridos  : {camiones:>12}")
    print(f"  {'─' * 48}\n")

    # ─────────────────────────────────────────────────────────────────
    # PASO 8 — SALIDA VISUAL (grid matplotlib)
    # ─────────────────────────────────────────────────────────────────
    overlay = img_rgb.copy()
    overlay[m_final > 0] = [220, 60, 0]
    blend   = cv2.addWeighted(img_rgb, 0.52, overlay, 0.48, 0)

    fig = plt.figure(figsize=(22, 12))
    fig.patch.set_facecolor("#16161e")
    fig.suptitle(
        f"Análisis de Sargazo — Corredor Akumal · Soliman · Tulum"
        f"   |   Altitud {altura_vuelo_m} m",
        fontsize=14, fontweight="bold", color="white", y=1.005
    )

    specs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.08)
    paneles = [
        (specs[0, 0], img_rgb,    None,   "① Imagen original (RGB)"),
        (specs[0, 1], m_H,        "gray", f"② Máscara H  [5 – 28]\n"
                                           f"Sargazo marrón-rojizo"),
        (specs[0, 2], m_S,        "gray", f"③ Máscara S  [40 – 255]\n"
                                           f"(canal expandido — excluye arena/espuma)"),
        (specs[1, 0], m_V,        "gray", f"④ Máscara V  [20 – 140]\n"
                                           f"Excluye sombras y brillos"),
        (specs[1, 1], m_final,    "gray", f"⑤ Máscara final\n"
                                           f"(triple HSV + blackhat + morfo)"),
        (specs[1, 2], blend,      None,   f"⑥ Overlay sargazo (naranja)\n"
                                           f"{pixeles_sargazo:,} px  ·  "
                                           f"{area_total_m2:,.2f} m²"),
    ]

    for spec, data, cmap, titulo in paneles:
        ax = fig.add_subplot(spec)
        ax.imshow(data, cmap=cmap)
        ax.set_title(titulo, color="white", fontsize=9.5,
                     fontweight="bold", pad=5)
        ax.set_facecolor("#16161e")
        ax.axis("off")

    fig2, ax_rep = plt.subplots(figsize=(10, 5.5))
    fig2.patch.set_facecolor("#16161e")
    ax_rep.set_facecolor("#16161e")
    fig2.suptitle("Reporte Logístico de Recolección",
                  fontsize=14, fontweight="bold", color="white")
    ax_rep.axis("off")

    txt = (
        f"FOTOGRAMETRÍA\n{'─' * 44}\n"
        f"  FOV horizontal           : {fov_grados}°\n"
        f"  GSD (metros por píxel)   : {gsd_m_px:.5f} m/px\n"
        f"  Área por píxel           : {area_px_m2:.7f} m²\n"
        f"  Resolución de la imagen  : {ancho_px} × {alto_px} px\n\n"
        f"SEGMENTACIÓN PDI\n{'─' * 44}\n"
        f"  Máscara H activa (5-28)  : {cv2.countNonZero(m_H):>10,} px\n"
        f"  Máscara S activa (40-255): {cv2.countNonZero(m_S):>10,} px\n"
        f"  Máscara V activa (20-140): {cv2.countNonZero(m_V):>10,} px\n"
        f"  Intersección triple AND  : {cv2.countNonZero(m_triple):>10,} px\n"
        f"  Pixeles finales sargazo  : {pixeles_sargazo:>10,} px\n\n"
        f"LOGÍSTICA DE RECOLECCIÓN\n{'─' * 44}\n"
        f"  Área real estimada       : {area_total_m2:>10,.4f} m²\n"
        f"  Espesor de biomasa       : 0.05 m\n"
        f"  Volumen de biomasa       : {volumen_m3:>10,.4f} m³\n"
        f"  Capacidad por camión     : {cap_camion_m3:.1f} m³\n"
        f"  Camiones requeridos      : {camiones:>10}\n"
    )
    ax_rep.text(0.04, 0.97, txt,
                transform=ax_rep.transAxes,
                fontsize=11, verticalalignment="top",
                fontfamily="monospace", color="#dcdcdc",
                bbox=dict(facecolor="#1e1e2e", alpha=0.95,
                          boxstyle="round,pad=0.9", edgecolor="#44445e"))
    plt.tight_layout()
    plt.show()

    return {
        "gsd_m"          : gsd_m_px,
        "area_m2"        : area_total_m2,
        "volumen_m3"     : volumen_m3,
        "camiones"       : camiones,
        "pixeles_sargazo": pixeles_sargazo,
    }


# =============================================================================
# EJECUCIÓN
# =============================================================================
if __name__ == "__main__":
    analizar_sargazo_riviera(
        ruta_imagen    = r"C:\Users\jasma\Documents\ESCOM\4to Semestre\PDI\Actividad Final\sargazo-deteccion\Tankah\Captura de pantalla (127).png",
        altura_vuelo_m = 200
    )