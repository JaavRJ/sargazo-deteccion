import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# Segmentacion flexible de sargazo
# =============================================================================
# La version anterior dependia casi por completo de rangos HSV fijos. Eso funciona
# en una toma concreta, pero tambien deja pasar techos, sombrillas u objetos
# rojo-naranja. Esta version mantiene la base cromatica, pero agrega:
#
# 1. Umbrales adaptativos por imagen para tolerar cambios de iluminacion.
# 2. Mascara de agua/costa para exigir contexto litoral cuando la imagen lo tiene.
# 3. Reglas de contorno para descartar blobs compactos y artificiales.
# 4. Salida opcional de figuras para validar el pipeline sin abrir ventanas.
# =============================================================================


def _u8(mask_bool):
    """Convierte una mascara booleana a uint8 compatible con OpenCV."""
    return mask_bool.astype(np.uint8) * 255


def _odd_size(value, min_size=3):
    """Devuelve un tamano impar para kernels morfologicos."""
    value = int(round(value))
    if value % 2 == 0:
        value += 1
    return max(min_size, value)


def _componentes_grandes(mask, area_min_px):
    """Conserva solo componentes conectados con area suficiente."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    salida = np.zeros_like(mask)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= area_min_px:
            salida[labels == label] = 255

    return salida


def _filtrar_componentes_agua(mask, H, S, V, R, G, B, total_px, area_min_px):
    """Conserva cuerpos de agua probables y descarta vegetacion/objetos verdes."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    salida = np.zeros_like(mask)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < area_min_px:
            continue

        pix = labels == label
        h_media = float(np.mean(H[pix]))
        s_media = float(np.mean(S[pix]))
        v_media = float(np.mean(V[pix]))
        r_media = float(np.mean(R[pix]))
        g_media = float(np.mean(G[pix]))
        b_media = float(np.mean(B[pix]))
        exg_media = float(np.mean(2 * G[pix] - R[pix] - B[pix]))

        componente_grande = area >= total_px * 0.08
        azul_turquesa = h_media >= 80 and b_media >= r_media + 7
        agua_verde_extensa = area >= total_px * 0.035 and s_media < 92 and exg_media < 36
        muy_brillante = v_media > 170 and s_media < 35

        if not muy_brillante and (componente_grande or azul_turquesa or agua_verde_extensa):
            salida[pix] = 255

    return salida


def _crear_mascara_contexto(H, S, V, R, G, B, alto_px, ancho_px):
    """Estima agua, costa y clases que suelen producir falsos positivos."""
    total_px = alto_px * ancho_px
    min_dim = min(alto_px, ancho_px)
    V32 = V.astype(np.float32)
    media_v = cv2.blur(V32, (9, 9))
    std_v = np.sqrt(np.maximum(cv2.blur(V32 * V32, (9, 9)) - media_v * media_v, 0))
    suave = std_v < 18

    # Vegetacion: verde dominante por tono y por exceso de verde.
    exg = 2 * G - R - B
    vegetacion = (
        ((H >= 32) & (H <= 88) & (S > 45) & (G > R + 4) & (G > B + 3))
        | ((exg > 22) & (G > R + 6) & (G > B + 4) & (V > 45))
    )

    # Arena/espuma/estructuras blancas: alto brillo y poca saturacion.
    v_alto = max(155, np.percentile(V, 78))
    arena_clara = (V > v_alto) & (S < 58) & (R > 105) & (G > 100) & (B > 85)

    # Techos/sombrillas/senales: calidos, muy saturados y relativamente brillantes.
    artificial_calido = (
        (((H <= 24) | (H >= 165)) & (S > 105) & (V > 92) & ((R - G) > 12) & ((R - B) > 20))
        | ((H > 24) & (H <= 36) & (S > 120) & (V > 105) & ((R - G) > 12))
    )

    # Agua: se combina tono HSV con dominancia azul-verde. La limpieza por
    # componentes grandes evita que lineas de interfaz o anotaciones del mapa
    # creen una costa falsa.
    agua_azul_cian = (H >= 78) & (H <= 132) & (S > 16) & (V > 22)
    agua_verde_suave = (H >= 45) & (H < 78) & (S > 12) & (V > 25) & (V < 190) & suave
    agua_turbia = (H >= 28) & (H < 45) & (S > 16) & (V > 25) & (V < 150) & (B > R - 22) & suave
    agua_dominante = (
        ((G + B) * 0.5 > R + 3)
        & ((G > R + 2) | (B > R + 2))
        & (S > 8)
        & (V > 20)
        & (((H >= 78) & (H <= 132)) | suave)
    )
    vegetacion_densa = vegetacion & (S > 70) & (G > B + 18) & (G > R + 18)
    agua = (agua_azul_cian | agua_verde_suave | agua_turbia | agua_dominante) & ~vegetacion_densa & ~arena_clara
    agua_u8 = _u8(agua)

    k_agua = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd_size(min_dim * 0.009),) * 2)
    agua_u8 = cv2.morphologyEx(agua_u8, cv2.MORPH_OPEN, k_agua)
    agua_u8 = cv2.morphologyEx(agua_u8, cv2.MORPH_CLOSE, k_agua, iterations=2)
    agua_u8 = _filtrar_componentes_agua(
        agua_u8,
        H,
        S,
        V,
        R,
        G,
        B,
        total_px,
        max(450, int(total_px * 0.0035)),
    )

    hay_agua = cv2.countNonZero(agua_u8) >= total_px * 0.025
    radio_costa = _odd_size(min_dim * 0.045, min_size=11)
    k_costa = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radio_costa, radio_costa))
    cerca_agua_u8 = cv2.dilate(agua_u8, k_costa, iterations=1)

    return {
        "vegetacion": vegetacion,
        "arena_clara": arena_clara,
        "artificial_calido": artificial_calido,
        "agua_u8": agua_u8,
        "cerca_agua_u8": cerca_agua_u8,
        "hay_agua": hay_agua,
        "radio_costa": radio_costa,
    }


def _crear_candidatos_sargazo(H, S, V, R, G, B, contexto):
    """Genera candidatos cromaticos de sargazo con limites adaptativos."""
    v_min = max(8, np.percentile(V, 1))
    v_max = min(190, max(105, np.percentile(V, 86) + 12))
    s_max = min(205, max(115, np.percentile(S, 96)))

    tono_rojo_marron = (H <= 26) | (H >= 150)
    tono_ocre_marron = (H > 26) & (H <= 40) & (S > 20) & ((R - G) > 3) & ((R - B) > 12)

    mas_rojo_que_azul = (R - B) > 5
    balance_marron = (R - G) > -3
    marron_oscuro = (
        (V < min(v_max, np.percentile(V, 67)))
        & (S > 8)
        & ((H <= 34) | (H >= 150))
        & (R > B + 5)
        & (R >= G - 2)
    )

    semillas = (
        ((tono_rojo_marron | tono_ocre_marron) & (S > 9) & (S < s_max) & (V > v_min) & (V < v_max))
        & mas_rojo_que_azul
        & balance_marron
    ) | marron_oscuro

    crecimiento_local = (
        (((H <= 62) | (H >= 145)) & (S > 8) & (S < min(215, s_max + 22)))
        & (V > v_min)
        & (V < min(205, v_max + 28))
        & ((R - B) > -3)
        & ((R - G) > -28)
    )

    exclusiones = (
        contexto["vegetacion"]
        | contexto["arena_clara"]
        | contexto["artificial_calido"]
    )

    semillas = semillas & ~exclusiones
    crecimiento_local = crecimiento_local & ~exclusiones

    if contexto["hay_agua"]:
        cerca_agua = contexto["cerca_agua_u8"] > 0
        semillas &= cerca_agua
        crecimiento_local &= cerca_agua

    k_crecer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    vecindad_semillas = cv2.dilate(_u8(semillas), k_crecer, iterations=1) > 0
    candidatos = semillas | (crecimiento_local & vecindad_semillas)

    return _u8(candidatos), _u8(semillas | crecimiento_local)


def _postprocesar_mascara(mask, alto_px, ancho_px):
    """Limpia ruido y conecta segmentos naturales de la misma franja."""
    min_dim = min(alto_px, ancho_px)

    k_mediana = _odd_size(min_dim * 0.008, min_size=3)
    k_apertura = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd_size(min_dim * 0.006),) * 2)
    k_cierre = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd_size(min_dim * 0.022),) * 2)

    salida = cv2.medianBlur(mask, k_mediana)
    salida = cv2.morphologyEx(salida, cv2.MORPH_OPEN, k_apertura)
    salida = cv2.morphologyEx(salida, cv2.MORPH_CLOSE, k_cierre, iterations=1)
    return salida


def _filtrar_contornos(mask, contexto, H, S, V, R, G, B, alto_px, ancho_px):
    """Filtra blobs por area, distancia al agua y forma geometrica."""
    total_px = alto_px * ancho_px
    area_min_px = max(70, int(total_px * 0.00016))

    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mascara_final = np.zeros_like(mask)
    blobs_validos = []

    agua_u8 = contexto["agua_u8"]
    cerca_agua_u8 = contexto["cerca_agua_u8"]
    artificial_u8 = _u8(contexto["artificial_calido"])

    if contexto["hay_agua"]:
        distancia_agua = cv2.distanceTransform(cv2.bitwise_not(agua_u8), cv2.DIST_L2, 5)
    else:
        distancia_agua = None

    for cnt in contornos:
        area_contorno = cv2.contourArea(cnt)
        if area_contorno < area_min_px:
            continue

        region = np.zeros_like(mask)
        cv2.drawContours(region, [cnt], -1, 255, -1)
        area_px = cv2.countNonZero(region)
        if area_px < area_min_px:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        lado_menor = max(1.0, min(rw, rh))
        elongacion = max(rw, rh) / lado_menor
        perimetro = max(1.0, cv2.arcLength(cnt, True))
        circularidad = 4 * math.pi * area_contorno / (perimetro * perimetro)
        extension = area_contorno / max(1.0, w * h)

        pix = region > 0
        artificial_frac = cv2.countNonZero(cv2.bitwise_and(region, artificial_u8)) / area_px
        s_media = float(np.mean(S[pix]))
        v_media = float(np.mean(V[pix]))
        r_menos_g = float(np.mean(R[pix] - G[pix]))
        r_menos_b = float(np.mean(R[pix] - B[pix]))

        compacto_artificial = (
            circularidad > 0.42
            and elongacion < 2.35
            and extension > 0.36
            and area_px < area_min_px * 10
        )
        color_artificial = (
            artificial_frac > 0.12
            or (s_media > 112 and v_media > 96 and r_menos_g > 10 and elongacion < 3.0)
        )

        if compacto_artificial or color_artificial:
            continue

        if contexto["hay_agua"]:
            cerca_frac = cv2.countNonZero(cv2.bitwise_and(region, cerca_agua_u8)) / area_px
            agua_frac = cv2.countNonZero(cv2.bitwise_and(region, agua_u8)) / area_px
            dist_mediana = float(np.median(distancia_agua[pix]))

            blob_natural_sin_contexto = elongacion >= 5.5 and area_px >= area_min_px * 18
            tiene_contexto = cerca_frac >= 0.72 or agua_frac >= 0.10
            lejos_de_costa = dist_mediana > contexto["radio_costa"] * 0.58 and agua_frac < 0.08
            pequeno_sin_contacto = area_px < area_min_px * 7 and elongacion < 3.2 and agua_frac < 0.03
            pequeno_compacto = area_px < area_min_px * 9 and elongacion < 2.45
            lobulo_costero = (
                area_px >= area_min_px * 2.2
                and cerca_frac >= 0.92
                and r_menos_b > 7
                and r_menos_g > -6
            )

            if (
                (not tiene_contexto and not blob_natural_sin_contexto)
                or (lejos_de_costa and not blob_natural_sin_contexto)
                or pequeno_sin_contacto
                or (pequeno_compacto and not lobulo_costero)
            ):
                continue

        cv2.drawContours(mascara_final, [cnt], -1, 255, -1)
        blobs_validos.append((area_px, cnt))

    blobs_validos.sort(key=lambda item: -item[0])
    return mascara_final, blobs_validos


def _expandir_bordes_conectados(mask, contexto, H, S, V, R, G, B, alto_px, ancho_px):
    """
    Recupera grosor y lobulos pegados a la franja principal sin aceptar objetos
    aislados. El crecimiento solo sale de componentes grandes ya validadas.
    """
    total_px = alto_px * ancho_px
    area_semilla_px = max(700, int(total_px * 0.004))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    semillas = np.zeros_like(mask)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= area_semilla_px:
            semillas[labels == label] = 255

    if cv2.countNonZero(semillas) == 0:
        return mask

    # Mascara mas permisiva para bordes de sargazo humedo/ocre que suelen ser
    # mas verdosos o claros que el centro de la acumulacion.
    borde_suave = (
        (((H <= 76) | (H >= 145)) & (S > 5) & (S < 180))
        & (V > 28)
        & (V < 205)
        & ((R - B) > -15)
        & ((R - G) > -38)
        & (contexto["cerca_agua_u8"] > 0)
    )

    vegetacion_densa = (H >= 45) & (H <= 88) & (S > 70) & (G > R + 18) & (G > B + 12)
    arena_clara = (V > 165) & (S < 38) & (R > 110) & (G > 105) & (B > 90)
    artificial_calido = (
        ((H <= 24) | (H >= 165))
        & (S > 105)
        & (V > 92)
        & ((R - G) > 12)
        & ((R - B) > 20)
    )

    borde_suave = _u8(borde_suave & ~vegetacion_densa & ~arena_clara & ~artificial_calido)

    k_crecimiento = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    crecimiento = semillas.copy()
    for _ in range(8):
        crecimiento = cv2.bitwise_and(cv2.dilate(crecimiento, k_crecimiento, iterations=1), borde_suave)

    expandida = cv2.bitwise_or(mask, crecimiento)

    # Une cortes cortos entre tramos grandes de la misma franja. No se usa en
    # objetos pequenos para no reconectar rocas o estructuras aisladas.
    contornos, _ = cv2.findContours(expandida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    grandes = []
    for cnt in contornos:
        region = np.zeros_like(mask)
        cv2.drawContours(region, [cnt], -1, 255, -1)
        area_px = cv2.countNonZero(region)
        if area_px >= area_semilla_px:
            grandes.append((area_px, cnt))

    grosor_puente = max(5, _odd_size(min(alto_px, ancho_px) * 0.012))
    distancia_max = min(alto_px, ancho_px) * 0.075

    for i in range(len(grandes)):
        pts_i = grandes[i][1].reshape(-1, 2)
        pts_i = pts_i[:: max(1, len(pts_i) // 250)]
        for j in range(i + 1, len(grandes)):
            pts_j = grandes[j][1].reshape(-1, 2)
            pts_j = pts_j[:: max(1, len(pts_j) // 250)]

            dif = pts_i[:, None, :] - pts_j[None, :, :]
            dist2 = np.sum(dif * dif, axis=2)
            idx = np.unravel_index(np.argmin(dist2), dist2.shape)
            distancia = math.sqrt(float(dist2[idx]))

            if distancia <= distancia_max:
                p1 = tuple(int(v) for v in pts_i[idx[0]])
                p2 = tuple(int(v) for v in pts_j[idx[1]])
                cv2.line(expandida, p1, p2, 255, thickness=grosor_puente)

    return expandida


def _contornos_validos_desde_mascara(mask, alto_px, ancho_px):
    """Reconstruye la lista de blobs despues de una expansion controlada."""
    area_min_px = max(70, int(alto_px * ancho_px * 0.00016))
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mascara_limpia = np.zeros_like(mask)
    blobs_validos = []

    for cnt in contornos:
        region = np.zeros_like(mask)
        cv2.drawContours(region, [cnt], -1, 255, -1)
        area_px = cv2.countNonZero(region)
        if area_px >= area_min_px:
            cv2.drawContours(mascara_limpia, [cnt], -1, 255, -1)
            blobs_validos.append((area_px, cnt))

    blobs_validos.sort(key=lambda item: -item[0])
    return mascara_limpia, blobs_validos


def analizar_sargazo_fotogrametria(
    ruta_imagen,
    altura_vuelo_m,
    fov_grados=82.0,
    espesor_biomasa_m=0.05,
    capacidad_camion_m3=14.0,
    mostrar=True,
    guardar_figura=None,
    devolver_mascaras=False,
):
    """
    Segmenta sargazo costero y calcula logistica de recoleccion usando
    fotogrametria (GSD) y un pipeline PDI flexible.

    Parametros
    ----------
    ruta_imagen: str
        Ruta a la imagen aerea/satelital.
    altura_vuelo_m: float
        Altitud de toma en metros.
    fov_grados: float
        Campo de vision horizontal de la camara.
    espesor_biomasa_m: float
        Espesor promedio asumido para estimar volumen de biomasa.
    capacidad_camion_m3: float
        Capacidad promedio de un camion o recurso de recoleccion.
    mostrar: bool
        Si True, muestra las figuras con matplotlib.
    guardar_figura: str | None
        Ruta para guardar la figura diagnostica. Tambien guarda un reporte con
        sufijo "_reporte".
    devolver_mascaras: bool
        Si True, agrega mascaras intermedias al diccionario de salida.

    Retorna
    -------
    dict con metricas principales y, opcionalmente, mascaras diagnosticas.
    """
    print("\n" + "=" * 60)
    print("  ANALISIS FOTOGRAMETRICO DE SARGAZO")
    print("=" * 60)
    print(f"  Altitud de toma   : {altura_vuelo_m} m")
    print(f"  FOV horizontal    : {fov_grados} deg")

    # ------------------------------------------------------------------
    # 1. Carga y validacion de imagen
    # ------------------------------------------------------------------
    img_bgr = cv2.imread(str(ruta_imagen))
    if img_bgr is None:
        raise ValueError(f"No se pudo cargar la imagen: {ruta_imagen}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    alto_px, ancho_px = img_bgr.shape[:2]
    print(f"  Resolucion imagen : {ancho_px} x {alto_px} px")

    # ------------------------------------------------------------------
    # 2. Fotogrametria - GSD (Ground Sample Distance)
    # ------------------------------------------------------------------
    fov_rad = math.radians(fov_grados)
    ancho_fisico_m = 2 * altura_vuelo_m * math.tan(fov_rad / 2)
    gsd_m_px = ancho_fisico_m / ancho_px
    area_px_m2 = gsd_m_px ** 2

    print(f"  GSD               : {gsd_m_px:.4f} m/px")
    print(f"  Area por pixel    : {area_px_m2:.6f} m2")

    # ------------------------------------------------------------------
    # 3. Preprocesamiento y espacios de color
    # ------------------------------------------------------------------
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H = img_hsv[:, :, 0].astype(float)
    S = img_hsv[:, :, 1].astype(float)
    V = img_hsv[:, :, 2].astype(float)

    R = img_bgr[:, :, 2].astype(float)
    G = img_bgr[:, :, 1].astype(float)
    B = img_bgr[:, :, 0].astype(float)

    # ------------------------------------------------------------------
    # 4. Contexto costero y candidatos cromaticos
    # ------------------------------------------------------------------
    contexto = _crear_mascara_contexto(H, S, V, R, G, B, alto_px, ancho_px)
    mascara_cromatica, mascara_precontexto = _crear_candidatos_sargazo(H, S, V, R, G, B, contexto)

    # ------------------------------------------------------------------
    # 5. Morfologia y filtrado de contornos
    # ------------------------------------------------------------------
    mascara_morfologica = _postprocesar_mascara(mascara_cromatica, alto_px, ancho_px)
    mascara_final, blobs_validos = _filtrar_contornos(
        mascara_morfologica, contexto, H, S, V, R, G, B, alto_px, ancho_px
    )
    mascara_final = _expandir_bordes_conectados(
        mascara_final, contexto, H, S, V, R, G, B, alto_px, ancho_px
    )
    mascara_final, blobs_validos = _contornos_validos_desde_mascara(mascara_final, alto_px, ancho_px)

    # ------------------------------------------------------------------
    # 6. Calculo de area total y modelo logistico
    # ------------------------------------------------------------------
    pixeles_sargazo = cv2.countNonZero(mascara_final)
    area_total_m2 = pixeles_sargazo * area_px_m2
    volumen_m3 = area_total_m2 * espesor_biomasa_m
    camiones_req = int(np.ceil(volumen_m3 / capacidad_camion_m3)) if volumen_m3 > 0 else 0

    print("\n  --- RESULTADOS ---")
    print(f"  Blobs sargazo detectados : {len(blobs_validos)}")
    print(f"  Pixeles de sargazo       : {pixeles_sargazo:,}")
    print(f"  Area estimada            : {area_total_m2:,.2f} m2")
    print(f"  Volumen estimado         : {volumen_m3:,.2f} m3")
    print(f"  Camiones requeridos      : {camiones_req}")

    # ------------------------------------------------------------------
    # 7. Visualizacion diagnostica
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(
        f"Analisis PDI de Sargazo - Toma a {altura_vuelo_m} m de altitud",
        fontsize=15,
        fontweight="bold",
    )

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(img_rgb)
    ax1.set_title("1. Imagen original (RGB)", fontsize=11)
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(gray_eq, cmap="gray")
    ax2.set_title("2. CLAHE\ncontraste local", fontsize=11)
    ax2.axis("off")

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(mascara_precontexto, cmap="gray")
    ax3.set_title("3. Candidatos por color\nHSV + indices RGB adaptativos", fontsize=11)
    ax3.axis("off")

    ax4 = fig.add_subplot(2, 3, 4)
    contexto_vis = np.zeros((alto_px, ancho_px, 3), dtype=np.uint8)
    contexto_vis[contexto["cerca_agua_u8"] > 0] = [70, 135, 210]
    contexto_vis[contexto["agua_u8"] > 0] = [30, 80, 150]
    contexto_vis[mascara_cromatica > 0] = [255, 255, 255]
    ax4.imshow(contexto_vis)
    ax4.set_title("4. Filtro contextual\nagua/costa + exclusiones", fontsize=11)
    ax4.axis("off")

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(mascara_final, cmap="gray")
    ax5.set_title("5. Mascara final\nmorfologia + forma/distancia", fontsize=11)
    ax5.axis("off")

    ax6 = fig.add_subplot(2, 3, 6)
    overlay = img_rgb.copy()
    overlay[mascara_final > 0] = [220, 80, 20]
    blended = cv2.addWeighted(img_rgb, 0.58, overlay, 0.42, 0)
    ax6.imshow(blended)

    for _, cnt in blobs_validos:
        cnt_squeezed = cnt.squeeze()
        if cnt_squeezed.ndim == 2:
            ax6.plot(
                np.append(cnt_squeezed[:, 0], cnt_squeezed[0, 0]),
                np.append(cnt_squeezed[:, 1], cnt_squeezed[0, 1]),
                color="yellow",
                linewidth=1.2,
                alpha=0.9,
            )

    ax6.set_title("6. Sargazo segmentado\noverlay naranja + contornos", fontsize=11)
    ax6.axis("off")

    fig.tight_layout()

    fig2, ax_rep = plt.subplots(figsize=(9, 5))
    fig2.suptitle("Reporte geometrico y logistico", fontsize=13, fontweight="bold")
    ax_rep.axis("off")

    texto = (
        "FOTOGRAMETRIA\n"
        f"{'-' * 38}\n"
        f"  Campo de vision (FOV)     : {fov_grados} deg\n"
        f"  GSD (tamano de pixel)     : {gsd_m_px:.4f} m/px\n"
        f"  Area por pixel            : {area_px_m2:.6f} m2\n"
        f"  Cobertura imagen          : {ancho_px} x {alto_px} px\n\n"
        "SEGMENTACION PDI\n"
        f"{'-' * 38}\n"
        f"  Blobs detectados          : {len(blobs_validos)}\n"
        f"  Pixeles de sargazo        : {pixeles_sargazo:,} px\n"
        f"  Area real estimada        : {area_total_m2:,.2f} m2\n\n"
        "LOGISTICA DE RECOLECCION\n"
        f"{'-' * 38}\n"
        f"  Espesor biomasa asumido   : {espesor_biomasa_m:.3f} m\n"
        f"  Volumen de biomasa        : {volumen_m3:,.2f} m3\n"
        f"  Capacidad camion          : {capacidad_camion_m3:.1f} m3\n"
        f"  Camiones requeridos       : {camiones_req}\n"
    )

    ax_rep.text(
        0.04,
        0.97,
        texto,
        transform=ax_rep.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(facecolor="#f7f7f7", alpha=0.92, boxstyle="round,pad=0.8", edgecolor="#cccccc"),
    )
    fig2.tight_layout()

    salida_figura = None
    salida_reporte = None
    if guardar_figura:
        salida_figura = Path(guardar_figura)
        salida_figura.parent.mkdir(parents=True, exist_ok=True)
        salida_reporte = salida_figura.with_name(f"{salida_figura.stem}_reporte{salida_figura.suffix}")
        fig.savefig(salida_figura, dpi=155, bbox_inches="tight")
        fig2.savefig(salida_reporte, dpi=155, bbox_inches="tight")

    if mostrar:
        plt.show()
    else:
        plt.close(fig)
        plt.close(fig2)

    resultado = {
        "gsd_m": gsd_m_px,
        "area_m2": area_total_m2,
        "camiones": camiones_req,
        "blobs": len(blobs_validos),
        "pixeles_sargazo": pixeles_sargazo,
        "volumen_m3": volumen_m3,
        "espesor_biomasa_m": espesor_biomasa_m,
        "capacidad_camion_m3": capacidad_camion_m3,
    }

    if salida_figura:
        resultado["figura"] = str(salida_figura)
        resultado["reporte"] = str(salida_reporte)

    if devolver_mascaras:
        resultado["mascaras"] = {
            "agua": contexto["agua_u8"],
            "cerca_agua": contexto["cerca_agua_u8"],
            "candidatos_precontexto": mascara_precontexto,
            "candidatos_contexto": mascara_cromatica,
            "morfologica": mascara_morfologica,
            "final": mascara_final,
        }

    return resultado


# =============================================================================
# Ejecucion directa
# =============================================================================
if __name__ == "__main__":
    resultado = analizar_sargazo_fotogrametria(
        ruta_imagen="prueba2.jpeg",
        altura_vuelo_m=50,
    )
