import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# =============================================================================
# CORRECCIONES AL PIPELINE ORIGINAL  (resumen de cambios)
# =============================================================================
# PROBLEMA 1 - Rango HSV demasiado amplio:
#   Original: inRange([5,50,20], [30,255,150]) → captura arena, techos, suelo seco
#   Solución: restricción adicional por índices espectrales (R-B, G-R) y
#             máscaras de exclusión por clase (agua, vegetación, arena, techos).
#
# PROBLEMA 2 - Un solo umbral HSV no discrimina bien en imágenes satelitales:
#   Solución: segmentación multi-condición (H + S + V + índices de color)
#             combinada con exclusión de clases de fondo.
#
# PROBLEMA 3 - Morfología insuficiente (solo mediana + cierre):
#   Original: medianBlur(5) + MORPH_CLOSE(5×5)
#   Solución: pipeline morfológico de 4 pasos ordenados:
#             mediana → apertura (elimina ruido) → cierre (conecta partes)
#             → dilatación/erosión balanceada (rellena huecos sin inflar).
#
# PROBLEMA 4 - Sin filtrado por área de contornos:
#   Solución: descarta blobs < área mínima, eliminando falsos positivos aislados.
# =============================================================================


def analizar_sargazo_fotogrametria(ruta_imagen, altura_vuelo_m, fov_grados=82.0):
    """
    Segmenta sargazo costero y calcula logística de recolección usando
    principios de fotogrametría (GSD) y un pipeline PDI multi-etapa mejorado.

    Parámetros
    ----------
    ruta_imagen   : str   Ruta a la imagen aérea/satelital.
    altura_vuelo_m: float Altitud de toma en metros.
    fov_grados    : float Campo de visión horizontal de la cámara (por defecto 82°).

    Retorna
    -------
    dict  con claves: gsd_m, area_m2, camiones
    """
    print("\n" + "="*60)
    print("  ANÁLISIS FOTOGRAMÉTRICO DE SARGAZO")
    print("="*60)
    print(f"  Altitud de toma   : {altura_vuelo_m} m")
    print(f"  FOV horizontal    : {fov_grados}°")

    # ------------------------------------------------------------------
    # 1. CARGA Y VALIDACIÓN DE IMAGEN
    # ------------------------------------------------------------------
    img_bgr = cv2.imread(ruta_imagen)
    if img_bgr is None:
        raise ValueError(f"No se pudo cargar la imagen: {ruta_imagen}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    alto_px, ancho_px = img_bgr.shape[:2]
    print(f"  Resolución imagen : {ancho_px} × {alto_px} px")

    # ------------------------------------------------------------------
    # 2. FOTOGRAMETRÍA — GSD (Ground Sample Distance)
    # ------------------------------------------------------------------
    fov_rad         = math.radians(fov_grados)
    ancho_fisico_m  = 2 * altura_vuelo_m * math.tan(fov_rad / 2)
    gsd_m_px        = ancho_fisico_m / ancho_px          # metros por píxel
    area_px_m2      = gsd_m_px ** 2                       # m² por píxel

    print(f"  GSD               : {gsd_m_px:.4f} m/px")
    print(f"  Área por píxel    : {area_px_m2:.6f} m²")

    # ------------------------------------------------------------------
    # 3. PREPROCESAMIENTO — CLAHE para realce de contraste local
    #    Mejora la discriminación entre sargazo y fondo en imágenes
    #    con iluminación desigual (tomadas a distintas horas del día).
    # ------------------------------------------------------------------
    gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe  = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)   # se usa solo para visualización diagnóstica

    # ------------------------------------------------------------------
    # 4. SEPARACIÓN DE CANALES EN MÚLTIPLES ESPACIOS DE COLOR
    # ------------------------------------------------------------------
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Canales HSV como arreglos float para cálculos aritméticos
    H = img_hsv[:, :, 0].astype(float)   # Tono       [0–179]
    S = img_hsv[:, :, 1].astype(float)   # Saturación [0–255]
    V = img_hsv[:, :, 2].astype(float)   # Valor      [0–255]

    # Canales BGR como float para índices espectrales
    R = img_bgr[:, :, 2].astype(float)
    G = img_bgr[:, :, 1].astype(float)
    B = img_bgr[:, :, 0].astype(float)

    # ------------------------------------------------------------------
    # 5. MÁSCARA POSITIVA — Candidatos a sargazo
    #    Criterios derivados del análisis espectral de la imagen:
    #    • H < 22 o H > 155 → tonos rojo-naranja-marrón
    #    • S ∈ [14, 159]   → saturación baja/media (sargazo seco no muy vívido)
    #    • V ∈ [18, 144]   → brillo bajo/medio (más oscuro que la arena)
    #    • R − B > 4       → componente rojiza siempre presente en marrón
    #    • G − R < 18      → excluye vegetación viva (clorofila: G >> R)
    # ------------------------------------------------------------------
    cond_H    = (H < 22) | (H > 155)
    cond_S    = (S > 14) & (S < 160)
    cond_V    = (V > 18) & (V < 145)
    cond_RB   = (R - B) > 4
    cond_noveg = (G - R) < 18

    mascara_positiva = (cond_H & cond_S & cond_V & cond_RB & cond_noveg
                        ).astype(np.uint8) * 255

    # ------------------------------------------------------------------
    # 6. MÁSCARAS DE EXCLUSIÓN POR CLASE
    #    Eliminan falsos positivos (FP) que también caen en el rango HSV.
    # ------------------------------------------------------------------

    # 6a. Agua / mar: tono azul-verdoso con saturación apreciable
    excl_agua = (H >= 70) & (H <= 135) & (S > 30)

    # 6b. Vegetación verde brillante: H verde, saturación alta, brillo alto
    excl_vegetacion = (H >= 32) & (H <= 82) & (S > 75) & (V > 85)

    # 6c. Arena / suelo claro: brillo muy alto con saturación mínima
    excl_arena = (V > 152) & (S < 45)

    # 6d. Techos y estructuras: tonos rojizos MUY saturados y brillantes
    excl_techos = (H < 20) & (S > 130) & (V > 120)

    excl_total = (excl_agua | excl_vegetacion | excl_arena | excl_techos
                  ).astype(np.uint8) * 255

    # Máscara de sargazo con FP descartados
    mascara_filtrada = cv2.bitwise_and(mascara_positiva,
                                       cv2.bitwise_not(excl_total))

    # ------------------------------------------------------------------
    # 7. MORFOLOGÍA — Pipeline de 4 pasos
    # ------------------------------------------------------------------

    # 7a. Filtro de mediana: elimina ruido sal-pimienta (píxeles aislados)
    m1 = cv2.medianBlur(mascara_filtrada, 5)

    # 7b. Apertura morfológica: elimina regiones pequeñas e irrelevantes
    k_apertura = np.ones((3, 3), np.uint8)
    m2 = cv2.morphologyEx(m1, cv2.MORPH_OPEN, k_apertura)

    # 7c. Cierre morfológico: une fragmentos de la banda de sargazo
    #     Kernel elíptico (más natural que cuadrado)
    k_cierre = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m3 = cv2.morphologyEx(m2, cv2.MORPH_CLOSE, k_cierre)

    # 7d. Dilatación + erosión balanceada: rellena huecos internos sin inflar
    k_balance = np.ones((5, 5), np.uint8)
    m4 = cv2.dilate(m3, k_balance, iterations=1)
    m5 = cv2.erode(m4,  k_balance, iterations=1)

    # ------------------------------------------------------------------
    # 8. FILTRADO POR ÁREA DE CONTORNOS
    #    Descarta blobs cuya área sea menor a un umbral mínimo.
    #    Calcula el GSD-equivalente en m² para convertir al mundo real.
    # ------------------------------------------------------------------
    AREA_MIN_PX = 100   # píxeles² mínimo por blob

    contornos, _ = cv2.findContours(m5, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    mascara_final = np.zeros_like(m5)
    blobs_validos = []

    for cnt in contornos:
        area_px = cv2.contourArea(cnt)
        if area_px >= AREA_MIN_PX:
            cv2.drawContours(mascara_final, [cnt], -1, 255, -1)
            blobs_validos.append((area_px, cnt))

    # Ordenar por área descendente (mayor primero)
    blobs_validos.sort(key=lambda x: -x[0])

    # ------------------------------------------------------------------
    # 9. CÁLCULO DE ÁREA TOTAL Y MODELO LOGÍSTICO
    # ------------------------------------------------------------------
    pixeles_sargazo   = cv2.countNonZero(mascara_final)
    area_total_m2     = pixeles_sargazo * area_px_m2

    # Estimación de volumen: altura media de 5 cm para sargazo depositado
    volumen_m3        = area_total_m2 * 0.05
    cap_camion_m3     = 14.0
    camiones_req      = int(np.ceil(volumen_m3 / cap_camion_m3))

    print(f"\n  --- RESULTADOS ---")
    print(f"  Blobs sargazo detectados : {len(blobs_validos)}")
    print(f"  Píxeles de sargazo       : {pixeles_sargazo:,}")
    print(f"  Área estimada            : {area_total_m2:,.2f} m²")
    print(f"  Volumen estimado         : {volumen_m3:,.2f} m³")
    print(f"  Camiones requeridos      : {camiones_req}")

    # ------------------------------------------------------------------
    # 10. DETECCIÓN DE CONTORNOS FINALES para visualización
    # ------------------------------------------------------------------
    img_contornos = img_rgb.copy()
    for _, cnt in blobs_validos:
        cv2.drawContours(img_contornos, [cnt], -1, (255, 80, 0), 2)

    # ------------------------------------------------------------------
    # 11. VISUALIZACIÓN — 6 paneles del pipeline
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(
        f"Análisis PDI de Sargazo — Toma a {altura_vuelo_m} m de altitud",
        fontsize=15, fontweight='bold')

    # Panel 1: Original
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(img_rgb)
    ax1.set_title("1. Imagen Original (RGB)", fontsize=11)
    ax1.axis('off')

    # Panel 2: CLAHE (realce de contraste)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(gray_eq, cmap='gray')
    ax2.set_title("2. Realce CLAHE\n(contraste local adaptativo)", fontsize=11)
    ax2.axis('off')

    # Panel 3: Máscara positiva inicial (antes de exclusiones)
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(mascara_positiva, cmap='gray')
    ax3.set_title("3. Máscara Inicial\n(HSV + índices R-B, G-R)", fontsize=11)
    ax3.axis('off')

    # Panel 4: Máscara tras exclusión
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(mascara_filtrada, cmap='gray')
    ax4.set_title("4. Tras Exclusión\n(descarte agua, veg., arena, techos)", fontsize=11)
    ax4.axis('off')

    # Panel 5: Máscara morfológica final
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(mascara_final, cmap='gray')
    ax5.set_title("5. Máscara Final\n(morfología + filtro área)", fontsize=11)
    ax5.axis('off')

    # Panel 6: Overlay + contornos + reporte
    ax6 = fig.add_subplot(2, 3, 6)
    overlay = img_rgb.copy()
    overlay[mascara_final > 0] = [220, 80, 20]
    blended = cv2.addWeighted(img_rgb, 0.55, overlay, 0.45, 0)
    ax6.imshow(blended)
    for _, cnt in blobs_validos:
        cnt_squeezed = cnt.squeeze()
        if cnt_squeezed.ndim == 2:
            ax6.plot(np.append(cnt_squeezed[:, 0], cnt_squeezed[0, 0]),
                     np.append(cnt_squeezed[:, 1], cnt_squeezed[0, 1]),
                     color='yellow', linewidth=1.2, alpha=0.85)
    ax6.set_title("6. Sargazo Segmentado\n(overlay naranja + contornos)", fontsize=11)
    ax6.axis('off')

    # ------------------------------------------------------------------
    # 12. REPORTE TEXTUAL en figura separada
    # ------------------------------------------------------------------
    fig2, ax_rep = plt.subplots(figsize=(9, 5))
    fig2.suptitle("Reporte Geométrico y Logístico", fontsize=13, fontweight='bold')
    ax_rep.axis('off')
    texto = (
        f"FOTOGRAMETRÍA\n"
        f"{'─'*38}\n"
        f"  Campo de visión (FOV)      : {fov_grados}°\n"
        f"  GSD (tamaño de píxel)      : {gsd_m_px:.4f} m/px\n"
        f"  Área por píxel             : {area_px_m2:.6f} m²\n"
        f"  Cobertura imagen           : {ancho_px} × {alto_px} px\n\n"
        f"SEGMENTACIÓN PDI\n"
        f"{'─'*38}\n"
        f"  Blobs detectados           : {len(blobs_validos)}\n"
        f"  Píxeles de sargazo         : {pixeles_sargazo:,} px\n"
        f"  Área real estimada         : {area_total_m2:,.2f} m²\n\n"
        f"LOGÍSTICA DE RECOLECCIÓN\n"
        f"{'─'*38}\n"
        f"  Espesor biomasa asumido    : 0.05 m\n"
        f"  Volumen de biomasa         : {volumen_m3:,.2f} m³\n"
        f"  Capacidad camión           : {cap_camion_m3:.1f} m³\n"
        f"  🚛 Camiones requeridos     : {camiones_req}\n"
    )
    ax_rep.text(0.04, 0.97, texto,
                transform=ax_rep.transAxes,
                fontsize=11, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(facecolor='#f7f7f7', alpha=0.9,
                          boxstyle='round,pad=0.8', edgecolor='#cccccc'))

    plt.tight_layout()
    plt.show()

    return {
        "gsd_m"   : gsd_m_px,
        "area_m2" : area_total_m2,
        "camiones": camiones_req,
    }


# =============================================================================
# EJECUCIÓN
# =============================================================================
if __name__ == "__main__":
    # Cambia la ruta y la altitud según tu imagen
    resultado = analizar_sargazo_fotogrametria(
        ruta_imagen    = "prueba2.jpeg",
        altura_vuelo_m = 50
    )