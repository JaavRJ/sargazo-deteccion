import os
import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from actividad_final import analizar_sargazo_fotogrametria


BASE_DIR = Path(__file__).resolve().parent
DIST_DIR = BASE_DIR / "dist"
OUTPUT_DIR = BASE_DIR / "salidas" / "web"
UPLOAD_DIR = OUTPUT_DIR / "uploads"
RESULT_DIR = OUTPUT_DIR / "resultados"

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024

ALLOWED_ORIGINS = {
    "http://127.0.0.1:5173",
    "http://localhost:5173",
}


@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin")
    if origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.get("/api/health")
def health():
    return jsonify({"ok": True})


@app.post("/api/analyze")
def analyze():
    image = request.files.get("image")
    if image is None or not image.filename:
        return jsonify({"error": "Selecciona una imagen valida."}), 400

    try:
        params = {
            "altura": _read_float("altura", min_value=1),
            "fov": _read_float("fov", min_value=10, max_value=170),
            "espesor": _read_float("espesor", min_value=0.001, max_value=5),
            "capacidad": _read_float("capacidad", min_value=0.1),
        }
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    suffix = Path(image.filename).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
        return jsonify({"error": "Formato no compatible. Usa JPG, PNG, BMP o TIFF."}), 400

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    safe_stem = _safe_stem(Path(image.filename).stem)
    run_id = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    uploaded_path = UPLOAD_DIR / f"{safe_stem}_{run_id}{suffix}"
    diagnostic_path = RESULT_DIR / f"{safe_stem}_{run_id}_diagnostico.png"
    overlay_path = RESULT_DIR / f"{safe_stem}_{run_id}_segmentado.png"
    report_path = RESULT_DIR / f"{safe_stem}_{run_id}_reporte.txt"

    image.save(uploaded_path)

    try:
        result = analizar_sargazo_fotogrametria(
            ruta_imagen=str(uploaded_path),
            altura_vuelo_m=params["altura"],
            fov_grados=params["fov"],
            espesor_biomasa_m=params["espesor"],
            capacidad_camion_m3=params["capacidad"],
            mostrar=False,
            guardar_figura=str(diagnostic_path),
            devolver_mascaras=True,
        )
        mask = result["mascaras"]["final"]
        _crear_overlay(uploaded_path, mask, overlay_path)
        result.pop("mascaras", None)
        report_path.write_text(_report_text(image.filename, result), encoding="utf-8")
    except Exception as exc:
        return jsonify({"error": f"No se pudo analizar la imagen: {exc}"}), 500

    return jsonify(
        {
            "metrics": {
                "area_m2": result["area_m2"],
                "volumen_m3": result["volumen_m3"],
                "camiones": result["camiones"],
                "pixeles_sargazo": result["pixeles_sargazo"],
                "blobs": result["blobs"],
                "gsd_m": result["gsd_m"],
                "espesor_biomasa_m": result["espesor_biomasa_m"],
                "capacidad_camion_m3": result["capacidad_camion_m3"],
            },
            "images": {
                "original": _output_url(uploaded_path),
                "overlay": _output_url(overlay_path),
                "diagnostic": _output_url(diagnostic_path),
                "report": _output_url(report_path),
            },
        }
    )


@app.route("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory(OUTPUT_DIR, filename)


@app.get("/")
def serve_app():
    return _serve_frontend("index.html")


@app.get("/<path:filename>")
def serve_frontend_asset(filename):
    return _serve_frontend(filename)


def _read_float(name, min_value=None, max_value=None):
    raw = request.form.get(name, "").strip().replace(",", ".")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name}: escribe un numero valido.") from exc

    if min_value is not None and value < min_value:
        raise ValueError(f"{name}: usa un valor mayor o igual a {min_value}.")
    if max_value is not None and value > max_value:
        raise ValueError(f"{name}: usa un valor menor o igual a {max_value}.")
    return value


def _safe_stem(value):
    value = re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-")
    return value or "imagen"


def _output_url(path):
    relative = path.relative_to(OUTPUT_DIR).as_posix()
    return f"/outputs/{relative}"


def _serve_frontend(filename):
    asset_path = DIST_DIR / filename
    if asset_path.is_file():
        return send_from_directory(DIST_DIR, filename)
    return send_from_directory(DIST_DIR, "index.html")


def _crear_overlay(image_path, mask, output_path):
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(f"No se pudo cargar la imagen: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    overlay = img_rgb.copy()
    overlay[mask > 0] = [216, 107, 46]
    blended = cv2.addWeighted(img_rgb, 0.62, overlay, 0.38, 0)

    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contornos, -1, (246, 205, 53), 2)
    cv2.imwrite(str(output_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))


def _report_text(image_name, result):
    return (
        "Reporte zargx\n"
        "=============\n\n"
        f"Imagen: {image_name}\n"
        f"Fecha: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n"
        "Resultados\n"
        "----------\n"
        f"Area estimada: {result['area_m2']:,.2f} m2\n"
        f"Volumen estimado: {result['volumen_m3']:,.2f} m3\n"
        f"Camiones requeridos: {result['camiones']}\n"
        f"Pixeles detectados: {result['pixeles_sargazo']:,} px\n"
        f"Zonas detectadas: {result['blobs']}\n\n"
        "Parametros\n"
        "----------\n"
        f"GSD: {result['gsd_m']:.4f} m/px\n"
        f"Espesor asumido: {result['espesor_biomasa_m']:.3f} m\n"
        f"Capacidad por camion: {result['capacidad_camion_m3']:.1f} m3\n"
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
