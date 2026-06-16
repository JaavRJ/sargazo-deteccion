from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse
import io
import json
import mimetypes
import re
import uuid

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from actividad_final import analizar_sargazo_riviera


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "salidas" / "web"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
PORT = 8000


def send_json(handler, payload, status=200):
    data = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    add_cors(handler)
    handler.end_headers()
    handler.wfile.write(data)


def add_cors(handler):
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")


def clean_filename(name):
    stem = Path(name or "imagen").stem
    suffix = Path(name or ".png").suffix.lower() or ".png"
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._") or "imagen"
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
      suffix = ".png"
    return f"{stem}{suffix}"


def parse_multipart(content_type, body):
    match = re.search(r"boundary=(?P<boundary>[^;]+)", content_type)
    if not match:
        raise ValueError("Solicitud multipart sin boundary.")

    boundary = match.group("boundary").strip('"').encode("utf-8")
    fields = {}
    files = {}

    for part in body.split(b"--" + boundary):
        part = part.strip()
        if not part or part == b"--":
            continue
        if part.endswith(b"--"):
            part = part[:-2].strip()
        if b"\r\n\r\n" not in part:
            continue

        raw_headers, raw_value = part.split(b"\r\n\r\n", 1)
        value = raw_value.rstrip(b"\r\n")
        headers = raw_headers.decode("utf-8", errors="ignore")
        disposition = next(
            (line for line in headers.split("\r\n") if line.lower().startswith("content-disposition")),
            "",
        )
        name_match = re.search(r'name="([^"]+)"', disposition)
        if not name_match:
            continue

        name = name_match.group(1)
        filename_match = re.search(r'filename="([^"]*)"', disposition)
        if filename_match:
            files[name] = {
                "filename": filename_match.group(1),
                "content": value,
            }
        else:
            fields[name] = value.decode("utf-8", errors="ignore")

    return fields, files


def output_url(path):
    return f"/outputs/{quote(path.name)}"


def save_diagnostic_steps(fig, job_id):
    titles = [
        "Imagen original RGB",
        "Mascara H",
        "Mascara S",
        "Mascara V",
        "Mascara final",
        "Overlay sargazo",
    ]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    paths = []

    for index, ax in enumerate(fig.axes[:6], start=1):
        bbox = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
        step_path = OUTPUT_DIR / f"{job_id}_paso_{index}.png"
        fig.savefig(
            step_path,
            dpi=180,
            bbox_inches=bbox.padded(0.12),
            facecolor=fig.get_facecolor(),
        )
        paths.append(
            {
                "title": titles[index - 1],
                "image": output_url(step_path),
            }
        )

    return paths


class SargazoHandler(BaseHTTPRequestHandler):
    server_version = "SargazoPDI/0.1"

    def do_OPTIONS(self):
        self.send_response(204)
        add_cors(self)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            send_json(self, {"ok": True})
            return

        if parsed.path.startswith("/outputs/"):
            self.serve_output(parsed.path.removeprefix("/outputs/"))
            return

        send_json(self, {"error": "Ruta no encontrada."}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/analyze":
            send_json(self, {"error": "Ruta no encontrada."}, 404)
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0 or content_length > MAX_UPLOAD_BYTES:
            send_json(self, {"error": "La imagen es demasiado grande o esta vacia."}, 413)
            return

        try:
            content_type = self.headers.get("Content-Type", "")
            body = self.rfile.read(content_length)
            fields, files = parse_multipart(content_type, body)
            image = files.get("image")
            if not image or not image["content"]:
                send_json(self, {"error": "Sube una imagen para analizar."}, 400)
                return

            altitude = float(fields.get("altitude", 50))
            fov = float(fields.get("fov", 82))
            if altitude <= 0 or fov <= 0:
                raise ValueError("Altitud y FOV deben ser mayores que cero.")

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            job_id = uuid.uuid4().hex[:10]
            safe_name = clean_filename(image["filename"])
            upload_path = OUTPUT_DIR / f"{job_id}_original_{safe_name}"
            upload_path.write_bytes(image["content"])

            plt.close("all")
            logs = io.StringIO()
            with redirect_stdout(logs):
                metrics = analizar_sargazo_riviera(str(upload_path), altitude, fov)

            figure_paths = []
            diagnostic_steps = []
            for index, number in enumerate(plt.get_fignums(), start=1):
                fig = plt.figure(number)
                name = "diagnostico" if index == 1 else "reporte"
                if index == 1:
                    diagnostic_steps = save_diagnostic_steps(fig, job_id)
                fig_path = OUTPUT_DIR / f"{job_id}_{name}.png"
                fig.savefig(fig_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
                figure_paths.append(fig_path)
            plt.close("all")

            send_json(
                self,
                {
                    "metrics": metrics,
                    "images": {
                        "original": output_url(upload_path),
                        "diagnostic": output_url(figure_paths[0]) if figure_paths else None,
                        "report": output_url(figure_paths[1]) if len(figure_paths) > 1 else None,
                    },
                    "diagnosticSteps": diagnostic_steps,
                    "steps": [
                        "Carga y validacion de imagen",
                        "Calculo fotogrametrico GSD",
                        "Filtro bilateral y realce black-hat",
                        "Mascaras HSV por H, S y V",
                        "Refuerzo de candidatos y exclusion de agua",
                        "Morfologia: mediana, apertura y cierre",
                        "Overlay final y reporte logistico",
                    ],
                    "log": logs.getvalue(),
                },
            )
        except Exception as exc:
            plt.close("all")
            send_json(self, {"error": str(exc)}, 500)

    def serve_output(self, raw_name):
        name = Path(unquote(raw_name)).name
        path = OUTPUT_DIR / name
        if not path.exists() or not path.is_file():
            self.send_response(404)
            add_cors(self)
            self.end_headers()
            return

        content = path.read_bytes()
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(len(content)))
        add_cors(self)
        self.end_headers()
        self.wfile.write(content)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", PORT), SargazoHandler)
    print(f"API de sargazo lista en http://127.0.0.1:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
