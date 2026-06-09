import React from "react";
import { createRoot } from "react-dom/client";
import { Activity, Download, FileImage, FileText, ImagePlus, Loader2, Truck } from "lucide-react";
import "./styles.css";

const API_HOST = window.location.hostname === "localhost" ? "localhost" : "127.0.0.1";
const API_BASE = `http://${API_HOST}:5000`;

function Metric({ label, value, unit }) {
  return (
    <section className="metric" aria-label={label}>
      <span>{label}</span>
      <strong>{value}</strong>
      {unit ? <small>{unit}</small> : null}
    </section>
  );
}

function Field({ label, value, onChange, suffix, min, max, step }) {
  return (
    <label className="field">
      <span>{label}</span>
      <div className="field-control">
        <input
          type="number"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(event) => onChange(event.target.value)}
        />
        <small>{suffix}</small>
      </div>
    </label>
  );
}

function App() {
  const [file, setFile] = React.useState(null);
  const [previewUrl, setPreviewUrl] = React.useState("");
  const [altura, setAltura] = React.useState("50");
  const [fov, setFov] = React.useState("82");
  const [espesor, setEspesor] = React.useState("0.05");
  const [capacidad, setCapacidad] = React.useState("14");
  const [result, setResult] = React.useState(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState("");

  React.useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const handleFileChange = (event) => {
    const nextFile = event.target.files?.[0];
    if (!nextFile) return;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(nextFile);
    setPreviewUrl(URL.createObjectURL(nextFile));
    setResult(null);
    setError("");
  };

  const analyze = async (event) => {
    event.preventDefault();
    if (!file) {
      setError("Selecciona una imagen antes de analizar.");
      return;
    }

    const formData = new FormData();
    formData.append("image", file);
    formData.append("altura", altura);
    formData.append("fov", fov);
    formData.append("espesor", espesor);
    formData.append("capacidad", capacidad);

    setLoading(true);
    setError("");
    try {
      const response = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "No se pudo analizar la imagen.");
      }
      setResult(data);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  };

  const metrics = result?.metrics;

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Analisis costero</p>
          <h1>Estimador de sargazo</h1>
        </div>
        <div className="status-pill">
          <Activity size={18} aria-hidden="true" />
          <span>{loading ? "Procesando" : result ? "Listo" : "Sin analizar"}</span>
        </div>
      </header>

      <div className="workspace">
        <form className="control-panel" onSubmit={analyze}>
          <section className="panel-section">
            <h2>Imagen</h2>
            <label className="upload-zone">
              <input type="file" accept="image/*" onChange={handleFileChange} />
              <ImagePlus size={28} aria-hidden="true" />
              <span>{file ? file.name : "Seleccionar imagen"}</span>
            </label>
          </section>

          <section className="panel-section">
            <h2>Toma</h2>
            <Field label="Altitud" value={altura} onChange={setAltura} suffix="m" min="1" step="1" />
            <Field label="FOV" value={fov} onChange={setFov} suffix="grados" min="10" max="170" step="0.1" />
          </section>

          <section className="panel-section">
            <h2>Recoleccion</h2>
            <Field label="Espesor" value={espesor} onChange={setEspesor} suffix="m" min="0.001" step="0.01" />
            <Field label="Capacidad camion" value={capacidad} onChange={setCapacidad} suffix="m3" min="0.1" step="0.5" />
          </section>

          <button className="primary-button" type="submit" disabled={loading}>
            {loading ? <Loader2 className="spin" size={20} aria-hidden="true" /> : <FileImage size={20} aria-hidden="true" />}
            <span>{loading ? "Analizando" : "Analizar imagen"}</span>
          </button>

          {error ? <p className="error-message" role="alert">{error}</p> : null}
        </form>

        <section className="results-panel">
          <div className="metrics-grid">
            <Metric label="Area" value={metrics ? formatNumber(metrics.area_m2, 2) : "--"} unit="m2" />
            <Metric label="Volumen" value={metrics ? formatNumber(metrics.volumen_m3, 2) : "--"} unit="m3" />
            <Metric label="Camiones" value={metrics ? metrics.camiones : "--"} unit="unidades" />
            <Metric label="Pixeles" value={metrics ? formatInteger(metrics.pixeles_sargazo) : "--"} unit="px" />
          </div>

          <div className="image-grid">
            <ImagePreview title="Original" url={previewUrl} empty="Selecciona una imagen" />
            <ImagePreview
              title="Segmentado"
              url={result ? `${API_BASE}${result.images.overlay}` : ""}
              empty={loading ? "Analizando" : "Sin resultado"}
            />
          </div>

          <footer className="result-actions">
            <div className="summary">
              <Truck size={22} aria-hidden="true" />
              <p>
                {metrics
                  ? `${metrics.blobs} zonas detectadas con GSD de ${formatNumber(metrics.gsd_m, 4)} m/px.`
                  : "Los resultados apareceran aqui despues del analisis."}
              </p>
            </div>
            <div className="download-row">
              <DownloadLink href={result?.images.overlay} icon={<Download size={18} />} label="Imagen" />
              <DownloadLink href={result?.images.report} icon={<FileText size={18} />} label="Reporte" />
              <DownloadLink href={result?.images.diagnostic} icon={<FileImage size={18} />} label="Diagnostico" />
            </div>
          </footer>
        </section>
      </div>
    </main>
  );
}

function ImagePreview({ title, url, empty }) {
  return (
    <section className="image-panel">
      <h2>{title}</h2>
      <div className="image-frame">
        {url ? <img src={url} alt={title} /> : <span>{empty}</span>}
      </div>
    </section>
  );
}

function DownloadLink({ href, icon, label }) {
  const disabled = !href;
  return (
    <a
      className={disabled ? "download disabled" : "download"}
      href={disabled ? undefined : `${API_BASE}${href}`}
      target="_blank"
      rel="noreferrer"
      aria-disabled={disabled}
      tabIndex={disabled ? -1 : 0}
    >
      {React.cloneElement(icon, { "aria-hidden": true })}
      <span>{label}</span>
    </a>
  );
}

function formatNumber(value, digits) {
  return Number(value).toLocaleString("es-MX", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function formatInteger(value) {
  return Number(value).toLocaleString("es-MX");
}

createRoot(document.getElementById("root")).render(<App />);
