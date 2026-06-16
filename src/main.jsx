import React from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  BarChart3,
  FileImage,
  FileText,
  Grid3X3,
  ImagePlus,
  Layers,
  Loader2,
  Ruler,
  Truck,
  Waves,
} from "lucide-react";
import "./styles.css";

const IS_VITE_DEV = ["localhost", "127.0.0.1"].includes(window.location.hostname) && window.location.port === "5173";
const API_BASE = IS_VITE_DEV ? `http://${window.location.hostname}:5000` : "";
const DEFAULT_ESPESOR = "0.05";
const DEFAULT_CAPACIDAD = "14";

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
  const [result, setResult] = React.useState(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState("");
  const [activeArtifact, setActiveArtifact] = React.useState("overlay");
  const [reportText, setReportText] = React.useState("");

  React.useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  React.useEffect(() => {
    if (!result?.images.report) {
      setReportText("");
      return;
    }

    fetch(`${API_BASE}${result.images.report}`)
      .then((response) => response.text())
      .then(setReportText)
      .catch(() => setReportText("No se pudo cargar el reporte."));
  }, [result]);

  const handleFileChange = (event) => {
    const nextFile = event.target.files?.[0];
    if (!nextFile) return;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(nextFile);
    setPreviewUrl(URL.createObjectURL(nextFile));
    setResult(null);
    setReportText("");
    setActiveArtifact("overlay");
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
    formData.append("espesor", DEFAULT_ESPESOR);
    formData.append("capacidad", DEFAULT_CAPACIDAD);

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
      setActiveArtifact("overlay");
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
        <div className="brand-lockup">
          <img className="brand-icon" src="/zargx-icon.svg" alt="" />
          <div>
            <p className="eyebrow">Analisis costero de sargazo</p>
            <h1>zargx</h1>
          </div>
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
              {previewUrl ? (
                <span className="upload-preview">
                  <img src={previewUrl} alt="Vista previa de la imagen seleccionada" />
                  <small>{file.name}</small>
                </span>
              ) : (
                <>
                  <ImagePlus size={28} aria-hidden="true" />
                  <span>Seleccionar imagen</span>
                </>
              )}
            </label>
          </section>

          <section className="panel-section">
            <h2>Toma</h2>
            <Field label="Altitud" value={altura} onChange={setAltura} suffix="m" min="1" step="1" />
            <Field label="FOV" value={fov} onChange={setFov} suffix="grados" min="10" max="170" step="0.1" />
          </section>

          <section className="auto-panel">
            <div className="auto-icon">
              <Truck size={20} aria-hidden="true" />
            </div>
            <div>
              <h2>Recoleccion automatica</h2>
              <p>{DEFAULT_ESPESOR} m de espesor y {DEFAULT_CAPACIDAD} m3 por camion.</p>
            </div>
          </section>

          <button className="primary-button" type="submit" disabled={loading}>
            {loading ? <Loader2 className="spin" size={20} aria-hidden="true" /> : <FileImage size={20} aria-hidden="true" />}
            <span>{loading ? "Analizando" : "Analizar imagen"}</span>
          </button>

          {error ? <p className="error-message" role="alert">{error}</p> : null}
        </form>

        <section className="results-panel">
          <div className="metrics-grid">
            <VisualMetric icon={<Ruler />} label="Area" value={metrics ? formatNumber(metrics.area_m2, 2) : "--"} unit="m2" tone="area" />
            <VisualMetric icon={<Waves />} label="Volumen" value={metrics ? formatNumber(metrics.volumen_m3, 2) : "--"} unit="m3" tone="volume" />
            <TruckMetric value={metrics?.camiones} />
            <VisualMetric icon={<Grid3X3 />} label="Pixeles" value={metrics ? formatInteger(metrics.pixeles_sargazo) : "--"} unit="px" tone="pixels" />
          </div>

          <div className="image-grid">
            <ImagePreview title="Original" url={previewUrl} empty="Selecciona una imagen" />
            <ImagePreview
              title="Segmentado"
              url={result ? `${API_BASE}${result.images.overlay}` : ""}
              empty={loading ? "Analizando" : "Sin resultado"}
              featured
            />
          </div>

          <footer className="result-actions">
            <div className="summary">
              <BarChart3 size={22} aria-hidden="true" />
              <p>
                {metrics
                  ? `${metrics.blobs} zonas detectadas con GSD de ${formatNumber(metrics.gsd_m, 4)} m/px.`
                  : "Los resultados apareceran aqui despues del analisis."}
              </p>
            </div>
            <div className="artifact-tabs" aria-label="Ver resultados">
              <ArtifactButton active={activeArtifact === "overlay"} onClick={() => setActiveArtifact("overlay")} icon={<Layers />} label="Segmentado" />
              <ArtifactButton active={activeArtifact === "diagnostic"} onClick={() => setActiveArtifact("diagnostic")} icon={<FileImage />} label="Diagnostico" />
              <ArtifactButton active={activeArtifact === "report"} onClick={() => setActiveArtifact("report")} icon={<FileText />} label="Reporte" />
            </div>
          </footer>

          <ArtifactViewer active={activeArtifact} result={result} reportText={reportText} loading={loading} />
        </section>
      </div>
    </main>
  );
}

function VisualMetric({ icon, label, value, unit, tone }) {
  return (
    <section className={`metric visual-metric ${tone}`} aria-label={label}>
      <div className="metric-topline">
        <span className="metric-icon">{React.cloneElement(icon, { size: 20, "aria-hidden": true })}</span>
        <span>{label}</span>
      </div>
      <strong>{value}</strong>
      <small>{unit}</small>
      <span className="metric-bar" aria-hidden="true" />
    </section>
  );
}

function TruckMetric({ value }) {
  const truckCount = Number(value || 0);
  const visibleTrucks = Math.min(Math.max(truckCount, 1), 6);

  return (
    <section className="metric truck-metric" aria-label="Camiones">
      <div className="metric-topline">
        <span className="metric-icon">
          <Truck size={20} aria-hidden="true" />
        </span>
        <span>Camiones</span>
      </div>
      <div className="truck-fleet" aria-hidden="true">
        {Array.from({ length: visibleTrucks }).map((_, index) => (
          <Truck key={index} className={index < truckCount ? "filled" : ""} size={25} />
        ))}
      </div>
      <strong>{value ?? "--"}</strong>
      <small>unidades</small>
    </section>
  );
}

function ImagePreview({ title, url, empty, featured = false }) {
  return (
    <section className={featured ? "image-panel featured" : "image-panel"}>
      <h2>{title}</h2>
      <div className="image-frame">
        {url ? <img src={url} alt={title} /> : <span>{empty}</span>}
      </div>
    </section>
  );
}

function ArtifactButton({ active, onClick, icon, label }) {
  return (
    <button className={active ? "artifact-button active" : "artifact-button"} type="button" onClick={onClick}>
      {React.cloneElement(icon, { size: 18, "aria-hidden": true })}
      <span>{label}</span>
    </button>
  );
}

function ArtifactViewer({ active, result, reportText, loading }) {
  const artifacts = {
    overlay: {
      title: "Imagen segmentada",
      url: result ? `${API_BASE}${result.images.overlay}` : "",
      empty: loading ? "Analizando imagen" : "Ejecuta el analisis para ver la segmentacion.",
    },
    diagnostic: {
      title: "Diagnostico del procesamiento",
      url: result ? `${API_BASE}${result.images.diagnostic}` : "",
      empty: loading ? "Generando diagnostico" : "Ejecuta el analisis para ver el proceso completo.",
    },
  };

  if (active === "report") {
    return (
      <section className="artifact-viewer">
        <h2>Reporte logistico</h2>
        <div className="report-frame">
          {reportText ? <pre>{reportText}</pre> : <span>Ejecuta el analisis para ver el reporte.</span>}
        </div>
      </section>
    );
  }

  const artifact = artifacts[active] || artifacts.overlay;

  return (
    <section className="artifact-viewer">
      <h2>{artifact.title}</h2>
      <div className="artifact-frame">
        {artifact.url ? <img src={artifact.url} alt={artifact.title} /> : <span>{artifact.empty}</span>}
      </div>
    </section>
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
