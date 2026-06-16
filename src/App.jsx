import { useMemo, useState } from 'react';
import {
  Activity,
  ArrowLeft,
  ArrowRight,
  CheckCircle2,
  FileImage,
  ImagePlus,
  Loader2,
  MapPinned,
  Ruler,
  Truck,
  Upload,
} from 'lucide-react';

const formatter = new Intl.NumberFormat('es-MX', {
  maximumFractionDigits: 2,
  minimumFractionDigits: 0,
});

const API_BASE = import.meta.env.PROD ? 'http://127.0.0.1:8000' : '';

function formatMetric(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '--';
  return new Intl.NumberFormat('es-MX', {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(Number(value));
}

function assetUrl(path) {
  if (!path) return '';
  return path.startsWith('http') ? path : `${API_BASE}${path}`;
}

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [altitude, setAltitude] = useState(50);
  const [fov, setFov] = useState(82);
  const [status, setStatus] = useState('idle');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [activeView, setActiveView] = useState('original');
  const [diagnosticIndex, setDiagnosticIndex] = useState(0);

  const statusLabel = useMemo(() => {
    if (status === 'loading') return 'Procesando';
    if (status === 'success') return 'Listo';
    if (status === 'error') return 'Error';
    return 'Sin analizar';
  }, [status]);

  const handleFile = (event) => {
    const nextFile = event.target.files?.[0];
    if (!nextFile) return;
    setFile(nextFile);
    setPreview(URL.createObjectURL(nextFile));
    setResult(null);
    setError('');
    setStatus('idle');
    setActiveView('original');
    setDiagnosticIndex(0);
  };

  const analyzeImage = async () => {
    if (!file) {
      setError('Sube una imagen antes de analizar.');
      setStatus('error');
      return;
    }

    const form = new FormData();
    form.append('image', file);
    form.append('altitude', String(altitude));
    form.append('fov', String(fov));

    setStatus('loading');
    setError('');

    try {
      const response = await fetch(`${API_BASE}/api/analyze`, {
        method: 'POST',
        body: form,
      });
      const payload = await response.json();

      if (!response.ok) {
        throw new Error(payload.error || 'No se pudo procesar la imagen.');
      }

      setResult(payload);
      setStatus('success');
      setActiveView('diagnostic');
      setDiagnosticIndex(0);
    } catch (nextError) {
      setError(nextError.message);
      setStatus('error');
    }
  };

  const metrics = result?.metrics;
  const views = ['original', 'diagnostic', 'report', 'steps'];
  const activeIndex = views.indexOf(activeView);
  const goToView = (direction) => {
    if (activeView === 'diagnostic' && result?.diagnosticSteps?.length) {
      const totalSteps = result.diagnosticSteps.length;
      setDiagnosticIndex((current) => (current + direction + totalSteps) % totalSteps);
      return;
    }

    const nextIndex = (activeIndex + direction + views.length) % views.length;
    setActiveView(views[nextIndex]);
  };
  const diagnosticStepCount = result?.diagnosticSteps?.length || 6;

  return (
    <main className="app-shell">
      <header className="app-header">
        <div>
          <span className="eyebrow">Analisis costero</span>
          <h1>Estimador de sargazo</h1>
        </div>
        <div className={`status-pill ${status}`}>
          {status === 'loading' ? (
            <Loader2 size={16} aria-hidden="true" />
          ) : (
            <CheckCircle2 size={16} aria-hidden="true" />
          )}
          {statusLabel}
        </div>
      </header>

      <section className="tool-layout">
        <aside className="input-panel">
          <section className="panel-block">
            <div className="panel-title">
              <FileImage size={17} aria-hidden="true" />
              Imagen
            </div>

            <label className={`drop-zone ${preview ? 'has-image' : ''}`}>
              {preview ? (
                <img src={preview} alt="Imagen seleccionada para analizar" />
              ) : (
                <span>
                  <ImagePlus size={30} aria-hidden="true" />
                  Subir imagen
                </span>
              )}
              <input type="file" accept="image/*" onChange={handleFile} />
            </label>

            {file && <p className="file-name">{file.name}</p>}
          </section>

          <section className="panel-block">
            <div className="panel-title">
              <MapPinned size={17} aria-hidden="true" />
              Toma
            </div>
            <NumberField
              icon={<MapPinned size={15} aria-hidden="true" />}
              label="Altitud"
              suffix="m"
              min="1"
              value={altitude}
              onChange={setAltitude}
            />
            <NumberField
              icon={<Ruler size={15} aria-hidden="true" />}
              label="FOV"
              suffix="grados"
              min="1"
              value={fov}
              onChange={setFov}
            />
          </section>

          <button className="analyze-button" type="button" onClick={analyzeImage} disabled={status === 'loading'}>
            {status === 'loading' ? <Loader2 size={18} aria-hidden="true" /> : <Upload size={18} aria-hidden="true" />}
            Analizar imagen
          </button>

          {error && <p className="error-message">{error}</p>}
        </aside>

        <section className="work-area">
          <div className="metrics-grid" aria-label="Resultados principales">
            <Metric title="Area" value={formatMetric(metrics?.area_m2, 2)} unit="m2" />
            <Metric title="Volumen" value={formatMetric(metrics?.volumen_m3, 2)} unit="m3" />
            <Metric title="Camiones" value={metrics ? formatter.format(metrics.camiones) : '--'} unit="unidades" />
            <Metric title="Pixeles" value={metrics ? formatter.format(metrics.pixeles_sargazo) : '--'} unit="px" />
          </div>

          <article className="image-card main-viewer">
            <div className="viewer-toolbar">
              <div>
                <span className="step-kicker">
                  {activeView === 'diagnostic'
                    ? `Diagnostico ${Math.min(diagnosticIndex + 1, diagnosticStepCount)} de ${diagnosticStepCount}`
                    : `Vista ${activeIndex + 1} de ${views.length}`}
                </span>
                <h2>{getViewTitle(activeView, result?.diagnosticSteps?.[diagnosticIndex])}</h2>
              </div>
              <div className="tab-group" aria-label="Vista de procesamiento">
                <button className={activeView === 'original' ? 'active' : ''} type="button" onClick={() => setActiveView('original')}>
                  Original
                </button>
                <button className={activeView === 'diagnostic' ? 'active' : ''} type="button" onClick={() => setActiveView('diagnostic')}>
                  Diagnostico
                </button>
                <button className={activeView === 'report' ? 'active' : ''} type="button" onClick={() => setActiveView('report')}>
                  Reporte
                </button>
                <button className={activeView === 'steps' ? 'active' : ''} type="button" onClick={() => setActiveView('steps')}>
                  Pasos
                </button>
              </div>
            </div>

            <AnalysisView
              activeView={activeView}
              result={result}
              metrics={metrics}
              preview={preview}
              diagnosticIndex={diagnosticIndex}
            />

            <div className="viewer-actions">
              <button type="button" onClick={() => goToView(-1)}>
                <ArrowLeft size={18} aria-hidden="true" />
                Anterior
              </button>
              <button type="button" onClick={() => goToView(1)}>
                Siguiente
                <ArrowRight size={18} aria-hidden="true" />
              </button>
            </div>
          </article>
        </section>
      </section>
    </main>
  );
}

function NumberField({ icon, label, suffix, value, onChange, ...props }) {
  return (
    <label className="number-field">
      <span>
        {icon}
        {label}
      </span>
      <div>
        <input
          type="number"
          value={value}
          onChange={(event) => onChange(Number(event.target.value))}
          {...props}
        />
        <small>{suffix}</small>
      </div>
    </label>
  );
}

function Metric({ title, value, unit }) {
  return (
    <article className="metric-card">
      <span>{title}</span>
      <strong>{value}</strong>
      <small>{unit}</small>
    </article>
  );
}

function getViewTitle(view, diagnosticStep) {
  const titles = {
    original: 'Imagen original',
    diagnostic: diagnosticStep?.title || 'Diagnostico del procesamiento',
    report: 'Reporte logistico',
    steps: 'Pasos del algoritmo',
  };
  return titles[view];
}

function AnalysisView({ activeView, result, metrics, preview, diagnosticIndex }) {
  if (activeView === 'steps') {
    return (
      <div className="steps-stage">
        <div className="panel-title">
          <Activity size={17} aria-hidden="true" />
          Flujo PDI
        </div>
        <ol>
          {(result?.steps || [
            'Carga una imagen',
            'Ajusta altitud y FOV',
            'Ejecuta el analisis para ver las mascaras',
          ]).map((step) => (
            <li key={step}>{step}</li>
          ))}
        </ol>
        {metrics && (
          <p className="gsd-note">
            GSD calculado: <strong>{formatMetric(metrics.gsd_m, 5)} m/px</strong>
          </p>
        )}
      </div>
    );
  }

  const diagnosticImage =
    result?.diagnosticSteps?.[diagnosticIndex]?.image || result?.images?.diagnostic;
  const imageMap = {
    original: preview || assetUrl(result?.images?.original),
    diagnostic: assetUrl(diagnosticImage),
    report: assetUrl(result?.images?.report),
  };
  const emptyMap = {
    original: 'Sube una imagen para verla aqui.',
    diagnostic: 'Aqui aparecera el diagnostico completo del pipeline.',
    report: 'El reporte se genera despues del analisis.',
  };
  const image = imageMap[activeView];
  const empty = emptyMap[activeView];

  return (
    <div className={`image-stage ${activeView === 'report' || activeView === 'steps' ? 'fit-contain' : 'fit-cover'}`}>
      {image ? <img src={image} alt={getViewTitle(activeView)} /> : <p>{empty}</p>}
    </div>
  );
}

export default App;
