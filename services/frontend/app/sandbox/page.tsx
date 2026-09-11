"use client";

import { useEffect, useRef, useState } from "react";
import { API_BASE, WS_BASE, getJSONOr } from "@/lib/api";

/**
 * Sandbox — everything you drive by hand.
 *
 * Same capabilities as the old test.html, plus the benchmarks page folded in
 * as a tab rather than living as its own top-level destination: it is a thing
 * you go and measure, not a thing you monitor, so it belongs with the other
 * manual tools.
 */

type Tab = "video" | "photo" | "dataset" | "benchmarks";

interface JobProgress {
  frames_processed?: number;
  total_frames?: number;
  percent?: number;
  fps?: number;
  detections?: number;
  ocr_calls?: number;
  unique_plates?: number;
}

interface JobPlate {
  plate?: string;
  track_id?: number;
  frame?: number;
  timestamp?: string;
  provisional?: boolean;
  confidence?: number;
}

interface MediaFile {
  name: string;
  path: string;
  size_bytes: number;
  size_label: string;
}

interface JobPreview {
  seq?: number;
  frame?: number;
  available?: boolean;
}

interface JobState {
  job_id: string;
  state: string;
  preview?: JobPreview;
  error?: string | null;
  source?: string;
  progress?: JobProgress;
  config?: Record<string, unknown>;
  stats?: Record<string, unknown>;
  recent_plates?: JobPlate[];
  plates?: JobPlate[];
  plate_count?: number;
  ingestion?: { events_inserted?: number; events_skipped?: number };
  compute?: Record<string, unknown>;
}

const TABS: { id: Tab; label: string }[] = [
  { id: "video", label: "Video ANPR" },
  { id: "photo", label: "Photo ANPR" },
  { id: "dataset", label: "City Dataset" },
  { id: "benchmarks", label: "Benchmarks" },
];

export default function SandboxPage() {
  const [tab, setTab] = useState<Tab>("video");

  return (
    <>
      <div
        style={{
          display: "flex",
          gap: 4,
          padding: "6px 10px",
          borderBottom: "1px solid var(--panel-border)",
          background: "var(--panel)",
          flex: "0 0 auto",
        }}
      >
        {TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            className={`nav-btn ${tab === t.id ? "active" : ""}`}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
        <span style={{ marginLeft: "auto" }} className="muted">
          Uploads run in an isolated sandbox database — they never touch the live
          Delhi deployment.
        </span>
      </div>

      <div className="body-grid" style={{ gridTemplateColumns: "1fr", minHeight: 0 }}>
        {tab === "video" && <MediaJob kind="video" />}
        {tab === "photo" && <MediaJob kind="photo" />}
        {tab === "dataset" && <DatasetUpload />}
        {tab === "benchmarks" && <Benchmarks />}
      </div>
    </>
  );
}

// ── video / photo ───────────────────────────────────────────────────────────

function MediaJob({ kind }: { kind: "video" | "photo" }) {
  const [cameraId, setCameraId] = useState("SANDBOX-01");
  const [sourcePath, setSourcePath] = useState("");
  const [bundled, setBundled] = useState<MediaFile[]>([]);
  const [picked, setPicked] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [stride, setStride] = useState(3);
  const [batch, setBatch] = useState(8);
  const [ocrEvery, setOcrEvery] = useState(3);
  const [job, setJob] = useState<JobState | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [log, setLog] = useState<string[]>([]);
  const socket = useRef<WebSocket | null>(null);

  const note = (line: string) =>
    setLog((prev) => [`${new Date().toLocaleTimeString("en-GB")}  ${line}`, ...prev].slice(0, 200));

  const submit = async () => {
    setError("");
    setBusy(true);
    setJob(null);
    setLog([]);
    try {
      const body = new FormData();
      body.append("camera_id", cameraId);
      body.append("sandbox", "true");
      if (file) body.append("file", file);
      else if (sourcePath) body.append("source_path", sourcePath);
      else throw new Error("Choose a file, or give a server-side path.");

      if (kind === "video") {
        body.append("stride", String(stride));
        body.append("batch", String(batch));
        body.append("ocr_every", String(ocrEvery));
      }

      const res = await fetch(`${API_BASE}/jobs/${kind === "video" ? "video" : "photo"}`, {
        method: "POST",
        body,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || `HTTP ${res.status}`);

      note(`job ${data.job_id} queued (${data.source})`);
      setJob({ job_id: data.job_id, state: data.state });
      watch(data.job_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  // Progress arrives over the job's own WebSocket; polling is the fallback so
  // the panel still advances if the socket cannot be established.
  const watch = (jobId: string) => {
    socket.current?.close();
    try {
      const ws = new WebSocket(`${WS_BASE}/ws/jobs/${jobId}`);
      socket.current = ws;
      ws.onmessage = (msg) => {
        try {
          const payload = JSON.parse(msg.data) as JobState;
          setJob(payload);
          const texts = payload.recent_plates?.filter((p) => p.plate) ?? [];
          if (texts.length) {
            const latest = texts[texts.length - 1];
            note(`read ${latest.plate}${latest.provisional ? " (voting…)" : ""}`);
          }
        } catch {
          /* ignore malformed frame */
        }
      };
      ws.onclose = () => poll(jobId);
      ws.onerror = () => ws.close();
    } catch {
      poll(jobId);
    }
  };

  const poll = (jobId: string) => {
    const id = setInterval(async () => {
      const data = await getJSONOr<JobState | null>(`/jobs/${jobId}`, null);
      if (!data) return;
      setJob(data);
      if (["DONE", "FAILED", "CANCELLED"].includes(data.state)) {
        clearInterval(id);
        note(`job ${data.state}`);
      }
    }, 1500);
  };

  // Footage already on the server, offered as click-to-run. Listed by the API
  // from the same allowlist `source_path` is validated against, so what is
  // offered and what is accepted cannot drift.
  useEffect(() => {
    let alive = true;
    getJSONOr<{ videos: MediaFile[]; images: MediaFile[] }>("/jobs/media", {
      videos: [],
      images: [],
    }).then((d) => {
      if (!alive) return;
      const files = kind === "video" ? d.videos : d.images;
      setBundled(files);
      // Default to the smallest clip: on a demo the useful default is whichever
      // finishes quickest.
      if (files.length && !picked && !sourcePath) {
        setPicked(files[0].path);
        setSourcePath(files[0].path);
      }
    });
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [kind]);

  useEffect(() => () => socket.current?.close(), []);

  const progress = job?.progress;
  const finished = job && ["DONE", "FAILED", "CANCELLED"].includes(job.state);
  const plates = (job?.plates?.length ? job.plates : job?.recent_plates) ?? [];

  return (
    <div style={{ display: "grid", gridTemplateColumns: "330px 1fr", gap: 1, minHeight: 0 }}>
      <div className="panel">
        <div className="panel-head">{kind === "video" ? "Video" : "Photo"} job</div>
        <div className="panel-body pad">
          <Field label="Camera ID">
            <input type="text" value={cameraId} onChange={(e) => setCameraId(e.target.value)} />
          </Field>

          <Field label={`${kind === "video" ? "Footage" : "Images"} on this machine — click to run`}>
            {bundled.length === 0 ? (
              <div className="muted" style={{ fontSize: 10.5 }}>
                none found in the allowlisted directories
              </div>
            ) : (
              <div
                style={{
                  maxHeight: 168,
                  overflowY: "auto",
                  border: "1px solid var(--panel-border)",
                  background: "var(--bg)",
                }}
              >
                {bundled.map((f) => {
                  const on = picked === f.path;
                  return (
                    <button
                      key={f.path}
                      type="button"
                      onClick={() => {
                        setPicked(f.path);
                        setSourcePath(f.path);
                        setFile(null); // a picked file and an upload are exclusive
                      }}
                      style={{
                        display: "flex",
                        gap: 8,
                        width: "100%",
                        textAlign: "left",
                        background: on ? "rgba(158,206,106,0.10)" : "transparent",
                        border: "none",
                        borderLeft: `2px solid ${on ? "var(--accent-green)" : "transparent"}`,
                        borderBottom: "1px solid rgba(30,36,51,0.5)",
                        color: on ? "var(--accent-green)" : "var(--text)",
                        font: "inherit",
                        fontSize: 10.5,
                        padding: "4px 7px",
                        cursor: "pointer",
                      }}
                      title={f.path}
                    >
                      <span className="muted" style={{ width: 46, flex: "0 0 46px" }}>
                        {f.size_label}
                      </span>
                      <span
                        style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
                      >
                        {f.name}
                      </span>
                    </button>
                  );
                })}
              </div>
            )}
            <div className="muted" style={{ fontSize: 10, marginTop: 3 }}>
              Runs in place — nothing is re-uploaded. Smallest first.
            </div>
          </Field>

          <Field label={`…or upload a new ${kind}`}>
            <input
              type="file"
              accept={kind === "video" ? "video/*" : "image/*"}
              onChange={(e) => {
                setFile(e.target.files?.[0] ?? null);
                if (e.target.files?.[0]) {
                  setPicked(null);
                  setSourcePath("");
                }
              }}
              style={{ fontSize: 11, color: "var(--text-muted)" }}
            />
          </Field>

          <Field label="…or an explicit server path">
            <input
              type="text"
              value={sourcePath}
              placeholder="/Users/…/archive/clip.mp4"
              onChange={(e) => setSourcePath(e.target.value)}
            />
            <div className="muted" style={{ fontSize: 10, marginTop: 3 }}>
              Avoids re-uploading large footage. Restricted to an allowlist of
              directories server-side.
            </div>
          </Field>

          {kind === "video" && (
            <>
              <div
                style={{
                  marginTop: 10,
                  paddingTop: 8,
                  borderTop: "1px solid var(--panel-border)",
                }}
              >
                <div className="kpi-label" style={{ marginBottom: 6 }}>
                  Throughput
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 6 }}>
                  <Field label="Stride" tight>
                    <input
                      type="number"
                      min={1}
                      value={stride}
                      onChange={(e) => setStride(Math.max(1, +e.target.value))}
                    />
                  </Field>
                  <Field label="Batch" tight>
                    <input
                      type="number"
                      min={1}
                      value={batch}
                      onChange={(e) => setBatch(Math.max(1, +e.target.value))}
                    />
                  </Field>
                  <Field label="OCR every" tight>
                    <input
                      type="number"
                      min={1}
                      value={ocrEvery}
                      onChange={(e) => setOcrEvery(Math.max(1, +e.target.value))}
                    />
                  </Field>
                </div>
                <div className="muted" style={{ fontSize: 10, marginTop: 4 }}>
                  Stride processes 1 frame in N — the dominant speed knob. Batch is
                  accuracy-neutral. Set stride 1 for a frame-exact run.
                </div>
              </div>
            </>
          )}

          <button
            className="btn primary"
            style={{ width: "100%", marginTop: 12, padding: 9 }}
            onClick={submit}
            disabled={busy || (!!job && !finished)}
          >
            {busy ? "Submitting…" : job && !finished ? "Running…" : "Run ANPR"}
          </button>

          {error && (
            <div className="red" style={{ fontSize: 11, marginTop: 8 }}>
              {error}
            </div>
          )}
          <div className="muted" style={{ fontSize: 10, marginTop: 8 }}>
            One job runs at a time by design — the detector and OCR reader are
            shared singletons, so a second submission queues.
          </div>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateRows: "auto 1.15fr 1fr", gap: 1, minHeight: 0 }}>
        <div className="panel" style={{ flex: "0 0 auto" }}>
          <div className="panel-head">
            <span className={`dot ${job && !finished ? "live" : finished ? "" : "warn"}`} />
            <span>Progress</span>
            <span className="spacer" />
            <span className="muted" style={{ letterSpacing: 0 }}>
              {job ? `${job.job_id.slice(0, 8)} · ${job.state}` : "idle"}
            </span>
          </div>
          {progress && (
            <>
              <div className="bar">
                <span style={{ width: `${progress.percent ?? 0}%` }} />
              </div>
              <div className="kpis" style={{ borderBottom: "none" }}>
                <Mini label="Frames" value={`${progress.frames_processed ?? 0}/${progress.total_frames ?? "?"}`} />
                <Mini label="Percent" value={`${(progress.percent ?? 0).toFixed(1)}%`} />
                <Mini label="FPS" value={(progress.fps ?? 0).toFixed(2)} />
                <Mini label="Detections" value={progress.detections ?? 0} />
                <Mini label="OCR calls" value={progress.ocr_calls ?? 0} />
                <Mini label="Plates" value={job?.plate_count ?? plates.length} />
              </div>
            </>
          )}
          {job?.error && (
            <div className="red" style={{ padding: 8, fontSize: 11 }}>
              {job.error}
            </div>
          )}
          {!job && (
            <div className="empty">
              Submit a job and progress streams here over a WebSocket.
            </div>
          )}
        </div>

        {kind === "video" && <FramePreview job={job} />}

        <div className="panel">
          <div className="panel-head">
            Plates {plates.length > 0 && <span className="muted">· {plates.length}</span>}
          </div>
          <div className="panel-body">
            {plates.length === 0 ? (
              <div className="empty">
                Confirmed plates appear here. A read is only logged once it has
                survived cross-frame voting, the country grammar and the duplicate
                cooldown — so the list stays short and trustworthy rather than
                showing every frame&apos;s guess.
              </div>
            ) : (
              <table className="log">
                <thead>
                  <tr>
                    <th>Plate</th>
                    <th>Track</th>
                    <th>Frame</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {plates.map((p, i) => (
                    <tr key={i}>
                      <td className="plate">{p.plate}</td>
                      <td className="muted">{p.track_id ?? "—"}</td>
                      <td className="muted">{p.frame ?? "—"}</td>
                      <td className={p.provisional ? "amber" : "green"}>
                        {p.provisional ? "voting" : "confirmed"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>

      </div>
    </div>
  );
}

/**
 * The frame the detector is looking at, right now.
 *
 * The backend encodes an annotated JPEG at ~6 fps (boxes with confidence, plus
 * the in-progress vote drawn top-left) and serves the newest one from
 * `/jobs/{id}/preview.jpg`. This pulls it rather than receiving it over the
 * progress socket, so a slow display never backs up the pipeline.
 *
 * `seq` drives the refetch: the URL carries it as a cache-buster, so a frame is
 * fetched exactly once and the browser is never asked to revalidate. While the
 * job is running we also poll on a timer, because between two status messages
 * several new frames will have been encoded.
 */
function FramePreview({ job }: { job: JobState | null }) {
  const [src, setSrc] = useState<string | null>(null);
  const [tick, setTick] = useState(0);
  const running = Boolean(job && !["DONE", "FAILED", "CANCELLED"].includes(job.state));
  // QUEUED is a real state with a real cause — one ANPR job runs at a time
  // because the detector and OCR reader are shared, non-thread-safe
  // singletons. Saying "waiting for the first encoded frame" during a queue
  // wait describes the symptom and hides the reason, which reads as a bug.
  const queued = job?.state === "QUEUED";

  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => setTick((n) => n + 1), 180);
    return () => clearInterval(id);
  }, [running]);

  useEffect(() => {
    if (!job) {
      setSrc(null);
      return;
    }
    const seq = job.preview?.seq ?? 0;
    setSrc(`${API_BASE}/jobs/${job.job_id}/preview.jpg?seq=${seq}-${tick}`);
  }, [job, tick]);

  return (
    <div className="panel">
      <div className="panel-head">
        <span className={`dot ${running ? "live" : ""}`} />
        <span>Live frame</span>
        <span className="spacer" />
        {job?.preview?.frame ? (
          <span className="muted" style={{ letterSpacing: 0 }}>
            frame {job.preview.frame.toLocaleString()}
          </span>
        ) : null}
      </div>
      <div
        className="panel-body"
        style={{
          display: "grid",
          placeItems: "center",
          background: "#05060a",
          overflow: "hidden",
          padding: 6,
        }}
      >
        {queued ? (
          <div className="empty" style={{ textAlign: "center" }}>
            <span className="amber">Queued.</span>
            <br />
            Another job is using the detector — one runs at a time, because the
            detector and OCR reader are shared singletons and neither is
            thread-safe. This starts automatically when the one ahead finishes.
          </div>
        ) : src && job?.preview?.available ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={src}
            alt="frame being processed"
            style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain" }}
            // Deliberately does NOT clear `src`. A transient 204 (the next
            // frame is not encoded yet) or a dropped request would otherwise
            // blank the panel and fall through to the placeholder, which looked
            // like the preview had died. Holding the last good frame until a
            // newer one decodes is the honest rendering of "still working".
            onError={() => {}}
          />
        ) : (
          <div className="empty" style={{ textAlign: "center" }}>
            {running
              ? "Starting up — the first frame appears once the detector has loaded."
              : "Pick a clip and run it to watch the detector work frame by frame."}
            <br />
            <span style={{ fontSize: 10, opacity: 0.75 }}>
              Green boxes are plate detections with confidence. Amber text is the
              cross-frame vote converging.
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

// ── city dataset ────────────────────────────────────────────────────────────

function DatasetUpload() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [schema, setSchema] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    getJSONOr<Record<string, unknown> | null>("/jobs/dataset/schema", null).then(setSchema);
  }, []);

  const upload = async () => {
    if (!file) return;
    setBusy(true);
    setError("");
    setResult(null);
    try {
      const body = new FormData();
      body.append("file", file);
      body.append("sandbox", "true");
      const res = await fetch(`${API_BASE}/jobs/dataset`, { method: "POST", body });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || `HTTP ${res.status}`);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div style={{ display: "grid", gridTemplateColumns: "330px 1fr", gap: 1, minHeight: 0 }}>
      <div className="panel">
        <div className="panel-head">Bring your own city</div>
        <div className="panel-body pad">
          <div className="muted" style={{ fontSize: 11, marginBottom: 10 }}>
            Upload your camera network plus the sightings recorded on it, and the
            whole platform — map, trajectories, heatmap, alerts, analytics — comes
            up on that city instead of Delhi.
          </div>
          <Field label="Dataset (JSON or CSV)">
            <input
              type="file"
              accept=".json,.csv"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              style={{ fontSize: 11, color: "var(--text-muted)" }}
            />
          </Field>
          <button
            className="btn primary"
            style={{ width: "100%", marginTop: 10, padding: 9 }}
            onClick={upload}
            disabled={busy || !file}
          >
            {busy ? "Validating…" : "Upload & ingest"}
          </button>
          {error && (
            <div className="red" style={{ fontSize: 11, marginTop: 8 }}>
              {error}
            </div>
          )}
          <div style={{ marginTop: 12 }}>
            <a href={`${API_BASE}/jobs/dataset/schema`} target="_blank" rel="noreferrer">
              → field reference (served by the API)
            </a>
          </div>
        </div>
      </div>

      <div className="panel">
        <div className="panel-head">Result</div>
        <div className="panel-body pad">
          {result ? (
            <pre style={{ fontSize: 10.5, whiteSpace: "pre-wrap", margin: 0 }}>
              {JSON.stringify(result, null, 2)}
            </pre>
          ) : schema ? (
            <>
              <div className="muted" style={{ fontSize: 11, marginBottom: 8 }}>
                The contract below is served by the API itself, so this table cannot
                drift from what the parser accepts.
              </div>
              <pre style={{ fontSize: 10, whiteSpace: "pre-wrap", margin: 0, opacity: 0.75 }}>
                {JSON.stringify(schema, null, 2).slice(0, 4000)}
              </pre>
            </>
          ) : (
            <div className="empty">Upload a dataset to see the ingestion report.</div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── benchmarks ──────────────────────────────────────────────────────────────

function Benchmarks() {
  const [hardware, setHardware] = useState<Record<string, unknown> | null>(null);
  const [projection, setProjection] = useState<Record<string, unknown> | null>(null);
  const [assumptions, setAssumptions] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    getJSONOr<Record<string, unknown> | null>("/benchmarks/hardware", null).then(setHardware);
    getJSONOr<Record<string, unknown> | null>("/benchmarks/projection", null).then(setProjection);
    getJSONOr<Record<string, unknown> | null>("/benchmarks/assumptions", null).then(setAssumptions);
  }, []);

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 1, minHeight: 0 }}>
      <Json title="Measured hardware" data={hardware} />
      <Json title="City-scale projection" data={projection} />
      <Json
        title="Assumptions"
        data={assumptions}
        note="Published so the projections can be argued with rather than believed."
      />
    </div>
  );
}

function Json({
  title,
  data,
  note,
}: {
  title: string;
  data: Record<string, unknown> | null;
  note?: string;
}) {
  return (
    <div className="panel">
      <div className="panel-head">{title}</div>
      <div className="panel-body pad">
        {note && (
          <div className="muted" style={{ fontSize: 10.5, marginBottom: 8 }}>
            {note}
          </div>
        )}
        {data ? (
          <pre style={{ fontSize: 10.5, whiteSpace: "pre-wrap", margin: 0 }}>
            {JSON.stringify(data, null, 2)}
          </pre>
        ) : (
          <div className="empty">unavailable</div>
        )}
      </div>
    </div>
  );
}

// ── small helpers ───────────────────────────────────────────────────────────

function Field({
  label,
  children,
  tight,
}: {
  label: string;
  children: React.ReactNode;
  tight?: boolean;
}) {
  return (
    <div style={{ marginBottom: tight ? 0 : 10 }}>
      <div className="kpi-label" style={{ marginBottom: 3 }}>
        {label}
      </div>
      {children}
    </div>
  );
}

function Mini({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="kpi" style={{ padding: "6px 10px" }}>
      <div className="kpi-label">{label}</div>
      <div className="kpi-value" style={{ fontSize: 15 }}>
        {value}
      </div>
    </div>
  );
}
