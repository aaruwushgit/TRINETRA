"use client";

import { useMemo, useState } from "react";
import dynamic from "next/dynamic";
import { useLiveFeed } from "@/lib/useLiveFeed";
import { TOMTOM_KEY, pick, type InferredPlace, type NextHop, type RouteLeg } from "@/lib/api";
import TrajectoryPanel from "./components/TrajectoryPanel";
import IngestionControls from "./components/IngestionControls";
import { STYLES, type StyleKey } from "./components/Map3D";
import PatternPanel from "./components/PatternPanel";
import HistoryPanel from "./components/HistoryPanel";

// MapLibre needs WebGL and `window`, so the map never server-renders.
const Map3D = dynamic(() => import("./components/Map3D"), {
  ssr: false,
  loading: () => (
    <div className="map" style={{ display: "grid", placeItems: "center" }}>
      <span className="muted">initialising 3D map…</span>
    </div>
  ),
});

/**
 * Surveillance — one operational screen.
 *
 * Previously this was three separate pages, and each looked broken in
 * isolation: a map of cameras that never changed, an empty trajectory view,
 * and a static log. None of them was wrong, they were just disconnected. Put
 * together and driven off the live feed, the same data reads as a system that
 * is running: cameras light up as they fire, the log grows, and the
 * trajectory panel already has a vehicle in it.
 */
export default function SurveillancePage() {
  const feed = useLiveFeed();
  const [legs, setLegs] = useState<RouteLeg[]>([]);
  const [predictions, setPredictions] = useState<NextHop[]>([]);
  // A plate chosen by clicking a row in the detection log. Lifted here so the
  // log and the trajectory panel — siblings, not parent and child — can talk.
  const [selectedPlate, setSelectedPlate] = useState<string | null>(null);
  const [places, setPlaces] = useState<InferredPlace[]>([]);
  const [showTraffic, setShowTraffic] = useState(Boolean(TOMTOM_KEY));
  const [showCameras, setShowCameras] = useState(true);
  const [showHeat, setShowHeat] = useState(true);
  const [showBuildings, setShowBuildings] = useState(true);
  // 0 is the flat overhead view; 55 is the tilted one. Kept as a number rather
  // than a boolean so the control can animate between them.
  const [pitch, setPitch] = useState(50);
  const [styleKey, setStyleKey] = useState<StyleKey>("dark");

  // `total_detections` is what /analytics/summary actually returns; the older
  // names are kept as fallbacks so a schema change degrades to 0 rather than
  // to a crash.
  const totalEvents = pick<number>(
    feed.summary,
    ["total_detections", "total_events", "events_total"],
    0,
  );
  const lastHour = pick<number>(feed.summary, ["detections_last_hour"], 0);
  const uniqueVehicles = pick<number>(
    feed.summary,
    ["unique_vehicles", "unique_plates"],
    0,
  );
  const avgSpeed = pick<number>(feed.summary, ["avg_speed_kmh", "average_speed"], 0);

  const withCoords = useMemo(
    () => feed.cameras.filter((c) => c.latitude != null && c.longitude != null),
    [feed.cameras],
  );

  return (
    <>
      {/* Two bands, not one row. The old strip mixed "60 sightings/min" with
          "36.6M total" as if they were the same kind of fact; they are a live
          rate and a year-long total, and reading them together is misleading. */}
      <div style={{ display: "flex", flex: "0 0 auto", background: "var(--panel-border)", gap: 1 }}>
        <div style={{ flex: "0 0 auto", display: "flex", alignItems: "center", padding: "0 10px", background: "var(--panel)" }}>
          <span className="dot live" style={{ marginRight: 6 }} />
          <span className="kpi-label" style={{ color: "var(--accent-green)" }}>LIVE</span>
        </div>
        <div className="kpis" style={{ flex: 1, borderBottom: "none" }}>
          <Kpi label="Cameras online" value={withCoords.length} tone="cyan" />
          <Kpi
            label="Firing now"
            value={feed.liveCameraIds.size}
            tone="green"
            pulse={feed.liveCameraIds.size > 0}
          />
          <Kpi label="Sightings / min" value={feed.eventsPerMin} tone="purple" />
          <Kpi label="Last hour" value={fmt(lastHour)} tone="green" />
          <Kpi
            label="Active alerts"
            value={feed.alerts.length}
            tone={feed.alerts.length ? "red" : "green"}
          />
        </div>

        <div style={{ flex: "0 0 auto", display: "flex", alignItems: "center", padding: "0 10px", background: "var(--panel)" }}>
          <span className="cyan" style={{ marginRight: 6 }}>▤</span>
          <span className="kpi-label" style={{ color: "var(--accent-cyan)" }}>ARCHIVE · 365d</span>
        </div>
        <div className="kpis" style={{ flex: 1, borderBottom: "none" }}>
          <Kpi label="Total sightings" value={fmt(totalEvents)} tone="cyan" />
          <Kpi label="Unique vehicles" value={fmt(uniqueVehicles)} tone="cyan" />
          <Kpi
            label="Avg speed"
            value={avgSpeed ? `${Number(avgSpeed).toFixed(0)} km/h` : "—"}
            tone="amber"
          />
        </div>
      </div>

      <div
        className="body-grid"
        style={{ gridTemplateColumns: "1fr 340px", gap: 1, background: "var(--panel-border)" }}
      >
        {/* ── map + log ───────────────────────────────────────────────────── */}
        <div style={{ display: "grid", gridTemplateRows: "1.35fr 1fr", gap: 1, minHeight: 0 }}>
          <div className="panel">
            <div className="panel-head">
              <span className="dot live" />
              <span>Live Camera Network — Delhi</span>
              <span className="spacer" />
              <Toggle on={showCameras} set={setShowCameras} label="Cameras" />
              <Toggle on={showHeat} set={setShowHeat} label="Congestion" />
              <select
                value={styleKey}
                onChange={(e) => setStyleKey(e.target.value as StyleKey)}
                title="Basemap theme"
                style={{
                  width: "auto",
                  padding: "1px 4px",
                  fontSize: 10,
                  background: "var(--bg)",
                  borderColor: "var(--panel-border)",
                  color: "var(--text-muted)",
                }}
              >
                {Object.entries(STYLES).map(([key, s]) => (
                  <option key={key} value={key}>
                    {s.label}
                  </option>
                ))}
              </select>
              <Toggle on={showBuildings} set={setShowBuildings} label="3D" />
              <Toggle
                on={pitch > 10}
                set={(v) => setPitch(v ? 50 : 0)}
                label="Tilt"
                title="Switch between the tilted 3D view and flat overhead"
              />
              <Toggle
                on={showTraffic}
                set={setShowTraffic}
                label="TomTom"
                disabled={!TOMTOM_KEY}
                title={
                  TOMTOM_KEY
                    ? "Live TomTom traffic flow"
                    : "Set NEXT_PUBLIC_TOMTOM_API_KEY in services/frontend/.env.local"
                }
              />
            </div>
            <IngestionControls />
            <div className="panel-body" style={{ overflow: "hidden" }}>
              <Map3D
                cameras={withCoords}
                heat={feed.heat}
                liveCameraIds={feed.liveCameraIds}
                legs={legs}
                predictions={predictions}
                showTraffic={showTraffic}
                showCameras={showCameras}
                showHeat={showHeat}
                showBuildings={showBuildings}
                pitch={pitch}
                styleKey={styleKey}
                places={places}
              />
            </div>
            <div className="legend">
              <span>
                <span
                  className="swatch"
                  style={{ background: "linear-gradient(90deg,#565f89,#bb9af7,#7dcfff)" }}
                />
                <b>Trip path</b> — dim = oldest, cyan = most recent
              </span>
              <span className="muted">hover a leg for times, distance &amp; both speeds</span>
              <span>
                <span
                  className="swatch"
                  style={{
                    background:
                      "repeating-linear-gradient(90deg,#e0af68 0 4px,transparent 4px 8px)",
                  }}
                />
                Straight-line fallback
              </span>
              <span>
                <span className="cam-marker live" style={{ display: "inline-block" }} />{" "}
                Firing now
              </span>
              <span>
                <span className="cam-marker" style={{ display: "inline-block" }} /> Idle
                camera
              </span>
              <span>
                <span
                  style={{
                    display: "inline-block",
                    width: 8,
                    height: 8,
                    borderRadius: "50%",
                    border: "2px solid var(--accent-red)",
                    marginRight: 4,
                    verticalAlign: "middle",
                  }}
                />
                <b>Blinking = predicted next hop</b>
              </span>
              <span className="spacer" style={{ marginLeft: "auto" }} />
              {!TOMTOM_KEY && (
                <span className="amber">
                  TomTom traffic: no key configured — showing our own congestion counts
                </span>
              )}
            </div>
          </div>

          {/* Live and archive side by side, so which half of the product you
              are looking at is never ambiguous. */}
          <div style={{ display: "grid", gridTemplateColumns: "1.5fr 1fr", gap: 1, minHeight: 0 }}>
            <DetectionLog
              feed={feed}
              selected={selectedPlate}
              onSelect={setSelectedPlate}
            />
            <HistoryPanel summary={feed.summary} />
          </div>
        </div>

        {/* ── right rail ──────────────────────────────────────────────────── */}
        <div
          style={{
            display: "grid",
            gridTemplateRows: "1.1fr 1.25fr 0.75fr",
            gap: 1,
            minHeight: 0,
          }}
        >
          <TrajectoryPanel
            events={feed.events}
            selected={selectedPlate}
            onSelect={setSelectedPlate}
            onLegs={setLegs}
            onPredictions={setPredictions}
          />
          <PatternPanel plate={selectedPlate} onPlaces={setPlaces} />
          <AlertsPanel feed={feed} />
        </div>
      </div>
    </>
  );
}

// ── pieces ──────────────────────────────────────────────────────────────────

function Kpi({
  label,
  value,
  tone,
  pulse,
}: {
  label: string;
  value: string | number;
  tone?: string;
  pulse?: boolean;
}) {
  return (
    <div className="kpi">
      <div className="kpi-label">
        {pulse && <span className="dot live" style={{ marginRight: 5 }} />}
        {label}
      </div>
      <div className={`kpi-value ${tone ?? ""}`}>{value}</div>
    </div>
  );
}

function Toggle({
  on,
  set,
  label,
  disabled,
  title,
}: {
  on: boolean;
  set: (v: boolean) => void;
  label: string;
  disabled?: boolean;
  title?: string;
}) {
  return (
    <button
      type="button"
      className="btn"
      title={title}
      disabled={disabled}
      onClick={() => set(!on)}
      style={{
        padding: "2px 7px",
        fontSize: 10,
        borderColor: on && !disabled ? "var(--accent-green)" : "var(--panel-border)",
        color: disabled ? "var(--text-muted)" : on ? "var(--accent-green)" : "var(--text-muted)",
      }}
    >
      {on && !disabled ? "◉" : "○"} {label}
    </button>
  );
}

function DetectionLog({
  feed,
  selected,
  onSelect,
}: {
  feed: ReturnType<typeof useLiveFeed>;
  selected: string | null;
  onSelect: (plate: string) => void;
}) {
  return (
    <div className="panel">
      <div className="panel-head">
        <span className={`dot ${feed.paused ? "warn" : "live"}`} />
        <span>Detection Log</span>
        <span className="muted" style={{ letterSpacing: 0 }}>
          streaming · {feed.events.length} buffered · click a row to trace it
        </span>
        <span className="spacer" />
        <button
          type="button"
          className="btn"
          style={{ padding: "2px 7px", fontSize: 10 }}
          onClick={() => feed.setPaused(!feed.paused)}
        >
          {feed.paused ? "▶ Resume" : "⏸ Pause"}
        </button>
      </div>
      <div className="panel-body">
        {feed.events.length === 0 ? (
          <div className="empty">
            No sightings yet. The <span className="cyan">live_feeder</span> container
            promotes staged events into the live table — if this stays empty, check
            that it is running.
          </div>
        ) : (
          <table className="log">
            <thead>
              <tr>
                <th>Time</th>
                <th>Plate</th>
                <th>Camera</th>
                <th>Type</th>
                <th>Colour</th>
                <th>Speed</th>
                <th>Conf</th>
              </tr>
            </thead>
            <tbody>
              {feed.events.map((e) => (
                <tr
                  key={e.event_id}
                  className={`${feed.freshIds.has(e.event_id) ? "fresh" : ""} ${
                    e.plate ? "clickable" : ""
                  } ${e.plate && e.plate === selected ? "selected" : ""}`}
                  onClick={() => e.plate && onSelect(e.plate)}
                  title={
                    e.plate
                      ? `Trace ${e.plate} across the network`
                      : "No plate on this sighting — nothing to trace"
                  }
                >
                  <td className="muted">{clock(e.timestamp)}</td>
                  <td>
                    {e.plate ? (
                      <span className="plate">{e.plate}</span>
                    ) : (
                      <span className="muted" title="Attribute-only sighting: no plate read">
                        —
                      </span>
                    )}
                  </td>
                  <td className="cyan">{e.camera_id}</td>
                  <td className="muted">{e.vehicle_type ?? "—"}</td>
                  <td className="muted">{e.vehicle_color ?? "—"}</td>
                  <td className={speedTone(e.speed)}>
                    {e.speed != null ? `${e.speed.toFixed(0)}` : "—"}
                  </td>
                  <td className="muted">
                    {e.plate_confidence != null
                      ? `${(e.plate_confidence * 100).toFixed(0)}%`
                      : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

function AlertsPanel({ feed }: { feed: ReturnType<typeof useLiveFeed> }) {
  return (
    <div className="panel">
      <div className="panel-head">
        <span className={`dot ${feed.wsConnected ? "live" : "warn"}`} />
        <span>Enforcement Alerts</span>
        <span className="spacer" />
        <span className="muted" style={{ letterSpacing: 0 }}>
          {feed.wsConnected ? "WS LIVE" : "POLLING"}
        </span>
      </div>
      <div className="panel-body">
        {feed.alerts.length === 0 ? (
          <div className="empty">
            No active alerts. Blacklist hits, speed violations and route anomalies
            appear here the moment they fire.
          </div>
        ) : (
          feed.alerts.map((a, i) => (
            <div
              key={a.alert_id ?? a.id ?? i}
              style={{
                padding: "6px 10px",
                borderBottom: "1px solid rgba(30,36,51,0.5)",
                borderLeft: `2px solid ${alertColour(a.alert_type)}`,
              }}
            >
              <div style={{ fontSize: 10.5 }}>
                <span style={{ color: alertColour(a.alert_type) }}>{a.alert_type}</span>
                {a.plate && <span className="plate"> {a.plate}</span>}
              </div>
              <div className="muted" style={{ fontSize: 10 }}>
                {a.camera_id ?? "—"} · {clock(a.timestamp ?? a.created_at)}
              </div>
              {a.message && (
                <div className="muted" style={{ fontSize: 10, opacity: 0.8 }}>
                  {a.message}
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}

// ── formatting ──────────────────────────────────────────────────────────────

function clock(ts?: string): string {
  if (!ts) return "—";
  const d = new Date(ts.endsWith("Z") || ts.includes("+") ? ts : `${ts}Z`);
  return Number.isNaN(d.getTime())
    ? "—"
    : d.toLocaleTimeString("en-GB", { hour12: false });
}

function fmt(n: number): string {
  if (!n) return "0";
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return String(n);
}

function speedTone(speed: number | null): string {
  if (speed == null) return "muted";
  if (speed > 80) return "red";
  if (speed > 60) return "amber";
  return "";
}

function alertColour(type: string): string {
  if (type?.includes("BLACKLIST")) return "#f7768e";
  if (type?.includes("SPEED")) return "#e0af68";
  if (type?.includes("ANOMALY")) return "#bb9af7";
  return "#7dcfff";
}
