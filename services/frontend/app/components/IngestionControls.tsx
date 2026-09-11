"use client";

import { useCallback, useEffect, useState } from "react";
import { API_BASE, getJSONOr } from "@/lib/api";

/**
 * Ingestion control — the simulation clock, driven from the surveillance view.
 *
 * The clock is what makes the live half of the platform live: `future_events`
 * holds a staged month, and the clock promotes each sighting into
 * `vehicle_events` at the moment it comes due. `speed` is simulated seconds
 * per real second, so 60x is one simulated hour per real minute.
 *
 * Putting it here rather than on its own page is the point — during a demo the
 * useful action is "speed the city up until something happens", and that
 * should be one click away from the map it affects, not a separate screen.
 */

interface SimStatus {
  state?: string;
  speed?: number;
  sim_time?: string;
  live_buffer?: number;
  tracked_vehicles?: number;
  counters?: {
    events_promoted?: number;
    backlog?: number;
    last_tick_ms?: number;
    errors?: number;
    last_error?: string | null;
  };
}

const PRESETS = [1, 10, 60, 300, 1000];

export default function IngestionControls() {
  const [status, setStatus] = useState<SimStatus>({});
  const [speed, setSpeed] = useState(60);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    const data = await getJSONOr<SimStatus>("/simulation/status", {});
    setStatus(data);
    if (typeof data.speed === "number") setSpeed(data.speed);
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 2000);
    return () => clearInterval(id);
  }, [refresh]);

  const call = async (path: string, label: string) => {
    setBusy(label);
    setError("");
    try {
      const res = await fetch(`${API_BASE}${path}`, { method: "POST" });
      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error(body?.detail || `HTTP ${res.status}`);
      }
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  };

  const state = (status.state || "UNKNOWN").toUpperCase();
  const running = state === "RUNNING";
  const paused = state === "PAUSED";
  const counters = status.counters || {};

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 10,
        padding: "6px 10px",
        borderBottom: "1px solid var(--panel-border)",
        flexWrap: "wrap",
        flex: "0 0 auto",
      }}
    >
      <span className="kpi-label" style={{ letterSpacing: "0.1em" }}>
        Ingestion
      </span>

      <span
        style={{
          fontSize: 10,
          padding: "1px 6px",
          border: `1px solid ${stateColour(state)}`,
          color: stateColour(state),
        }}
      >
        {running && <span className="dot live" style={{ marginRight: 5 }} />}
        {state}
      </span>

      {/* start / pause / resume / stop */}
      {!running && !paused && (
        <button
          className="btn primary"
          style={btn}
          disabled={busy !== null}
          onClick={() => call(`/simulation/start?speed=${speed}&skip_to_first=true`, "start")}
        >
          ▶ Start
        </button>
      )}
      {running && (
        <button
          className="btn"
          style={btn}
          disabled={busy !== null}
          onClick={() => call("/simulation/pause", "pause")}
        >
          ⏸ Pause
        </button>
      )}
      {paused && (
        <button
          className="btn primary"
          style={btn}
          disabled={busy !== null}
          onClick={() => call("/simulation/resume", "resume")}
        >
          ▶ Resume
        </button>
      )}
      {(running || paused) && (
        <button
          className="btn"
          style={btn}
          disabled={busy !== null}
          onClick={() => call("/simulation/stop", "stop")}
        >
          ■ Stop
        </button>
      )}

      {/* speed */}
      <span
        style={{
          display: "flex",
          alignItems: "center",
          gap: 5,
          borderLeft: "1px solid var(--panel-border)",
          paddingLeft: 10,
        }}
      >
        <span className="muted" style={{ fontSize: 10 }}>
          SPEED
        </span>
        <input
          type="range"
          min={1}
          max={1000}
          step={1}
          value={speed}
          onChange={(e) => setSpeed(Number(e.target.value))}
          onMouseUp={() => call(`/simulation/speed?speed=${speed}`, "speed")}
          onTouchEnd={() => call(`/simulation/speed?speed=${speed}`, "speed")}
          style={{ width: 104, accentColor: "var(--accent-green)" }}
          title="Simulated seconds per real second"
        />
        <span
          className="green"
          style={{ fontSize: 11, minWidth: 42, fontVariantNumeric: "tabular-nums" }}
        >
          {speed}×
        </span>
        {PRESETS.map((p) => (
          <button
            key={p}
            className="btn"
            style={{
              ...btn,
              padding: "1px 5px",
              borderColor: speed === p ? "var(--accent-green)" : "var(--panel-border)",
              color: speed === p ? "var(--accent-green)" : "var(--text-muted)",
            }}
            disabled={busy !== null}
            onClick={() => {
              setSpeed(p);
              call(`/simulation/speed?speed=${p}`, "speed");
            }}
          >
            {p}×
          </button>
        ))}
      </span>

      {/* read-outs */}
      <span
        className="muted"
        style={{
          marginLeft: "auto",
          display: "flex",
          gap: 12,
          fontSize: 10,
          fontVariantNumeric: "tabular-nums",
        }}
      >
        <span>
          SIM CLOCK <span style={{ color: "var(--text)" }}>{simClock(status.sim_time)}</span>
        </span>
        <span>
          PROMOTED{" "}
          <span className="cyan">{(counters.events_promoted ?? 0).toLocaleString()}</span>
        </span>
        <span>
          BACKLOG{" "}
          <span className={(counters.backlog ?? 0) > 500 ? "red" : "green"}>
            {counters.backlog ?? 0}
          </span>
        </span>
        <span>
          TICK <span className="cyan">{(counters.last_tick_ms ?? 0).toFixed(0)}ms</span>
        </span>
      </span>

      {/* A rising backlog is the honest signal that `speed` is set faster than
          the database can promote — the fix is a lower speed, not a bigger box. */}
      {(counters.backlog ?? 0) > 500 && (
        <span className="amber" style={{ fontSize: 10, width: "100%" }}>
          ⚠ Backlog {counters.backlog} — the clock is outrunning the database. Lower the
          speed.
        </span>
      )}
      {(error || counters.last_error) && (
        <span className="red" style={{ fontSize: 10, width: "100%" }}>
          {error || counters.last_error}
        </span>
      )}
    </div>
  );
}

const btn: React.CSSProperties = { padding: "2px 7px", fontSize: 10 };

function stateColour(state: string): string {
  if (state === "RUNNING") return "var(--accent-green)";
  if (state === "PAUSED") return "var(--accent-amber)";
  if (state === "STOPPED") return "var(--text-muted)";
  return "var(--accent-red)";
}

function simClock(iso?: string): string {
  if (!iso) return "—";
  const d = new Date(iso.endsWith("Z") ? iso : `${iso}Z`);
  return Number.isNaN(d.getTime())
    ? "—"
    : d.toLocaleString("en-GB", {
        day: "2-digit",
        month: "short",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        timeZone: "UTC",
      });
}
