"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  getJSON,
  getJSONOr,
  type NextHop,
  type PredictionResponse,
  type RouteLeg,
  type TrajectoryResponse,
  type VehicleEvent,
} from "@/lib/api";

/**
 * Plate -> road-snapped trajectory.
 *
 * Two things here are load-bearing.
 *
 * **It always hits `/routing/trajectory`.** The old UI had two trajectory
 * views and one of them called `/vehicles/{plate}/trajectory`, which returns
 * bare camera coordinates — so it drew straight lines between junctions,
 * through buildings and across the Yamuna. That endpoint is a cheap lower
 * bound, not a path. The routing router resolves each hop against OSRM (3,902
 * routes are already cached in the DB) and returns real road geometry.
 *
 * **It never shows an empty panel if it can avoid it.** An operator screen
 * that says "search for a plate" tells you nothing about whether the system
 * works. Given the most recent sighting, it loads that vehicle's trail on its
 * own, so the panel arrives populated and the demo starts from something real.
 */

/**
 * How far back to reconstruct, in hours, per named window.
 *
 * This is a cost control as much as a UI affordance. On 36.9M rows a cold
 * `/routing/trajectory` over a year measured 19.9s, because it resolves an
 * OSRM leg for every consecutive pair of sightings and a year of a commuter is
 * hundreds of hops. The live window answers in milliseconds off the same
 * index. Asking for a year by default made the panel look broken.
 */
export const WINDOWS = {
  live: { label: "Live · 6h", hours: 6, limit: 40 },
  day: { label: "24h", hours: 24, limit: 80 },
  week: { label: "7d", hours: 168, limit: 150 },
  year: { label: "Full year", hours: 8760, limit: 300 },
} as const;

export type WindowKey = keyof typeof WINDOWS;

export interface TrajectoryPanelProps {
  /** Newest sightings, used to pick a plate to show unprompted. */
  events: VehicleEvent[];
  onLegs: (legs: RouteLeg[]) => void;
  /** Lifted to the page so the map can blink the predicted destinations. */
  onPredictions: (hops: NextHop[]) => void;
  /** A plate selected elsewhere — clicking a row in the detection log. */
  selected: string | null;
  onSelect: (plate: string) => void;
}

export default function TrajectoryPanel({
  events,
  selected,
  onSelect,
  onLegs,
  onPredictions,
}: TrajectoryPanelProps) {
  const [plate, setPlate] = useState("");
  const [active, setActive] = useState<string | null>(null);
  const [legs, setLegs] = useState<RouteLeg[]>([]);
  const [roadKm, setRoadKm] = useState<number | null>(null);
  const [status, setStatus] = useState<"idle" | "loading" | "ok" | "empty" | "error">("idle");
  const [detail, setDetail] = useState("");
  const [autoPicked, setAutoPicked] = useState(false);
  const [hops, setHops] = useState<NextHop[]>([]);
  const [windowKey, setWindowKey] = useState<WindowKey>("day");
  // Monotonic request id; see the generation guard in `load`.
  const generation = useRef(0);
  const [lastSeen, setLastSeen] = useState<PredictionResponse["last_sighting"] | null>(null);

  const load = useCallback(
    async (target: string, windowKey: WindowKey) => {
      const query = target.trim().toUpperCase();
      if (!query) return;

      // Generation guard. The auto-pick and the external-selection effect can
      // both fire for the same plate, and a slow in-flight request must never
      // be able to overwrite a newer one's results — nor, on failure, wipe
      // them. Without this a duplicate request that errored cleared the
      // predictions the successful one had just set, which is exactly why the
      // next-hop markers intermittently never appeared.
      const gen = ++generation.current;
      const stale = () => gen !== generation.current;

      const win = WINDOWS[windowKey];
      setStatus("loading");
      setDetail("");
      setActive(query);
      try {
        const data = await getJSON<TrajectoryResponse>(
          `/routing/trajectory/${encodeURIComponent(query)}` +
            `?limit=${win.limit}&hours=${win.hours}`,
        );
        if (stale()) return;
        const found = (data.legs || data.segments || []) as RouteLeg[];
        setLegs(found);
        onLegs(found);
        setRoadKm(data.total_road_km ?? data.total_distance_km ?? null);
        if (found.length === 0) {
          setStatus("empty");
          setDetail(`No multi-camera hops for ${query} in the last ${win.label}.`);
        } else {
          setStatus("ok");
        }

      } catch (err) {
        if (stale()) return;
        setStatus("error");
        setLegs([]);
        onLegs([]);
        setDetail(err instanceof Error ? err.message : "routing unavailable");
      }

      // Prediction is fetched OUTSIDE the trajectory try/catch, which the old
      // comment claimed but the code did not do — it sat inside the same block,
      // so a trajectory failure skipped it and the catch then cleared any
      // predictions already on screen. They are genuinely independent: a
      // vehicle can have a fine history and still be unpredictable (one
      // sighting, or a last camera with no observed onward transitions).
      try {
        const forecast = await getJSONOr<PredictionResponse>(
          `/vehicles/${encodeURIComponent(query)}/predict-next-location`,
          {},
        );
        if (stale()) return;
        const destinations = forecast.predicted_destinations ?? [];
        setHops(destinations);
        setLastSeen(forecast.last_sighting ?? null);
        onPredictions(destinations);
      } catch {
        if (stale()) return;
        setHops([]);
        onPredictions([]);
      }
    },
    [onLegs, onPredictions],
  );

  // Auto-load once, from the first plate the live feed produces.
  useEffect(() => {
    if (autoPicked || selected) return;
    const candidate = events.find((e) => e.plate)?.plate;
    if (!candidate) return;
    setAutoPicked(true);
    setPlate(candidate);
    // Publish the auto-pick too, not just the local input. PatternPanel is a
    // sibling and learns which vehicle is on screen only through this, so
    // without it the pattern panel sat on its empty state even though a
    // trajectory was rendered right above it.
    onSelect(candidate);
    load(candidate, windowKey);
  }, [events, autoPicked, selected, load, onSelect, windowKey]);

  // Clicking a sighting in the detection log lands here. Guarded on `active`
  // so re-renders of the parent do not refetch the plate already displayed.
  useEffect(() => {
    if (!selected || selected === active) return;
    setAutoPicked(true);
    setPlate(selected);
    load(selected, windowKey);
  }, [selected, active, load, windowKey]);

  const realHops = legs.filter(
    (l) => l.is_real_road !== false && l.source !== "fallback_straight",
  ).length;

  return (
    <div className="panel" style={{ minHeight: 0 }}>
      <div className="panel-head">
        <span>Trajectory Tracker</span>
        <span className="spacer" />
        {status === "ok" && (
          <span className="green" style={{ letterSpacing: 0 }}>
            ROAD-SNAPPED
          </span>
        )}
        {status === "loading" && <span className="amber">RESOLVING…</span>}
      </div>

      <div style={{ padding: 8, borderBottom: "1px solid var(--panel-border)" }}>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (plate.trim()) onSelect(plate.trim().toUpperCase());
            load(plate, windowKey);
          }}
          style={{ display: "flex", gap: 6 }}
        >
          <input
            type="text"
            value={plate}
            placeholder="PLATE e.g. DL3CAY9019"
            onChange={(e) => setPlate(e.target.value.toUpperCase())}
            spellCheck={false}
          />
          <button className="btn primary" type="submit" disabled={status === "loading"}>
            Track
          </button>
        </form>
      </div>

      <div className="panel-body">
        {active && status === "ok" && (
          <div style={{ padding: "8px 10px", borderBottom: "1px solid var(--panel-border)" }}>
            <div style={{ fontSize: 15, fontWeight: 700 }} className="plate">
              {active}
            </div>
            <div className="muted" style={{ fontSize: 10.5, marginTop: 2 }}>
              {legs.length} hop{legs.length === 1 ? "" : "s"}
              {roadKm != null && ` · ${roadKm.toFixed(2)} km of road`}
              {realHops !== legs.length &&
                ` · ${legs.length - realHops} straight-line fallback`}
            </div>
          </div>
        )}

        {hops.length > 0 && (
          <div
            style={{
              padding: "7px 10px",
              borderBottom: "1px solid var(--panel-border)",
              background: "rgba(247,118,142,0.045)",
            }}
          >
            <div
              className="kpi-label"
              style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}
            >
              <span className="dot" style={{ background: "var(--accent-red)" }} />
              Predicted next hop
              {lastSeen?.camera_id && (
                <span className="muted" style={{ letterSpacing: 0, textTransform: "none" }}>
                  from {lastSeen.camera_id}
                </span>
              )}
            </div>
            {hops.map((hop, rank) => (
              <div
                key={hop.camera_id}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                  fontSize: 10.5,
                  padding: "2px 0",
                }}
              >
                <span
                  style={{
                    width: 26,
                    color: rank === 0 ? "var(--accent-red)" : "var(--accent-purple)",
                    fontWeight: rank === 0 ? 700 : 400,
                  }}
                >
                  {(hop.probability * 100).toFixed(0)}%
                </span>
                {/* probability as a bar, so the ranking reads without arithmetic */}
                <span
                  style={{
                    flex: "0 0 48px",
                    height: 3,
                    background: "var(--panel-border)",
                  }}
                >
                  <span
                    style={{
                      display: "block",
                      height: "100%",
                      width: `${hop.probability * 100}%`,
                      background:
                        rank === 0 ? "var(--accent-red)" : "var(--accent-purple)",
                    }}
                  />
                </span>
                <span className="cyan" style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis" }}>
                  {hop.camera_id}
                </span>
                <span className="muted">
                  {hop.eta_minutes != null ? `${hop.eta_minutes.toFixed(0)}m` : "—"}
                </span>
                {hop.interception_priority === "HIGH" && (
                  <span className="amber" style={{ fontSize: 9 }}>
                    INTERCEPT
                  </span>
                )}
              </div>
            ))}
            <div className="muted" style={{ fontSize: 9.5, marginTop: 3 }}>
              Markov transitions over observed history, direction-filtered. Blinking on
              the map; fastest blink = most likely.
            </div>
          </div>
        )}

        {status === "ok" &&
          legs.map((leg, i) => {
            const real = leg.is_real_road !== false && leg.source !== "fallback_straight";
            return (
              <div
                key={i}
                style={{
                  padding: "6px 10px",
                  borderBottom: "1px solid rgba(30,36,51,0.5)",
                  fontSize: 11,
                }}
              >
                <div>
                  <span className="muted">{String(i + 1).padStart(2, "0")}</span>{" "}
                  <span className="cyan">{leg.from_camera_id}</span>
                  <span className="muted"> → </span>
                  <span className="cyan">{leg.to_camera_id}</span>
                  {!real && (
                    <span className="amber" style={{ fontSize: 9.5 }}>
                      {" "}
                      ⚠ STRAIGHT
                    </span>
                  )}
                </div>
                <div className="muted" style={{ fontSize: 10 }}>
                  {((leg.road_distance_m ?? 0) / 1000).toFixed(2)} km road
                  {leg.road_speed_kmh != null && (
                    <>
                      {" · "}
                      <span
                        className={
                          leg.road_speed_kmh > 60 ? "red" : leg.road_speed_kmh > 40 ? "amber" : ""
                        }
                      >
                        {leg.road_speed_kmh.toFixed(0)} km/h
                      </span>
                    </>
                  )}
                  {leg.straight_speed_kmh != null && leg.road_speed_kmh != null && (
                    <span style={{ opacity: 0.65 }}>
                      {" "}
                      (crow-flies {leg.straight_speed_kmh.toFixed(0)})
                    </span>
                  )}
                </div>
              </div>
            );
          })}

        {status === "idle" && (
          <div className="empty">
            Waiting for the first sighting with a plate, then this loads
            automatically.
            <br />
            <br />
            Or enter a plate above to reconstruct its path across the camera
            network — snapped to real roads via OSRM, not drawn point-to-point.
          </div>
        )}
        {status === "empty" && <div className="empty">{detail}</div>}
        {status === "error" && (
          <div className="empty">
            <span className="red">Routing unavailable.</span>
            <br />
            {detail}
          </div>
        )}
      </div>
    </div>
  );
}
