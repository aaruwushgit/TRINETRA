"use client";

import { useEffect, useState } from "react";
import { getJSONOr, pick, type AnalyticsSummary } from "@/lib/api";

/**
 * The archive half of the dashboard: a year of history, as aggregates.
 *
 * Deliberately separate from the live feed rather than tabbed behind it,
 * because the two answer different questions and have wildly different costs,
 * and conflating them is how you get a dashboard that looks live but is stale.
 *
 * Everything here reads a *precomputed rollup* — `road_usage`, `camera_totals`,
 * `dataset_kpi` — not `vehicle_events`. That is the whole reason it is usable:
 * measured on 36.9M rows, `/analytics/road-usage` answers in 0.08s and
 * `/analytics/congestion` in 0.68s off the rollups, whereas `/analytics/flow`,
 * which still scans the raw table, takes 17.9s and returns 39,000 rows. The
 * fast ones are used; the slow one is not.
 *
 * The tradeoff is stated on the panel itself: these figures are as fresh as the
 * last rollup refresh, not to-the-second. That is the correct trade for
 * "what has this city done over a year" and the wrong one for "what is
 * happening now" — which is what the live panel is for.
 */

/** Shape as /analytics/road-usage actually returns it — verified against the
 *  live endpoint rather than assumed. An earlier version guessed `road_label`
 *  / `from_road` / `avg_travel_minutes` and every row rendered as "? → ?". */
interface RoadUsage {
  from_camera_id?: string;
  to_camera_id?: string;
  road?: string | null;
  trip_count?: number;
  avg_speed_kmh?: number | null;
  avg_duration_minutes?: number | null;
}

interface CongestionRow {
  camera_id?: string;
  road?: string | null;
  congestion_level?: string | null;
  vehicle_count?: number;
  avg_speed_kmh?: number | null;
}

export default function HistoryPanel({ summary }: { summary: AnalyticsSummary }) {
  const [roads, setRoads] = useState<RoadUsage[]>([]);
  const [congestion, setCongestion] = useState<CongestionRow[]>([]);
  const [loading, setLoading] = useState(true);

  // Fetched once, not polled. The archive does not change while you look at
  // it, and these are the only two endpoints cheap enough to be worth calling
  // at all — polling them would add load for no new information.
  useEffect(() => {
    let alive = true;
    Promise.all([
      getJSONOr<RoadUsage[]>("/analytics/road-usage?limit=8", []),
      getJSONOr<CongestionRow[]>("/analytics/congestion", []),
    ]).then(([r, c]) => {
      if (!alive) return;
      setRoads(Array.isArray(r) ? r : []);
      setCongestion(Array.isArray(c) ? c : []);
      setLoading(false);
    });
    return () => {
      alive = false;
    };
  }, []);

  const total = pick<number>(summary, ["total_detections", "total_events"], 0);
  const unique = pick<number>(summary, ["unique_vehicles"], 0);

  const worst = [...congestion]
    .filter((c) => (c.avg_speed_kmh ?? 99) > 0)
    .sort((a, b) => (a.avg_speed_kmh ?? 99) - (b.avg_speed_kmh ?? 99))
    .slice(0, 6);

  return (
    <div className="panel">
      <div className="panel-head">
        <span className="cyan">▤</span>
        <span>Archive — 365 days</span>
        <span className="spacer" />
        <span className="muted" style={{ letterSpacing: 0 }}>
          {loading ? "loading…" : "from rollups"}
        </span>
      </div>

      <div className="panel-body">
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: "2px 10px",
            padding: "7px 10px",
            borderBottom: "1px solid var(--panel-border)",
            fontSize: 10.5,
          }}
        >
          <Row k="Total sightings" v={fmt(total)} />
          <Row k="Unique vehicles" v={fmt(unique)} />
          <Row k="Segments shown" v={roads.length || "—"} />
          <Row k="Cameras" v={congestion.length || "—"} />
        </div>

        <div style={{ padding: "7px 10px", borderBottom: "1px solid var(--panel-border)" }}>
          <div className="kpi-label" style={{ marginBottom: 4 }}>
            Most-travelled segments (all time)
          </div>
          {roads.length === 0 ? (
            <div className="muted" style={{ fontSize: 10.5 }}>
              {loading ? "…" : "no rollup data"}
            </div>
          ) : (
            roads.map((r, i) => {
              const share = roads[0]?.trip_count
                ? (r.trip_count ?? 0) / (roads[0].trip_count ?? 1)
                : 0;
              return (
                <div key={i} style={{ fontSize: 10.5, padding: "1px 0" }}>
                  <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                    <span className="muted" style={{ width: 52, flex: "0 0 52px" }}>
                      {fmt(r.trip_count ?? 0)}
                    </span>
                    {/* bar, so the ranking reads without arithmetic */}
                    <span style={{ flex: "0 0 40px", height: 3, background: "var(--panel-border)" }}>
                      <span
                        style={{
                          display: "block",
                          height: "100%",
                          width: `${share * 100}%`,
                          background: "var(--accent-cyan)",
                        }}
                      />
                    </span>
                    <span
                      className="cyan"
                      style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
                      title={`${r.from_camera_id ?? "?"} → ${r.to_camera_id ?? "?"}`}
                    >
                      {r.road || `${r.from_camera_id ?? "?"} → ${r.to_camera_id ?? "?"}`}
                      {r.avg_duration_minutes != null && (
                        <span className="muted"> · {r.avg_duration_minutes.toFixed(1)}m</span>
                      )}
                    </span>
                    <span className={speedTone(r.avg_speed_kmh)}>
                      {r.avg_speed_kmh != null ? `${r.avg_speed_kmh.toFixed(0)}` : "—"}
                    </span>
                  </div>
                </div>
              );
            })
          )}
        </div>

        <div style={{ padding: "7px 10px" }}>
          <div className="kpi-label" style={{ marginBottom: 4 }}>
            Slowest junctions (lifetime average)
          </div>
          {worst.length === 0 ? (
            <div className="muted" style={{ fontSize: 10.5 }}>
              {loading ? "…" : "no rollup data"}
            </div>
          ) : (
            worst.map((c) => (
              <div
                key={c.camera_id}
                style={{ display: "flex", gap: 6, fontSize: 10.5, padding: "1px 0" }}
              >
                <span className={speedTone(c.avg_speed_kmh)} style={{ width: 42, flex: "0 0 42px" }}>
                  {c.avg_speed_kmh != null ? `${c.avg_speed_kmh.toFixed(1)}` : "—"}
                </span>
                <span
                  className="muted"
                  style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
                >
                  {c.road || c.camera_id}
                </span>
                {c.congestion_level && (
                  <span className={levelTone(c.congestion_level)} style={{ fontSize: 9 }}>
                    {c.congestion_level}
                  </span>
                )}
              </div>
            ))
          )}
        </div>

        <div
          className="muted"
          style={{
            fontSize: 9.5,
            padding: "6px 10px",
            borderTop: "1px solid var(--panel-border)",
            lineHeight: 1.5,
          }}
        >
          · Read from precomputed rollups, so these are as fresh as the last
          refresh — not to-the-second. That is the right trade for a year of
          history; the live panel is what is exact.
          <br />· km/h figures are averages over the whole window, which is why
          they sit below any individual observed speed.
        </div>
      </div>
    </div>
  );
}

function Row({ k, v }: { k: string; v: string | number }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", gap: 6 }}>
      <span className="muted">{k}</span>
      <span style={{ color: "var(--text)" }}>{v}</span>
    </div>
  );
}

function fmt(n: number): string {
  if (!n) return "0";
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return String(n);
}

function speedTone(s: number | null | undefined): string {
  if (s == null) return "muted";
  if (s < 12) return "red";
  if (s < 20) return "amber";
  return "green";
}

function levelTone(level: string): string {
  const l = level.toUpperCase();
  if (l.includes("HEAVY") || l.includes("SEVERE") || l.includes("HIGH")) return "red";
  if (l.includes("MODERATE") || l.includes("MEDIUM")) return "amber";
  return "green";
}
