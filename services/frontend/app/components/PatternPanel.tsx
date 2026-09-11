"use client";

import { useEffect, useState } from "react";
import { getJSONOr, type InferredPlace, type VehiclePattern } from "@/lib/api";

/**
 * What we have inferred about a vehicle, as opposed to what we observed.
 *
 * Kept visually distinct from the trajectory panel above it, because the
 * epistemic status is different: a trajectory is a record of sightings, while
 * home, workplace and "unusual day" are *inferences* from a year of history.
 * The panel says so, and it reports the plain countable facts (active days,
 * corridor concentration) next to the model output so the inferred parts can
 * be sanity-checked against something verifiable.
 */

const DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

export interface PatternPanelProps {
  plate: string | null;
  onPlaces: (places: InferredPlace[]) => void;
}

export default function PatternPanel({ plate, onPlaces }: PatternPanelProps) {
  const [data, setData] = useState<VehiclePattern | null>(null);
  const [loading, setLoading] = useState(false);
  const [unavailable, setUnavailable] = useState(false);

  useEffect(() => {
    if (!plate) {
      setData(null);
      onPlaces([]);
      return;
    }
    let alive = true;
    setLoading(true);
    setUnavailable(false);

    (async () => {
      const result = await getJSONOr<VehiclePattern | null>(
        `/patterns/${encodeURIComponent(plate)}?days=365`,
        null,
      );
      if (!alive) return;
      setLoading(false);
      if (!result) {
        // The router is optional (it is the only thing needing scikit-learn),
        // so "not mounted" is a real state and not an error to shout about.
        setUnavailable(true);
        setData(null);
        onPlaces([]);
        return;
      }
      setData(result);
      const places = [result.places?.home, result.places?.work].filter(
        Boolean,
      ) as InferredPlace[];
      onPlaces(places);
    })();

    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [plate]);

  const routine = data?.routine;

  return (
    <div className="panel">
      <div className="panel-head">
        <span className="purple">◆</span>
        <span>Pattern Analysis</span>
        <span className="spacer" />
        {loading && <span className="amber">MINING…</span>}
        {data?.status === "ok" && (
          <span className="purple" style={{ letterSpacing: 0 }}>
            INFERRED
          </span>
        )}
      </div>

      <div className="panel-body">
        {!plate && (
          <div className="empty">
            Select a vehicle to mine a year of its history — inferred home and
            workplace, how regular it is, and days that broke its own routine.
          </div>
        )}

        {unavailable && (
          <div className="empty">
            <span className="amber">Pattern router unavailable.</span>
            <br />
            It is the only feature that requires scikit-learn, and it is mounted
            optionally so a missing wheel cannot take the API down.
          </div>
        )}

        {data?.status === "insufficient" && (
          <div className="empty">
            <span className="amber">Not enough history.</span>
            <br />
            {data.detail}
            <br />
            <br />
            <span style={{ fontSize: 10, opacity: 0.8 }}>
              Most of the fleet is seen a handful of times a year, so this is the
              common case — reported rather than guessed at.
            </span>
          </div>
        )}

        {data?.status === "ok" && (
          <>
            {/* ── inferred places ─────────────────────────────────────── */}
            <div style={{ padding: "7px 10px", borderBottom: "1px solid var(--panel-border)" }}>
              <div className="kpi-label" style={{ marginBottom: 4 }}>
                Inferred locations
              </div>
              <PlaceRow place={data.places?.home ?? null} label="HOME" colour="var(--accent-green)" />
              <PlaceRow place={data.places?.work ?? null} label="WORK" colour="var(--accent-cyan)" />
              {data.commute_km != null && (
                <div className="muted" style={{ fontSize: 10, marginTop: 3 }}>
                  commute distance{" "}
                  <span className="cyan">{data.commute_km.toFixed(2)} km</span> (straight line)
                </div>
              )}
            </div>

            {/* ── countable facts, for sanity-checking the inferences ── */}
            {routine && (
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
                <Fact
                  k="Active days"
                  v={`${routine.active_days ?? "—"} / ${routine.window_days ?? "—"}`}
                />
                <Fact
                  k="Activity rate"
                  v={
                    routine.activity_rate != null
                      ? `${(routine.activity_rate * 100).toFixed(0)}%`
                      : "—"
                  }
                />
                <Fact k="Sightings" v={(routine.total_sightings ?? 0).toLocaleString()} />
                <Fact k="Per active day" v={routine.sightings_per_active_day ?? "—"} />
                <Fact k="Cameras seen" v={routine.distinct_cameras ?? "—"} />
                <Fact
                  k="Corridor focus"
                  v={
                    routine.corridor_concentration != null
                      ? `${(routine.corridor_concentration * 100).toFixed(0)}%`
                      : "—"
                  }
                  title="Share of all sightings on its top 5 cameras. High = a fixed commute corridor, which is what makes next-hop prediction work for this vehicle."
                />
                <Fact k="Peak hour" v={routine.peak_hour != null ? `${routine.peak_hour}:00` : "—"} />
                <Fact
                  k="Busiest day"
                  v={routine.busiest_weekday != null ? DOW[routine.busiest_weekday] : "—"}
                />
              </div>
            )}

            {/* ── anomalies ───────────────────────────────────────────── */}
            <div style={{ padding: "7px 10px" }}>
              <div className="kpi-label" style={{ marginBottom: 4 }}>
                Days unlike this vehicle&apos;s own routine
              </div>
              {!data.anomalies?.length ? (
                <div className="muted" style={{ fontSize: 10.5 }}>
                  None flagged — either the vehicle is highly regular, or it has
                  fewer than 14 active days, which is the minimum for a baseline.
                </div>
              ) : (
                data.anomalies.map((a) => (
                  <div
                    key={a.day}
                    style={{
                      borderLeft: "2px solid var(--accent-purple)",
                      paddingLeft: 6,
                      marginBottom: 5,
                      fontSize: 10.5,
                    }}
                  >
                    <div>
                      <span className="purple">{a.day}</span>
                      <span className="muted">
                        {" "}
                        · {a.sightings} sightings · {a.distinct_cameras} cameras ·{" "}
                        {a.first_hour}:00–{a.last_hour}:00
                      </span>
                    </div>
                    <div className="muted" style={{ fontSize: 9.5 }}>
                      {a.reasons.join("; ")}
                      {a.km_from_home != null && ` · ${a.km_from_home} km from base`}
                    </div>
                  </div>
                ))
              )}
            </div>

            {data.caveats?.length ? (
              <div
                className="muted"
                style={{
                  fontSize: 9.5,
                  padding: "6px 10px",
                  borderTop: "1px solid var(--panel-border)",
                  lineHeight: 1.5,
                }}
              >
                {data.caveats.map((c, i) => (
                  <div key={i}>· {c}</div>
                ))}
              </div>
            ) : null}
          </>
        )}
      </div>
    </div>
  );
}

function PlaceRow({
  place,
  label,
  colour,
}: {
  place: InferredPlace | null;
  label: string;
  colour: string;
}) {
  if (!place) {
    return (
      <div style={{ fontSize: 10.5 }}>
        <span style={{ color: colour, width: 38, display: "inline-block" }}>{label}</span>
        <span className="muted">no stable cluster found</span>
      </div>
    );
  }
  return (
    <div style={{ fontSize: 10.5 }}>
      <span style={{ color: colour, width: 38, display: "inline-block" }}>{label}</span>
      <span className="cyan">{place.dominant_camera}</span>
      <span className="muted">
        {" "}
        · {place.sightings} sightings · {(place.confidence * 100).toFixed(0)}% of window
      </span>
    </div>
  );
}

function Fact({ k, v, title }: { k: string; v: string | number; title?: string }) {
  return (
    <div title={title} style={{ display: "flex", justifyContent: "space-between", gap: 6 }}>
      <span className="muted">{k}</span>
      <span style={{ color: "var(--text)" }}>{v}</span>
    </div>
  );
}
