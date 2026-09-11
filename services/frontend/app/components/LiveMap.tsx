"use client";

import { useEffect, useRef, useState } from "react";
import type { Camera, HeatmapPoint, RouteLeg } from "@/lib/api";
import { TOMTOM_KEY } from "@/lib/api";

/**
 * The map, and the only place Leaflet is touched.
 *
 * Leaflet is imported dynamically inside an effect rather than at module
 * scope: it reaches for `window` on import, which throws during Next's
 * server render. Everything it creates is held in refs and mutated in place,
 * because rebuilding layers on every React render would fight the map's own
 * pan/zoom state and make the view jump while an operator is using it.
 */

const DELHI: [number, number] = [28.6139, 77.209];

export interface LiveMapProps {
  cameras: Camera[];
  /** Per-camera sighting counts, used to size/colour the congestion overlay. */
  heat: HeatmapPoint[];
  /** Cameras that have produced a sighting in the last few seconds. */
  liveCameraIds: Set<string>;
  /** Road-snapped trajectory legs to draw, if a plate is being tracked. */
  legs: RouteLeg[];
  showTraffic: boolean;
  showCameras: boolean;
  showHeat: boolean;
  onCameraClick?: (cameraId: string) => void;
}

/** Leaflet's own types, kept loose — we only ever hold these opaquely. */
type L = typeof import("leaflet");
type LMap = import("leaflet").Map;
type LLayerGroup = import("leaflet").LayerGroup;
type LTileLayer = import("leaflet").TileLayer;

export default function LiveMap({
  cameras,
  heat,
  liveCameraIds,
  legs,
  showTraffic,
  showCameras,
  showHeat,
  onCameraClick,
}: LiveMapProps) {
  const holder = useRef<HTMLDivElement | null>(null);
  const leaflet = useRef<L | null>(null);
  const map = useRef<LMap | null>(null);
  const camLayer = useRef<LLayerGroup | null>(null);
  const heatLayer = useRef<LLayerGroup | null>(null);
  const trajLayer = useRef<LLayerGroup | null>(null);
  const trafficLayer = useRef<LTileLayer | null>(null);
  const [ready, setReady] = useState(false);

  // ── init, once ────────────────────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false;

    (async () => {
      const L = (await import("leaflet")).default;
      if (cancelled || !holder.current || map.current) return;

      leaflet.current = L;
      const m = L.map(holder.current, {
        center: DELHI,
        zoom: 11,
        zoomControl: true,
        attributionControl: true,
        preferCanvas: true, // 266 markers + polylines: canvas beats 266 DOM nodes
      });

      // Dark basemap with no API key, so the map still renders for anyone who
      // has not configured TomTom.
      L.tileLayer(
        "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        {
          subdomains: "abcd",
          maxZoom: 19,
          attribution: "&copy; OpenStreetMap &copy; CARTO",
        },
      ).addTo(m);

      camLayer.current = L.layerGroup().addTo(m);
      heatLayer.current = L.layerGroup().addTo(m);
      trajLayer.current = L.layerGroup().addTo(m);

      map.current = m;
      setReady(true);
    })();

    return () => {
      cancelled = true;
      map.current?.remove();
      map.current = null;
    };
  }, []);

  // ── TomTom live traffic flow ───────────────────────────────────────────────
  // Raster flow tiles rather than the TomTom JS SDK: it is one TileLayer, it
  // needs no extra dependency, and it composites straight onto the dark
  // basemap. `relative0` colours each road by speed relative to its own
  // free-flow, which is the "where is it actually congested" question.
  useEffect(() => {
    const L = leaflet.current;
    const m = map.current;
    if (!ready || !L || !m) return;

    if (trafficLayer.current) {
      m.removeLayer(trafficLayer.current);
      trafficLayer.current = null;
    }
    if (!showTraffic || !TOMTOM_KEY) return;

    const layer = L.tileLayer(
      `https://{s}.api.tomtom.com/traffic/map/4/tile/flow/relative0/{z}/{x}/{y}.png?key=${TOMTOM_KEY}&thickness=6`,
      {
        subdomains: "abcd",
        maxZoom: 22,
        opacity: 0.85,
        attribution: "Traffic &copy; TomTom",
      },
    );
    layer.addTo(m);
    trafficLayer.current = layer;
  }, [ready, showTraffic]);

  // ── camera markers ────────────────────────────────────────────────────────
  useEffect(() => {
    const L = leaflet.current;
    const group = camLayer.current;
    if (!ready || !L || !group) return;

    group.clearLayers();
    if (!showCameras) return;

    for (const cam of cameras) {
      if (cam.latitude == null || cam.longitude == null) continue;
      const live = liveCameraIds.has(cam.camera_id);
      const marker = L.marker([cam.latitude, cam.longitude], {
        icon: L.divIcon({
          className: "",
          html: `<div class="cam-marker ${live ? "live" : ""}"></div>`,
          iconSize: [8, 8],
          iconAnchor: [4, 4],
        }),
        keyboard: false,
      });
      marker.bindTooltip(
        `<b>${cam.camera_id}</b>${cam.road_name ? `<br/>${cam.road_name}` : ""}${
          cam.direction ? `<br/>dir ${cam.direction}` : ""
        }${live ? '<br/><span style="color:#9ece6a">● ACTIVE NOW</span>' : ""}`,
        { direction: "top", opacity: 0.95 },
      );
      if (onCameraClick) marker.on("click", () => onCameraClick(cam.camera_id));
      marker.addTo(group);
    }
    // `onCameraClick` intentionally omitted: callers pass a fresh closure each
    // render, and including it would rebuild 266 markers on every tick.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, cameras, liveCameraIds, showCameras]);

  // ── congestion overlay from our own sightings ─────────────────────────────
  // This is the platform's own data, not TomTom's: counts per camera over the
  // recent window. The two answer different questions and are shown as
  // separate toggles rather than blended into one ambiguous colour.
  useEffect(() => {
    const L = leaflet.current;
    const group = heatLayer.current;
    if (!ready || !L || !group) return;

    group.clearLayers();
    if (!showHeat || heat.length === 0) return;

    const max = Math.max(...heat.map((p) => p.count || 0), 1);
    for (const point of heat) {
      if (point.latitude == null || point.longitude == null) continue;
      const share = (point.count || 0) / max;
      group.addLayer(
        L.circleMarker([point.latitude, point.longitude], {
          radius: 6 + share * 20,
          color: colourFor(share),
          fillColor: colourFor(share),
          fillOpacity: 0.16 + share * 0.32,
          weight: 1,
        }).bindTooltip(
          `<b>${point.camera_id ?? "camera"}</b><br/>${point.count} sightings${
            point.road_name ? `<br/>${point.road_name}` : ""
          }`,
          { direction: "top" },
        ),
      );
    }
  }, [ready, heat, showHeat]);

  // ── trajectory ────────────────────────────────────────────────────────────
  useEffect(() => {
    const L = leaflet.current;
    const group = trajLayer.current;
    const m = map.current;
    if (!ready || !L || !group || !m) return;

    group.clearLayers();
    if (legs.length === 0) return;

    const all: [number, number][] = [];

    legs.forEach((leg, i) => {
      const pts = (leg.geometry || leg.points || []) as [number, number][];
      if (pts.length < 2) return;
      all.push(...pts);

      const real = leg.is_real_road !== false && leg.source !== "fallback_straight";

      // A straight fallback is drawn dashed and amber so it is visually
      // obvious that it is NOT a real road path. Silently drawing it like a
      // route would be the map lying about what it knows.
      L.polyline(pts, {
        color: real ? "#7dcfff" : "#e0af68",
        weight: real ? 3 : 2,
        opacity: 0.9,
        dashArray: real ? undefined : "5,6",
      })
        .bindTooltip(
          `hop ${i + 1}: ${leg.from_camera_id ?? "?"} → ${leg.to_camera_id ?? "?"}` +
            `<br/>road ${((leg.road_distance_m ?? 0) / 1000).toFixed(2)} km` +
            (leg.road_speed_kmh != null
              ? `<br/>road speed ${leg.road_speed_kmh.toFixed(1)} km/h`
              : "") +
            (leg.straight_speed_kmh != null
              ? `<br/><span style="color:#565f89">crow-flies ${leg.straight_speed_kmh.toFixed(1)} km/h</span>`
              : "") +
            (real ? "" : "<br/><b>straight-line fallback</b>"),
          { sticky: true },
        )
        .addTo(group);
    });

    // Endpoints, so direction of travel is readable at a glance.
    if (all.length) {
      const start = all[0];
      const end = all[all.length - 1];
      L.circleMarker(start, {
        radius: 5,
        color: "#9ece6a",
        fillColor: "#9ece6a",
        fillOpacity: 1,
        weight: 1,
      })
        .bindTooltip("first sighting", { direction: "top" })
        .addTo(group);
      L.circleMarker(end, {
        radius: 6,
        color: "#f7768e",
        fillColor: "#f7768e",
        fillOpacity: 1,
        weight: 2,
      })
        .bindTooltip("latest sighting", { direction: "top" })
        .addTo(group);

      m.fitBounds(L.latLngBounds(all).pad(0.18), { animate: true });
    }
  }, [ready, legs]);

  return <div ref={holder} className="map" />;
}

/** Green → amber → red as a camera's share of the busiest camera rises. */
function colourFor(share: number): string {
  if (share > 0.66) return "#f7768e";
  if (share > 0.33) return "#e0af68";
  return "#9ece6a";
}
