"use client";

import { useEffect, useRef, useState } from "react";
import maplibregl, { type Map as MLMap, type Marker } from "maplibre-gl";
import type { Camera, HeatmapPoint, InferredPlace, NextHop, RouteLeg } from "@/lib/api";
import { TOMTOM_KEY } from "@/lib/api";

/**
 * 3D map. MapLibre GL rather than Leaflet, and OpenFreeMap rather than Mapbox.
 *
 * Leaflet cannot tilt — it is a 2D raster compositor, so "3D" is not a setting
 * it has. MapLibre renders vector tiles through WebGL, which gives pitch,
 * bearing and true building extrusion from the tile data itself.
 *
 * OpenFreeMap serves the vector tiles: it is free, needs no API key and its
 * `liberty` style carries per-building height, which is what the extrusion
 * layer reads. That keeps the 3D view working for anyone who clones this
 * without signing up for anything — the same reasoning as the TomTom layer
 * being optional.
 *
 * Trajectories are drawn from the geometry the routing service returns, which
 * is OSRM road geometry — so a path follows the carriageway rather than
 * cutting between camera coordinates. Legs that fell back to a straight line
 * are drawn dashed and amber, because a map that renders a guess identically
 * to a measurement is lying.
 */

const DELHI: [number, number] = [77.209, 28.6139]; // MapLibre is lng,lat

/**
 * Basemaps, all key-free.
 *
 * `dark` is the default and the reason is not taste: the previous `liberty`
 * style is a LIGHT basemap, which washed out every overlay — cyan camera dots
 * on near-white streets are close to invisible, and the whole panel read as
 * blank. `dark` uses the same `openmaptiles` vector source, so the 3D building
 * extrusion below keeps working unchanged.
 *
 * Dark Matter is CARTO's, on their own `carto` source, so the extrusion
 * source-layer is resolved at install time rather than hardcoded.
 */
export const STYLES = {
  dark: { label: "Dark", url: "https://tiles.openfreemap.org/styles/dark" },
  matter: {
    label: "Dark Matter",
    url: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
  },
  liberty: { label: "Light", url: "https://tiles.openfreemap.org/styles/liberty" },
} as const;

export type StyleKey = keyof typeof STYLES;

const SRC_CAMERAS = "cameras";
const SRC_HEAT = "congestion";
const SRC_TRAJ = "trajectory";

export interface Map3DProps {
  cameras: Camera[];
  heat: HeatmapPoint[];
  liveCameraIds: Set<string>;
  legs: RouteLeg[];
  /** Predicted next hops for the tracked vehicle. The top one blinks hardest. */
  predictions: NextHop[];
  /** Inferred home/workplace for the tracked vehicle. Static, never blinks —
   *  blinking is reserved for "about to happen", and these are conclusions. */
  places: InferredPlace[];
  showTraffic: boolean;
  showCameras: boolean;
  showHeat: boolean;
  showBuildings: boolean;
  pitch: number;
  styleKey: StyleKey;
}

export default function Map3D({
  cameras,
  heat,
  liveCameraIds,
  legs,
  predictions,
  places,
  showTraffic,
  showCameras,
  showHeat,
  showBuildings,
  pitch,
  styleKey,
}: Map3DProps) {
  const holder = useRef<HTMLDivElement | null>(null);
  const map = useRef<MLMap | null>(null);
  const markers = useRef<Marker[]>([]);
  const placeMarkers = useRef<Marker[]>([]);
  const [ready, setReady] = useState(false);
  const [failed, setFailed] = useState(false);
  // Bumped every time layers are (re)installed. `setStyle` destroys every
  // custom source, so the effects that push camera/congestion/trajectory data
  // must run again afterwards — otherwise the overlays come back EMPTY on a
  // theme switch and the trajectory silently disappears.
  const [styleEpoch, setStyleEpoch] = useState(0);

  // ── init ──────────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!holder.current || map.current) return;

    const m = new maplibregl.Map({
      container: holder.current,
      style: STYLES[styleKey].url,
      center: DELHI,
      zoom: 11,
      pitch,
      bearing: -18,
      // MapLibre v5 moved WebGL attributes here. Antialiasing matters for the
      // building extrusions — without it the rooflines stair-step badly.
      canvasContextAttributes: { antialias: true },
      attributionControl: { compact: true },
    });
    map.current = m;

    m.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "top-left");
    m.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-left");

    // A tile host that is unreachable must degrade to a message, not a blank
    // rectangle that looks like a broken app.
    m.on("error", (e) => {
      if (String(e?.error?.message || "").includes("style")) setFailed(true);
    });

    m.on("load", () => installLayers(m));

    // Changing the basemap destroys every custom source and layer, so they are
    // reinstalled on each style load rather than only on first load.
    m.on("styledata", () => {
      if (m.isStyleLoaded() && !m.getSource(SRC_CAMERAS)) installLayers(m);
    });

    function installLayers(m: MLMap) {
      // ── 3D buildings ────────────────────────────────────────────────────
      // Inserted beneath the style's own label layers so street names stay
      // readable on top of the extrusions.
      const firstSymbol = m
        .getStyle()
        .layers?.find((l) => l.type === "symbol")?.id;

      // Which vector source carries buildings differs per style (openmaptiles
      // vs carto), so it is read from the style rather than assumed.
      const buildingLayer = m
        .getStyle()
        .layers?.find((l) => (l as { "source-layer"?: string })["source-layer"] === "building");
      const buildingSource = (buildingLayer as { source?: string } | undefined)?.source;

      if (buildingSource && !m.getLayer("buildings-3d")) {
        m.addLayer(
          {
            id: "buildings-3d",
            type: "fill-extrusion",
            source: buildingSource,
            "source-layer": "building",
            minzoom: 13,
            paint: {
              // Tokyo Night panel blue, lifting slightly with height so the
              // skyline reads without colour noise.
              "fill-extrusion-color": [
                "interpolate",
                ["linear"],
                ["coalesce", ["get", "render_height"], 6],
                0, "#161b2b",
                40, "#1e2740",
                120, "#2a3557",
              ],
              "fill-extrusion-height": [
                "coalesce", ["get", "render_height"], 6,
              ],
              "fill-extrusion-base": ["coalesce", ["get", "render_min_height"], 0],
              "fill-extrusion-opacity": 0.85,
            },
          },
          firstSymbol,
        );
      }

      // ── congestion (our own sighting counts) ────────────────────────────
      m.addSource(SRC_HEAT, { type: "geojson", data: empty() });
      m.addLayer({
        id: "congestion-blur",
        type: "circle",
        source: SRC_HEAT,
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["get", "share"], 0, 8, 1, 34],
          "circle-color": [
            "interpolate", ["linear"], ["get", "share"],
            0, "#9ece6a", 0.5, "#e0af68", 1, "#f7768e",
          ],
          "circle-opacity": 0.3,
          "circle-blur": 0.6,
        },
      });

      // ── trajectory: real road vs straight-line fallback ─────────────────
      m.addSource(SRC_TRAJ, { type: "geojson", data: empty() });
      m.addLayer({
        id: "traj-glow",
        type: "line",
        source: SRC_TRAJ,
        filter: ["==", ["get", "real"], true],
        layout: { "line-cap": "round", "line-join": "round" },
        paint: { "line-color": "#7dcfff", "line-width": 9, "line-opacity": 0.16, "line-blur": 4 },
      });
      m.addLayer({
        id: "traj-road",
        type: "line",
        source: SRC_TRAJ,
        filter: ["==", ["get", "real"], true],
        layout: { "line-cap": "round", "line-join": "round" },
        paint: { "line-color": "#7dcfff", "line-width": 3.2 },
      });
      m.addLayer({
        id: "traj-fallback",
        type: "line",
        source: SRC_TRAJ,
        filter: ["==", ["get", "real"], false],
        paint: {
          "line-color": "#e0af68",
          "line-width": 2,
          "line-dasharray": [2, 2.5],
          "line-opacity": 0.9,
        },
      });

      // ── cameras ─────────────────────────────────────────────────────────
      m.addSource(SRC_CAMERAS, { type: "geojson", data: empty() });
      m.addLayer({
        id: "cameras-halo",
        type: "circle",
        source: SRC_CAMERAS,
        filter: ["==", ["get", "live"], true],
        paint: {
          "circle-radius": 15,
          "circle-color": "#9ece6a",
          "circle-opacity": 0.3,
          "circle-blur": 0.5,
        },
      });
      m.addLayer({
        id: "cameras-dot",
        type: "circle",
        source: SRC_CAMERAS,
        paint: {
          // Sized up and given a bright ring: 266 nodes on a dark basemap need
          // to be legible at city zoom, not just present.
          "circle-radius": [
            "interpolate", ["linear"], ["zoom"],
            9, ["case", ["get", "live"], 5, 3.6],
            14, ["case", ["get", "live"], 8, 6],
          ],
          "circle-color": ["case", ["get", "live"], "#b9f27c", "#7dcfff"],
          "circle-opacity": 0.95,
          "circle-stroke-width": 1.6,
          "circle-stroke-color": ["case", ["get", "live"], "#eaffd0", "#cfefff"],
        },
      });

      const popup = new maplibregl.Popup({ closeButton: false, offset: 10 });
      m.on("mouseenter", "cameras-dot", (e) => {
        m.getCanvas().style.cursor = "pointer";
        const p = e.features?.[0]?.properties as Record<string, string> | undefined;
        if (!p) return;
        popup
          .setLngLat(e.lngLat)
          .setHTML(
            `<b>${p.camera_id}</b>${p.road_name ? `<br/>${p.road_name}` : ""}${
              p.live === "true" ? '<br/><span style="color:#9ece6a">● ACTIVE NOW</span>' : ""
            }`,
          )
          .addTo(m);
      });
      m.on("mouseleave", "cameras-dot", () => {
        m.getCanvas().style.cursor = "";
        popup.remove();
      });

      setReady(true);
      setStyleEpoch((n) => n + 1);
    }

    return () => {
      markers.current.forEach((mk) => mk.remove());
      markers.current = [];
      placeMarkers.current.forEach((mk) => mk.remove());
      placeMarkers.current = [];
      m.remove();
      map.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── reactive props ────────────────────────────────────────────────────────

  useEffect(() => {
    map.current?.easeTo({ pitch, duration: 600 });
  }, [pitch]);

  // Swapping the basemap. `styledata` above reinstalls the overlays once the
  // new style finishes loading, so the data effects below do not need to know
  // this happened.
  useEffect(() => {
    const m = map.current;
    if (!m || !ready) return;
    const url = STYLES[styleKey]?.url;
    if (url) m.setStyle(url);
  }, [ready, styleKey]);

  useEffect(() => {
    const m = map.current;
    if (!ready || !m || !m.getLayer("buildings-3d")) return;
    m.setLayoutProperty("buildings-3d", "visibility", showBuildings ? "visible" : "none");
  }, [ready, styleEpoch, showBuildings]);

  useEffect(() => {
    const m = map.current;
    if (!ready || !m) return;
    for (const id of ["cameras-dot", "cameras-halo"]) {
      if (m.getLayer(id)) {
        m.setLayoutProperty(id, "visibility", showCameras ? "visible" : "none");
      }
    }
  }, [ready, styleEpoch, showCameras]);

  useEffect(() => {
    const m = map.current;
    if (!ready || !m || !m.getLayer("congestion-blur")) return;
    m.setLayoutProperty("congestion-blur", "visibility", showHeat ? "visible" : "none");
  }, [ready, styleEpoch, showHeat]);

  // TomTom flow, as a raster source composited under the labels.
  useEffect(() => {
    const m = map.current;
    if (!ready || !m) return;
    const id = "tomtom-flow";
    if (m.getLayer(id)) m.removeLayer(id);
    if (m.getSource(id)) m.removeSource(id);
    if (!showTraffic || !TOMTOM_KEY) return;

    m.addSource(id, {
      type: "raster",
      tiles: [
        `https://api.tomtom.com/traffic/map/4/tile/flow/relative0/{z}/{x}/{y}.png?key=${TOMTOM_KEY}&thickness=4`,
      ],
      tileSize: 256,
      attribution: "Traffic © TomTom",
    });
    const firstSymbol = m.getStyle().layers?.find((l) => l.type === "symbol")?.id;
    m.addLayer({ id, type: "raster", source: id, paint: { "raster-opacity": 0.75 } }, firstSymbol);
  }, [ready, styleEpoch, showTraffic]);

  useEffect(() => {
    const m = map.current;
    if (!ready || !m) return;
    const src = m.getSource(SRC_CAMERAS) as maplibregl.GeoJSONSource | undefined;
    src?.setData({
      type: "FeatureCollection",
      features: cameras
        .filter((c) => c.latitude != null && c.longitude != null)
        .map((c) => ({
          type: "Feature" as const,
          geometry: { type: "Point" as const, coordinates: [c.longitude!, c.latitude!] },
          properties: {
            camera_id: c.camera_id,
            road_name: c.road_name ?? "",
            live: liveCameraIds.has(c.camera_id),
          },
        })),
    });
  }, [ready, cameras, liveCameraIds]);

  useEffect(() => {
    const m = map.current;
    if (!ready || !m) return;
    const max = Math.max(...heat.map((p) => p.count || 0), 1);
    const src = m.getSource(SRC_HEAT) as maplibregl.GeoJSONSource | undefined;
    src?.setData({
      type: "FeatureCollection",
      features: heat
        .filter((p) => p.latitude != null && p.longitude != null)
        .map((p) => ({
          type: "Feature" as const,
          geometry: { type: "Point" as const, coordinates: [p.longitude, p.latitude] },
          properties: { share: (p.count || 0) / max, count: p.count },
        })),
    });
  }, [ready, styleEpoch, heat]);

  // ── trajectory ────────────────────────────────────────────────────────────
  useEffect(() => {
    const m = map.current;
    if (!ready || !m) return;
    const src = m.getSource(SRC_TRAJ) as maplibregl.GeoJSONSource | undefined;
    if (!src) return;

    const features = legs
      .map((leg) => {
        const pts = (leg.geometry || leg.points || []) as [number, number][];
        if (pts.length < 2) return null;
        return {
          type: "Feature" as const,
          // The API returns [lat, lng]; GeoJSON wants [lng, lat].
          geometry: {
            type: "LineString" as const,
            coordinates: pts.map(([lat, lng]) => [lng, lat]),
          },
          properties: {
            real: leg.is_real_road !== false && leg.source !== "fallback_straight",
          },
        };
      })
      .filter(Boolean) as GeoJSON.Feature[];

    src.setData({ type: "FeatureCollection", features });

    if (features.length) {
      const bounds = new maplibregl.LngLatBounds();
      for (const f of features) {
        for (const c of (f.geometry as GeoJSON.LineString).coordinates) {
          bounds.extend(c as [number, number]);
        }
      }
      m.fitBounds(bounds, { padding: 80, duration: 900, pitch: m.getPitch(), maxZoom: 15 });
    }
  }, [ready, styleEpoch, legs]);

  // ── blinking next-hop predictions ─────────────────────────────────────────
  // DOM markers rather than a GL layer: a CSS keyframe animation is the
  // simplest honest way to blink, and there are only ever a handful of these.
  // Blink speed and size scale with probability, so the likeliest destination
  // is the one that draws the eye rather than all of them competing.
  useEffect(() => {
    const m = map.current;
    if (!ready || !m) return;

    markers.current.forEach((mk) => mk.remove());
    markers.current = [];
    if (predictions.length === 0) return;

    predictions.forEach((hop, rank) => {
      if (hop.latitude == null || hop.longitude == null) return;

      const el = document.createElement("div");
      el.className = "pred-marker";
      // Faster blink and a wider ring for higher probability.
      const period = 1.9 - Math.min(hop.probability, 1) * 0.95;
      el.style.setProperty("--period", `${period.toFixed(2)}s`);
      el.style.setProperty("--size", `${16 + hop.probability * 26}px`);
      el.classList.add(rank === 0 ? "primary" : "secondary");
      el.innerHTML = `
        <span class="pred-ring"></span>
        <span class="pred-core">${Math.round(hop.probability * 100)}</span>
        <span class="pred-tag">${rank === 0 ? "▲ MOST LIKELY" : `#${rank + 1}`} · ETA ${hop.eta_minutes?.toFixed(0) ?? "?"}m</span>`;

      const marker = new maplibregl.Marker({ element: el, anchor: "center" })
        .setLngLat([hop.longitude, hop.latitude])
        .setPopup(
          new maplibregl.Popup({ offset: 18, closeButton: false }).setHTML(
            `<b>${hop.camera_id}</b><br/>${hop.camera_name ?? ""}` +
              `<br/>probability <b>${(hop.probability * 100).toFixed(1)}%</b>` +
              `<br/>ETA ${hop.eta_minutes?.toFixed(1) ?? "?"} min · ${hop.distance_km?.toFixed(2) ?? "?"} km` +
              `<br/>congestion ${hop.congestion ?? "—"}` +
              (hop.interception_priority
                ? `<br/>intercept priority <b>${hop.interception_priority}</b>`
                : ""),
          ),
        )
        .addTo(m);
      markers.current.push(marker);
    });
  }, [ready, styleEpoch, predictions]);

  // ── inferred home / workplace ─────────────────────────────────────────────
  useEffect(() => {
    const m = map.current;
    if (!ready || !m) return;

    placeMarkers.current.forEach((mk) => mk.remove());
    placeMarkers.current = [];

    for (const place of places) {
      if (place.latitude == null || place.longitude == null) continue;
      const el = document.createElement("div");
      el.className = `place-marker ${place.label}`;
      el.innerHTML = `<span class="place-glyph">${
        place.label === "home" ? "\u2302" : "\u25A0"
      }</span><span class="place-tag">${place.label.toUpperCase()}</span>`;

      placeMarkers.current.push(
        new maplibregl.Marker({ element: el, anchor: "center" })
          .setLngLat([place.longitude, place.latitude])
          .setPopup(
            new maplibregl.Popup({ offset: 16, closeButton: false }).setHTML(
              `<b>inferred ${place.label}</b><br/>${place.dominant_camera ?? ""}` +
                `<br/>${place.sightings} sightings in this window` +
                `<br/>${(place.confidence * 100).toFixed(0)}% of the window's sightings` +
                `<br/><span style="color:#565f89">a neighbourhood junction, not an address</span>`,
            ),
          )
          .addTo(m),
      );
    }
  }, [ready, styleEpoch, places]);

  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
      <div ref={holder} className="map" />
      {failed && (
        <div
          className="empty"
          style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center" }}
        >
          Map tiles unreachable — check network access to tiles.openfreemap.org
        </div>
      )}
    </div>
  );
}

function empty(): GeoJSON.FeatureCollection {
  return { type: "FeatureCollection", features: [] };
}
