/**
 * The single place that knows where the backend is.
 *
 * Calls go straight from the browser to FastAPI rather than through a Next.js
 * rewrite. Two reasons: the backend already sets `allow_origins=["*"]`, and
 * the live pages depend on WebSockets, which a rewrite would have to proxy
 * separately anyway. One base URL for both keeps the dev and compose setups
 * identical apart from the value of this variable.
 */
export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") || "http://localhost:8000";

export const WS_BASE = API_BASE.replace(/^http/, "ws");

/** TomTom traffic tiles are optional; absence is a degraded state, not an error. */
export const TOMTOM_KEY = process.env.NEXT_PUBLIC_TOMTOM_API_KEY || "";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

/**
 * GET and parse JSON.
 *
 * Every panel on the surveillance page fetches independently, so one failing
 * endpoint must not blank the others. Callers handle their own errors and the
 * page keeps rendering whatever did arrive.
 */
export async function getJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    cache: "no-store",
    headers: { Accept: "application/json", ...(init?.headers || {}) },
  });
  if (!res.ok) {
    throw new ApiError(`${path} returned HTTP ${res.status}`, res.status);
  }
  return (await res.json()) as T;
}

/** Like getJSON but resolves to `fallback` instead of throwing. */
export async function getJSONOr<T>(path: string, fallback: T): Promise<T> {
  try {
    return await getJSON<T>(path);
  } catch {
    return fallback;
  }
}

// ── shapes we rely on ───────────────────────────────────────────────────────
// Only the fields actually read are declared. The backend returns more, and
// pinning every field here would mean this file has to change every time an
// unrelated column is added.

export interface Camera {
  camera_id: string;
  latitude: number | null;
  longitude: number | null;
  road_name?: string | null;
  direction?: string | null;
  status?: string | null;
}

export interface VehicleEvent {
  event_id: string;
  camera_id: string;
  timestamp: string;
  plate: string | null;
  plate_confidence: number | null;
  vehicle_type: string | null;
  vehicle_color: string | null;
  speed: number | null;
  latitude: number | null;
  longitude: number | null;
  direction: string | null;
}

export interface Alert {
  alert_id?: string;
  id?: string;
  alert_type: string;
  plate: string | null;
  camera_id: string | null;
  message?: string | null;
  severity?: string | null;
  timestamp?: string;
  created_at?: string;
}

export interface AnalyticsSummary {
  total_events?: number;
  unique_vehicles?: number;
  active_cameras?: number;
  active_alerts?: number;
  avg_speed_kmh?: number;
  [k: string]: unknown;
}

export interface HeatmapPoint {
  latitude: number;
  longitude: number;
  count: number;
  camera_id?: string;
  road_name?: string | null;
}

/** One camera-to-camera hop of a trajectory, as `/routing/trajectory` returns it. */
export interface RouteLeg {
  from_camera_id?: string;
  to_camera_id?: string;
  geometry?: [number, number][];
  points?: [number, number][];
  road_distance_m?: number;
  straight_line_m?: number;
  road_speed_kmh?: number | null;
  straight_speed_kmh?: number | null;
  is_real_road?: boolean;
  source?: string;
  from_timestamp?: string;
  to_timestamp?: string;
}

export interface TrajectoryResponse {
  plate?: string;
  legs?: RouteLeg[];
  segments?: RouteLeg[];
  total_road_km?: number;
  total_distance_km?: number;
  snapped?: boolean;
}

/** One predicted next camera, from `/vehicles/{plate}/predict-next-location`. */
export interface NextHop {
  camera_id: string;
  camera_name?: string | null;
  latitude: number;
  longitude: number;
  probability: number;
  distance_km?: number;
  eta_minutes?: number;
  congestion?: string | null;
  interception_priority?: string | null;
}

export interface PredictionResponse {
  plate?: string;
  global_vehicle_id?: string;
  last_sighting?: {
    camera_id?: string;
    timestamp?: string;
    latitude?: number;
    longitude?: number;
    speed_kmh?: number;
    direction?: string;
  };
  predicted_destinations?: NextHop[];
}

/** An inferred significant location from /patterns/{plate}. */
export interface InferredPlace {
  label: string;
  latitude: number;
  longitude: number;
  camera_ids: string[];
  sightings: number;
  confidence: number;
  dominant_camera: string | null;
}

export interface AnomalyDay {
  day: string;
  score: number;
  sightings: number;
  distinct_cameras: number;
  first_hour: number;
  last_hour: number;
  max_speed_kmh: number | null;
  km_from_home: number | null;
  reasons: string[];
}

export interface VehiclePattern {
  plate: string;
  status: "ok" | "insufficient";
  sightings?: number;
  required?: number;
  detail?: string;
  window_days?: number;
  routine?: {
    first_seen?: string;
    last_seen?: string;
    window_days?: number;
    active_days?: number;
    activity_rate?: number;
    total_sightings?: number;
    sightings_per_active_day?: number;
    distinct_cameras?: number;
    corridor_concentration?: number;
    peak_hour?: number;
    busiest_weekday?: number;
    weekend_share?: number;
    top_cameras?: { camera_id: string; sightings: number }[];
  };
  places?: { home: InferredPlace | null; work: InferredPlace | null };
  commute_km?: number | null;
  anomalies?: AnomalyDay[];
  caveats?: string[];
}

/** Pick the first present key — the API is not perfectly consistent across routers. */
export function pick<T>(obj: Record<string, unknown> | undefined, keys: string[], fallback: T): T {
  if (!obj) return fallback;
  for (const key of keys) {
    const value = obj[key];
    if (value !== undefined && value !== null) return value as T;
  }
  return fallback;
}
