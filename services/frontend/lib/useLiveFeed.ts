"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  API_BASE,
  WS_BASE,
  getJSONOr,
  type Alert,
  type AnalyticsSummary,
  type Camera,
  type HeatmapPoint,
  type VehicleEvent,
} from "./api";

/**
 * Everything the Surveillance page needs, kept live.
 *
 * The old dashboard's problem was not missing data — it was that each panel
 * loaded once and then sat there, so a screen full of real numbers looked
 * indistinguishable from a mockup. This hook fixes that at the source:
 *
 *  - `/events/recent` is polled on a short interval and diffed, so new
 *    sightings arrive as *new rows* and can be animated rather than silently
 *    replacing the list.
 *  - `liveCameraIds` marks cameras that produced a sighting in the last few
 *    seconds, which is what lets the map show the network actually firing
 *    instead of 266 identical dots.
 *  - alerts arrive pushed over the existing `/ws/alerts` WebSocket.
 *
 * Polling rather than a WebSocket for events is deliberate: the backend only
 * publishes alerts and aggregate stats over WS, not individual sightings, and
 * a 2s poll of an indexed `ORDER BY timestamp DESC LIMIT n` is cheap. Adding a
 * per-sighting WS channel is a backend change, not a frontend one.
 */

const EVENT_POLL_MS = 2000;
const SLOW_POLL_MS = 15000;
const LIVE_CAMERA_TTL_MS = 6000;
const MAX_FEED = 250;

export interface LiveFeed {
  cameras: Camera[];
  events: VehicleEvent[];
  freshIds: Set<string>;
  liveCameraIds: Set<string>;
  alerts: Alert[];
  summary: AnalyticsSummary;
  heat: HeatmapPoint[];
  wsConnected: boolean;
  eventsPerMin: number;
  paused: boolean;
  setPaused: (v: boolean) => void;
}

export function useLiveFeed(): LiveFeed {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [events, setEvents] = useState<VehicleEvent[]>([]);
  const [freshIds, setFreshIds] = useState<Set<string>>(new Set());
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [summary, setSummary] = useState<AnalyticsSummary>({});
  const [heat, setHeat] = useState<HeatmapPoint[]>([]);
  const [wsConnected, setWsConnected] = useState(false);
  const [paused, setPaused] = useState(false);
  const [eventsPerMin, setEventsPerMin] = useState(0);

  // camera_id -> epoch ms of its most recent sighting
  const lastSeen = useRef<Map<string, number>>(new Map());
  const [liveCameraIds, setLiveCameraIds] = useState<Set<string>>(new Set());
  const seenEventIds = useRef<Set<string>>(new Set());
  const arrivals = useRef<number[]>([]);
  const pausedRef = useRef(paused);
  pausedRef.current = paused;

  // ── static-ish: cameras ───────────────────────────────────────────────────
  useEffect(() => {
    let alive = true;
    (async () => {
      const list = await getJSONOr<Camera[]>("/cameras/", []);
      if (alive) setCameras(Array.isArray(list) ? list : []);
    })();
    return () => {
      alive = false;
    };
  }, []);

  // ── fast: the sighting feed ───────────────────────────────────────────────
  useEffect(() => {
    let alive = true;

    const poll = async () => {
      if (pausedRef.current) return;
      const batch = await getJSONOr<VehicleEvent[]>("/events/recent?limit=60", []);
      if (!alive || !Array.isArray(batch)) return;

      const incoming = batch.filter(
        (e) => e.event_id && !seenEventIds.current.has(e.event_id),
      );
      if (incoming.length === 0) return;

      const now = Date.now();
      for (const e of incoming) {
        seenEventIds.current.add(e.event_id);
        if (e.camera_id) lastSeen.current.set(e.camera_id, now);
        arrivals.current.push(now);
      }
      // Bound the dedupe set: this runs for hours during a demo.
      if (seenEventIds.current.size > 4000) {
        seenEventIds.current = new Set(
          Array.from(seenEventIds.current).slice(-2000),
        );
      }

      setEvents((prev) => [...incoming, ...prev].slice(0, MAX_FEED));
      setFreshIds(new Set(incoming.map((e) => e.event_id)));
      // Clear the flash highlight after the CSS animation has played.
      setTimeout(() => alive && setFreshIds(new Set()), 1200);
    };

    poll();
    const id = setInterval(poll, EVENT_POLL_MS);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  // ── which cameras are "hot" right now ─────────────────────────────────────
  useEffect(() => {
    const id = setInterval(() => {
      const cutoff = Date.now() - LIVE_CAMERA_TTL_MS;
      const live = new Set<string>();
      for (const [cam, at] of lastSeen.current) if (at >= cutoff) live.add(cam);
      setLiveCameraIds(live);

      // Arrival rate over the last 60s, for the header.
      const minuteAgo = Date.now() - 60_000;
      arrivals.current = arrivals.current.filter((t) => t >= minuteAgo);
      setEventsPerMin(arrivals.current.length);
    }, 1000);
    return () => clearInterval(id);
  }, []);

  // ── slow: summary, heatmap, alerts ────────────────────────────────────────
  useEffect(() => {
    let alive = true;
    const poll = async () => {
      const [s, h, a] = await Promise.all([
        getJSONOr<AnalyticsSummary>("/analytics/summary", {}),
        getJSONOr<HeatmapPoint[]>("/analytics/heatmap?hours=24", []),
        getJSONOr<Alert[]>("/alerts?status=ACTIVE", []),
      ]);
      if (!alive) return;
      setSummary(s || {});
      if (Array.isArray(h)) setHeat(h);
      if (Array.isArray(a)) setAlerts(a.slice(0, 60));
    };
    poll();
    const id = setInterval(poll, SLOW_POLL_MS);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  // ── pushed alerts ─────────────────────────────────────────────────────────
  // Reconnects with a fixed backoff. The page must survive the backend
  // restarting mid-demo without a manual refresh.
  useEffect(() => {
    let socket: WebSocket | null = null;
    let retry: ReturnType<typeof setTimeout> | null = null;
    let closed = false;

    const connect = () => {
      if (closed) return;
      try {
        socket = new WebSocket(`${WS_BASE}/ws/alerts`);
      } catch {
        retry = setTimeout(connect, 4000);
        return;
      }
      socket.onopen = () => setWsConnected(true);
      socket.onclose = () => {
        setWsConnected(false);
        if (!closed) retry = setTimeout(connect, 4000);
      };
      socket.onerror = () => socket?.close();
      socket.onmessage = (msg) => {
        try {
          const payload = JSON.parse(msg.data);
          const alert: Alert = payload?.alert ?? payload;
          if (alert && alert.alert_type) {
            setAlerts((prev) => [alert, ...prev].slice(0, 60));
          }
        } catch {
          /* a non-JSON frame is not worth breaking the feed over */
        }
      };
    };

    connect();
    return () => {
      closed = true;
      if (retry) clearTimeout(retry);
      socket?.close();
    };
  }, []);

  return {
    cameras,
    events,
    freshIds,
    liveCameraIds,
    alerts,
    summary,
    heat,
    wsConnected,
    eventsPerMin,
    paused,
    setPaused: useCallback((v: boolean) => setPaused(v), []),
  };
}

export { API_BASE };
