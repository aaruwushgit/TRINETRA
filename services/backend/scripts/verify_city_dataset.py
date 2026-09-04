"""
End-to-end verification pass over a city-scale dataset already loaded into
a *running* backend (see scripts/generate_city_dataset.py).

Unlike the other benchmark scripts in this folder, this one talks to the
live HTTP server (not FastAPI's in-process TestClient) — it's checking the
same thing a real camera worker / dashboard would see over the network.

Checks:
  1. Every analytics endpoint responds and reports its latency.
  2. Speed-defaulters report contains no physically-impossible speeds.
  3. Blacklisted "hero" plates actually produced ACTIVE alerts.
  4. Trajectory + ML next-hop prediction work for a real multi-hop vehicle.
  5. Heatmap coverage spans a meaningful fraction of the camera network.

Usage:
  python scripts/verify_city_dataset.py --backend-url http://localhost:8000
"""
from __future__ import annotations

import argparse
import time

import requests


def timed_get(session, url, label):
    t0 = time.perf_counter()
    res = session.get(url, timeout=30)
    ms = (time.perf_counter() - t0) * 1000
    ok = res.status_code == 200
    status = "✓" if ok else "✗"
    print(f"  {status} {label:<38} {ms:7.1f} ms  status={res.status_code}")
    return res.json() if ok else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-url", default="http://localhost:8000")
    args = parser.parse_args()
    base = args.backend_url.rstrip("/")
    s = requests.Session()

    print("=" * 80)
    print("🔎 CITY-SCALE END-TO-END VERIFICATION (live HTTP, not TestClient)")
    print("=" * 80)

    print("\n[1] Analytics endpoint latency + availability")
    summary = timed_get(s, f"{base}/analytics/summary", "GET /analytics/summary")
    cams = timed_get(s, f"{base}/cameras/?deployment=citywide_demo", "GET /cameras (citywide_demo)")
    heatmap = timed_get(s, f"{base}/analytics/heatmap?hours=168", "GET /analytics/heatmap")
    timed_get(s, f"{base}/analytics/density?hours=168", "GET /analytics/density")
    timed_get(s, f"{base}/analytics/speed?hours=168", "GET /analytics/speed")
    timed_get(s, f"{base}/analytics/congestion?minutes=1440", "GET /analytics/congestion")
    od = timed_get(s, f"{base}/analytics/od-matrix?hours=168", "GET /analytics/od-matrix")
    timed_get(s, f"{base}/analytics/flow?hours=72&bucket_minutes=60", "GET /analytics/flow")

    if summary:
        print(f"\n  Total detections: {summary['total_detections']} | "
              f"Unique vehicles: {summary['unique_vehicles']} | "
              f"Avg speed: {summary['avg_speed_kmh']} km/h")
    if cams is not None:
        print(f"  Cameras registered (citywide_demo): {len(cams)}")
    if od is not None:
        print(f"  Origin-Destination pairs discovered: {len(od)}")

    print("\n[2] Speed-defaulters sanity (no physically-impossible speeds)")
    defaulters_res = timed_get(s, f"{base}/vehicles/analytics/speed-defaulters?hours=168", "GET /vehicles/analytics/speed-defaulters")
    if defaulters_res:
        d = defaulters_res["defaulters"]
        max_speed = max((x["effective_speed_kmh"] for x in d), default=0)
        print(f"  Total defaulters: {defaulters_res['total_defaulters']} | "
              f"speed_limit={defaulters_res['speed_limit_kmh']} km/h | "
              f"max reported speed: {max_speed} km/h")
        insane = [x for x in d if x["effective_speed_kmh"] > 150]
        print(f"  {'✓' if not insane else '✗'} Physically-impossible readings (>150 km/h) leaking through: {len(insane)}")

    print("\n[3] Blacklist alerts fired for pre-registered POI vehicles")
    bl = timed_get(s, f"{base}/alerts/blacklist", "GET /alerts/blacklist")
    alerts = timed_get(s, f"{base}/alerts?status=ACTIVE", "GET /alerts")
    if bl and alerts is not None:
        bl_plates = {b["plate"] for b in bl if "HERO" in b["plate"]}
        fired_plates = {a["vehicle_id"] for a in alerts if a["alert_type"] == "BLACKLIST"}
        matched = bl_plates & fired_plates
        print(f"  Hero blacklist plates: {len(bl_plates)} | Fired BLACKLIST alerts for: {len(matched)}")
        print(f"  {'✓' if matched else '✗'} At least one blacklist hit fired a live alert: {bool(matched)}")
        by_type = {}
        for a in alerts:
            by_type[a["alert_type"]] = by_type.get(a["alert_type"], 0) + 1
        print(f"  Active alert breakdown: {by_type}")

    print("\n[4] Trajectory + ML next-hop prediction for a real multi-hop vehicle")
    if bl:
        sample_plate = next(iter(bl))["plate"] if isinstance(bl, list) and bl else None
    else:
        sample_plate = None
    if sample_plate:
        traj = timed_get(s, f"{base}/vehicles/{sample_plate}/trajectory", f"GET /vehicles/{sample_plate}/trajectory")
        if traj:
            print(f"  Trajectory waypoints for {sample_plate}: {len(traj['points'])}")
        pred = timed_get(s, f"{base}/vehicles/{sample_plate}/predict-next-location?top_n=3",
                          f"GET /vehicles/{sample_plate}/predict-next-location")
        if pred and "predicted_destinations" in pred:
            print(f"  Predicted next hops: {[d['camera_id'] for d in pred['predicted_destinations']]}")

    print("\n[5] Heatmap coverage across the camera network")
    if heatmap is not None and cams is not None:
        covered = {p["camera_id"] for p in heatmap}
        total = {c["camera_id"] for c in cams}
        pct = 100 * len(covered & total) / max(1, len(total))
        print(f"  Cameras with heatmap activity: {len(covered & total)}/{len(total)} ({pct:.0f}%)")

    print("\n" + "=" * 80)
    print("✅ VERIFICATION PASS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
