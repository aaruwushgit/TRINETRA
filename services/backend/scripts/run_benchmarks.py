#!/usr/bin/env python
"""
Run the benchmark suite and persist the results.

This is what you run ONCE before a demo so the API can serve numbers instantly:

    .venv/bin/python scripts/run_benchmarks.py --suite all

It measures on THIS machine and writes deployments/benchmarks/latest.json, which
GET /benchmarks/ then serves without re-measuring.

Suites:
  hardware  CPU/RAM/GPU/torch profile. Instant.
  anpr      plate detection ms/frame and OCR ms/crop on real media. ~40-60 s
            (model loading dominates; PaddleOCR may download models the first
            time ever, which is excluded from the timings as warmup).
  ingest    events/sec through the real POST /events/ingest + /bulk-ingest.
  query     latency of the dashboard-critical reads at a stated row count.

SAFETY: the ingest and query suites run in a child process pointed at a
throwaway sqlite database. They never write to dev.db. You do not need to set
DATABASE_URL yourself.

The console output separates MEASURED from PROJECTED deliberately — that
distinction is the whole point of the exercise, and a table that blurs it is
worse than no table.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services import benchmark_service as bench  # noqa: E402

WIDTH = 78


def hr(char: str = "─") -> None:
    print(char * WIDTH)


def heading(text: str) -> None:
    print()
    hr("═")
    print(f"  {text}")
    hr("═")


def kv(label: str, value: Any, unit: str = "", indent: int = 2) -> None:
    pad = " " * indent
    shown = "—" if value is None else value
    print(f"{pad}{label:<44} {shown} {unit}".rstrip())


def table(rows: list[list[Any]], headers: list[str]) -> None:
    """Minimal fixed-width table — no third-party dependency for one table."""
    cols = len(headers)
    widths = [len(h) for h in headers]
    str_rows = [["—" if c is None else str(c) for c in row] for row in rows]
    for row in str_rows:
        for i in range(cols):
            widths[i] = max(widths[i], len(row[i]))
    line = "  " + "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("  " + "  ".join("-" * widths[i] for i in range(cols)))
    for row in str_rows:
        print("  " + "  ".join(row[i].ljust(widths[i]) for i in range(cols)))


def val(node: Any) -> Any:
    """Unwrap a measured()/projected() dict to its value."""
    if isinstance(node, dict) and "value" in node:
        return node["value"]
    return node


def dig(data: Any, *path: str) -> Any:
    cur = data
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


# ── printers ─────────────────────────────────────────────────────────────────


def print_hardware(hw: dict[str, Any]) -> None:
    heading("HARDWARE / RUNTIME  (measured)")
    cpu = hw.get("cpu", {})
    mem = hw.get("memory", {})
    acc = hw.get("accelerator", {})
    rt = hw.get("runtime", {})
    kv("CPU", cpu.get("model"))
    kv("Machine model", cpu.get("machine_model"))
    kv("Cores (physical / logical)", f"{cpu.get('physical_cores')} / {cpu.get('logical_cores')}")
    if cpu.get("performance_cores"):
        kv("  of which P-cores / E-cores",
           f"{cpu.get('performance_cores')} / {cpu.get('efficiency_cores')}")
    kv("RAM total", mem.get("total_gb"), "GB")
    kv("OS", hw.get("os", {}).get("platform"))
    kv("Python / torch", f"{hw.get('os', {}).get('python')} / {rt.get('torch')}")
    kv("CUDA usable", acc.get("cuda_available"))
    kv("MPS built / usable", f"{acc.get('mps_built')} / {acc.get('mps_usable')}")
    kv("Device selected for inference", acc.get("selected_device"))
    cm = hw.get("compute_monitor", {})
    kv("compute_monitor module", "available" if cm.get("available") else f"absent ({cm.get('reason')})")


def print_anpr(anpr: dict[str, Any]) -> None:
    heading("ANPR INFERENCE COST  (measured, warm)")
    if "error" in anpr:
        print(f"  suite error: {anpr['error']}")
        return
    kv("Device used", anpr.get("device_used"))
    res = anpr.get("input_resolution", {})
    kv("Source frame", "x".join(str(v) for v in (res.get("frame_wh") or ["?", "?"])))
    kv("YOLO inference size (imgsz)", res.get("yolo_imgsz"))
    print()
    print(f"  {anpr.get('warmup_note', '')}")
    print()

    rows = []
    det = anpr.get("detection", {}).get("stats", {})
    if det:
        rows.append(["plate detection", det.get("n"), det.get("p50_ms"), det.get("p95_ms"),
                     det.get("mean_ms"), det.get("max_ms"), "ms/frame"])
    ocr = anpr.get("ocr", {}).get("stats", {})
    if ocr:
        rows.append(["OCR (PaddleOCR)", ocr.get("n"), ocr.get("p50_ms"), ocr.get("p95_ms"),
                     ocr.get("mean_ms"), ocr.get("max_ms"), "ms/plate-crop"])
    dec = anpr.get("decode", {}).get("stats", {})
    if dec:
        rows.append(["video decode", dec.get("n"), dec.get("p50_ms"), dec.get("p95_ms"),
                     dec.get("mean_ms"), dec.get("max_ms"), "ms/frame"])
    table(rows, ["stage", "n", "p50", "p95", "mean", "max", "unit"])

    print()
    kv("Detection fps (single stream, p50)", dig(anpr, "detection", "fps_single_stream_p50"))
    kv("Plates detected per frame (this clip)",
       val(dig(anpr, "detection", "plates_detected_per_frame_mean")))
    kv("CPU cost per frame", val(dig(anpr, "detection", "cpu_cost",
                                     "process_cpu_seconds_per_frame")), "core-s")
    kv("CPU cost per OCR crop", val(dig(anpr, "ocr", "process_cpu_seconds_per_crop")), "core-s")
    kv("Detector model load", val(dig(anpr, "detection", "model_load_seconds")), "s")
    kv("OCR model load", val(dig(anpr, "ocr", "model_load_seconds")), "s")
    kv("Peak RSS with both models", val(anpr.get("peak_rss_mb")), "MB")
    if anpr.get("ocr", {}).get("sample_reads"):
        kv("Sample OCR reads", ", ".join(anpr["ocr"]["sample_reads"][:4]))
    if anpr.get("detection", {}).get("plates_per_frame_caveat"):
        print(f"      ! {anpr['detection']['plates_per_frame_caveat']}")

    pipe = anpr.get("pipeline_estimate", {})
    if pipe:
        print()
        print("  Single-worker pipeline estimate (DERIVED from the two stages above):")
        for key, node in pipe.items():
            kv(f"  {key}", node.get("value"), "fps", indent=4)

    check = anpr.get("upstream_cross_check", {})
    if check:
        print()
        print("  Cross-check against the upstream repo README (M4 MacBook Air):")
        for label, verdict in check.get("verdict", {}).items():
            if isinstance(verdict, dict):
                kv(f"  {label}",
                   f"ours {verdict['ours_ms']} ms vs ref {verdict['reference_ms']} ms "
                   f"({verdict['ratio_ours_over_reference']}x) — {verdict['assessment']}",
                   indent=4)
                kv("    risk", f"{verdict.get('direction')}; {verdict.get('risk')}", indent=6)
            else:
                kv(f"  {label}", verdict, indent=4)


def print_ingest(ingest: dict[str, Any]) -> None:
    heading("BACKEND INGEST THROUGHPUT  (measured, real endpoints)")
    print(f"  Work per event: {ingest.get('work_per_event')}")
    print(f"  Concurrency:    {ingest.get('concurrency')}")
    print()
    single = ingest.get("single", {})
    st = single.get("stats", {})
    table(
        [["POST /events/ingest (single)", st.get("n"), st.get("p50_ms"), st.get("p95_ms"),
          st.get("mean_ms"), val(single.get("events_per_second"))]],
        ["endpoint", "n", "p50 ms", "p95 ms", "mean ms", "events/s"],
    )
    print()
    rows = []
    for entry in ingest.get("bulk", {}).get("by_batch_size", []):
        rows.append([
            f"bulk-ingest x{entry['batch_size']}",
            entry["repeats"],
            val(entry["per_event_ms_mean"]),
            entry["batch_seconds_mean"],
            val(entry["events_per_second"]),
        ])
    if rows:
        table(rows, ["endpoint", "batches", "ms/event", "s/batch", "events/s"])
    print()
    kv("Best measured throughput", ingest.get("bulk", {}).get("best_events_per_second"), "events/s")
    kv("Rows written during suite", ingest.get("rows_written"))
    print()
    print(f"  NOTE: {single.get('note', '')}")


def print_query(query: dict[str, Any]) -> None:
    heading("DASHBOARD QUERY LATENCY  (measured; row count stated)")
    print(f"  Engine: {query.get('engine')}")
    for scale in query.get("scales", []):
        rows_n = scale["vehicle_event_rows"]
        print()
        print(f"  At {rows_n:,} vehicle_events rows:")
        table(
            [[e["name"], e["http_status"], e["stats"].get("p50_ms"), e["stats"].get("p95_ms"),
              e.get("response_bytes")] for e in scale["endpoints"]],
            ["endpoint", "HTTP", "p50 ms", "p95 ms", "resp bytes"],
        )
    scaling = query.get("scaling", {})
    if scaling.get("endpoints"):
        print()
        print("  How latency grows with row count (DERIVED from the two scales above):")
        table(
            [[e["name"], e["latency_ratio"], e["row_ratio"], e["scaling_exponent_k"], e["verdict"]]
             for e in scaling["endpoints"]],
            ["endpoint", "latency x", "rows x", "k", "verdict"],
        )
        print(f"      ! {scaling.get('two_point_caveat')}")
        print(f"      ! {scaling.get('what_rows_means_here')}")
    impact = query.get("index_impact") or {}
    if impact.get("comparison"):
        print()
        print(f"  Effect of the composite indexes at {impact['rows']:,} rows "
              f"(built in {sum(impact['index_build_ms'].values()):.0f} ms):")
        table(
            [[c["name"], c["p50_ms_without_indexes"], c["p50_ms_with_indexes"], c["speedup"]]
             for c in impact["comparison"]],
            ["endpoint", "p50 no index", "p50 indexed", "speedup"],
        )
        print(f"      ! FINDING: {impact['finding']}")
    print()
    print(f"  CAVEAT: {query.get('row_count_caveat')}")
    print(f"  LIMITATION: {query.get('known_limitation')}")


def print_storage(storage: dict[str, Any]) -> None:
    heading("STORAGE COST PER EVENT  (measured from a real DB file)")
    for label, entry in storage.items():
        bpe = entry.get("bytes_per_event")
        kv(label, f"{entry.get('vehicle_event_rows')} rows / "
                  f"{entry.get('file_bytes', 0) / 1e6:.1f} MB"
                  + (f" -> {val(bpe)} bytes/event" if bpe else ""))
        if entry.get("note"):
            print(f"      {entry['note']}")


def print_projection(name: str, proj: dict[str, Any]) -> None:
    a = proj["assumptions"]
    inf = proj["inference"]
    data = proj["data_tier"]
    net = proj["network_and_architecture"]
    res = proj.get("compute_resources", {})

    heading(f"PROJECTION — {name} cameras  (PROJECTED: arithmetic + assumptions)")
    print("  Assumptions used (all overridable via GET /benchmarks/projection):")
    for key in ("cameras", "fps_per_camera", "ocr_every", "plates_per_frame",
                "vehicles_per_camera_per_hour", "utilisation_headroom",
                "target_latency_s", "gpu_class", "gpu_speedup_vs_measured_device",
                "stream_bitrate_mbps", "retention_days"):
        kv(f"  {key}", a.get(key), indent=4)
    print()
    print("  Inference demand and sizing:")
    kv("  Required detection throughput", val(inf["required_detection_throughput"]), "frames/s", indent=4)
    kv("  Required OCR throughput", val(inf["required_ocr_throughput"]), "crops/s", indent=4)
    kv("  Cost per analysed frame", val(inf["cost_per_analysed_frame"]), "ms", indent=4)
    kv("  Capacity per worker", val(inf["per_worker_capacity"]), "frames/s", indent=4)
    kv("  Workers (measured-class machines)", val(inf["workers_of_measured_class"]), indent=4)
    kv("  Cameras per worker", val(inf["cameras_per_worker"]), indent=4)
    kv(f"  GPUs assumed {a['gpu_class']}", val(inf["gpus"]["gpus_required"]), indent=4)
    kv("  Cameras per GPU", val(inf["gpus"]["cameras_per_gpu"]), indent=4)
    print(f"      ! {inf['gpus']['assumption_warning']}")
    rec = inf.get("sizing_reconciliation")
    if rec:
        print()
        print("  Slots vs physical machines (why 'workers' != 'machines'):")
        kv("  CPU cores consumed per slot", val(rec["cpu_cores_consumed_per_slot"]), indent=4)
        kv("  Slots per host (CPU / RAM limit)",
           f"{rec['slots_limited_by_cpu']} / {rec['slots_limited_by_ram']} "
           f"-> {val(rec['slots_per_host'])} ({rec['binding_constraint']} binds)", indent=4)
        kv("  Physical hosts of measured class",
           val(rec["physical_hosts_of_measured_class"]), indent=4)
        print(f"      ! {val(rec['slots_per_host']) and rec['slots_per_host']['caveat']}")
        print(f"      {rec['binding_constraint_note']}")
    if res:
        print()
        print("  Host resources (inference tier):")
        if "cpu_cores" in res:
            kv("  CPU cores", val(res["cpu_cores"]), indent=4)
        if "ram_gb" in res:
            kv("  RAM", val(res["ram_gb"]), "GB", indent=4)
    print()
    print("  Data tier:")
    kv("  Events into the DB", val(data["events_per_second"]), "events/s", indent=4)
    kv("  Events per day", f"{val(data['events_per_day']):,}", indent=4)
    if "ingest_headroom_ratio" in data:
        kv("  Measured ingest capacity", val(data["measured_ingest_capacity"]), "events/s", indent=4)
        kv("  Headroom vs requirement", val(data["ingest_headroom_ratio"]), "x", indent=4)
        kv("  Ingest processes needed", val(data["ingest_processes_needed"]), indent=4)
    if "storage" in data:
        kv("  Storage growth", val(data["storage"]["per_day_gb"]), "GB/day", indent=4)
        kv("  Storage per month", val(data["storage"]["per_month_gb"]), "GB", indent=4)
        kv(f"  Storage at {a['retention_days']}d retention",
           val(data["storage"]["at_retention_gb"]), "GB", indent=4)
    print()
    print("  Latency:")
    lat = proj["latency"]
    kv("  Measured single-pass cost", val(lat["measured_pipeline_ms"]), "ms", indent=4)
    kv(f"  Within {lat['budget_s']}s budget", lat["fits_budget"], indent=4)
    kv("  Max worker backlog", val(lat["max_queue_depth_frames"]), "frames", indent=4)
    print()
    print("  Edge vs central (the architectural argument):")
    kv("  Central: video uplink", val(net["central_architecture"]["uplink_gbps"]), "Gbps", indent=4)
    kv("  Edge: event uplink", val(net["edge_architecture"]["uplink"]), "Mbps", indent=4)
    kv("  Bandwidth reduction", val(net["bandwidth_reduction_factor"]), "x less for edge", indent=4)
    print(f"      {net['recommendation']}")


def print_worked_example(report: dict[str, Any]) -> None:
    """Show the arithmetic by hand for one configuration.

    A projection nobody can check is a slogan. This prints the multiplication so
    a reviewer can redo it on paper in ten seconds.
    """
    baseline = report.get("baseline", {})
    proj = (report.get("projections") or {}).get("200")
    if not proj:
        return
    a = proj["assumptions"]
    inf = proj["inference"]
    det = baseline.get("detection_ms_p95")
    ocr = baseline.get("ocr_ms_p95")

    heading("WORKED EXAMPLE — check the arithmetic by hand (200 cameras)")
    print(f"  MEASURED on this machine: detection p95 = {det} ms/frame, "
          f"OCR p95 = {ocr} ms/crop")
    print()
    print(f"  1) Aggregate frames to analyse")
    print(f"       {a['cameras']} cameras x {a['fps_per_camera']} analysed fps"
          f" = {val(inf['required_detection_throughput'])} frames/s")
    print(f"  2) OCR crops needed")
    print(f"       {val(inf['required_detection_throughput'])} frames/s / ocr_every {a['ocr_every']}"
          f" x {a['plates_per_frame']} plates/frame"
          f" = {val(inf['required_ocr_throughput'])} crops/s")
    print(f"  3) Cost of one analysed frame on one worker")
    print(f"       {det} ms + ({a['plates_per_frame']}/{a['ocr_every']}) x {ocr} ms"
          f" = {val(inf['cost_per_analysed_frame'])} ms")
    print(f"  4) Usable capacity of one worker at {a['utilisation_headroom']} utilisation")
    print(f"       1000 / {val(inf['cost_per_analysed_frame'])} ms x {a['utilisation_headroom']}"
          f" = {val(inf['per_worker_capacity'])} frames/s")
    print(f"  5) Workers required")
    print(f"       ceil({val(inf['required_detection_throughput'])} / "
          f"{val(inf['per_worker_capacity'])}) = "
          f"{val(inf['workers_of_measured_class'])} machines of the measured class")
    print(f"  6) Slots -> physical machines (a slot uses a fraction of a core)")
    rec = inf.get("sizing_reconciliation")
    if rec:
        print(f"       {val(rec['slots_per_host'])} slots fit one measured-class host "
              f"({rec['binding_constraint']}-bound), so "
              f"ceil({val(inf['workers_of_measured_class'])} / "
              f"{val(rec['slots_per_host'])}) = "
              f"{val(rec['physical_hosts_of_measured_class'])} machines")
    print(f"  7) GPU translation (ASSUMPTION, not measured)")
    print(f"       divide by an assumed {a['gpu_speedup_vs_measured_device']}x speedup for "
          f"{a['gpu_class']} -> {val(inf['gpus']['gpus_required'])} GPUs")
    print()
    print("  Every step above is one multiplication or division. If you disagree")
    print("  with an assumption, change it on /benchmarks/projection and the")
    print("  numbers move — the two MEASURED inputs stay fixed.")


def print_provenance(report: dict[str, Any]) -> None:
    heading("PROVENANCE")
    for key, text in report.get("how_to_read_this", {}).items():
        kv(key, text)
    print()
    kv("Suites completed", ", ".join(report.get("suites_completed", [])) or "none")
    carried = report.get("carried_over_from_previous_run")
    if carried:
        print()
        print("  CARRIED OVER from a previous run (NOT measured just now):")
        for key, meta in carried["sections"].items():
            kv(f"  {key}", f"measured at {meta.get('measured_at')}", indent=4)
    if report.get("errors"):
        print()
        print("  ERRORS (these sections hold no numbers rather than fake ones):")
        for key, err in report["errors"].items():
            kv(f"  {key}", err, indent=4)
    kv("Total run time", report.get("duration_seconds"), "s")
    kv("Persisted to", report.get("persisted_to"))


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure ANPR/ingest/query cost on this machine and project "
                    "a city-scale deployment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n"
               "  .venv/bin/python scripts/run_benchmarks.py --suite all\n"
               "  .venv/bin/python scripts/run_benchmarks.py --suite anpr --frames 120",
    )
    parser.add_argument(
        "--suite", default="all",
        help="all | hardware | anpr | ingest | query (comma-separated for several).",
    )
    parser.add_argument("--out", default=None,
                        help="Also write the JSON report here (latest.json is always updated).")
    parser.add_argument("--no-persist", action="store_true",
                        help="Do not touch deployments/benchmarks/latest.json.")
    parser.add_argument("--frames", type=int, default=None, help="Timed detection frames.")
    parser.add_argument("--ocr-crops", type=int, default=None, help="Timed OCR reads.")
    parser.add_argument("--imgsz", type=int, default=None, help="YOLO inference size.")
    parser.add_argument("--ingest-events", type=int, default=None,
                        help="Single-event ingest requests to time.")
    parser.add_argument("--query-scales", default=None,
                        help="Comma-separated row counts to measure query latency at.")
    parser.add_argument("--quiet", action="store_true", help="Only print the final tables.")
    parser.add_argument("--json", action="store_true", help="Print the raw JSON report instead.")
    args = parser.parse_args()

    requested = [s.strip() for s in args.suite.split(",") if s.strip()]
    suites = list(bench.ALL_SUITES) if "all" in requested else requested
    invalid = [s for s in suites if s not in bench.ALL_SUITES]
    if invalid:
        print(f"Unknown suite(s): {invalid}. Valid: all, {', '.join(bench.ALL_SUITES)}",
              file=sys.stderr)
        return 2

    anpr_cfg = None
    if any(v is not None for v in (args.frames, args.ocr_crops, args.imgsz)):
        d = bench.AnprSuiteConfig()
        anpr_cfg = bench.AnprSuiteConfig(
            frames=args.frames or d.frames,
            ocr_crops=args.ocr_crops or d.ocr_crops,
            imgsz=args.imgsz or d.imgsz,
        )
    db_cfg = None
    if args.ingest_events is not None or args.query_scales is not None:
        d = bench.DbSuiteConfig()
        db_cfg = bench.DbSuiteConfig(
            single_events=args.ingest_events or d.single_events,
            query_scales=tuple(int(x) for x in args.query_scales.split(",")) if args.query_scales
            else d.query_scales,
        )

    t0 = time.perf_counter()
    step = {"n": 0}

    def progress(message: str) -> None:
        step["n"] += 1
        if not args.quiet:
            print(f"  [{time.perf_counter() - t0:6.1f}s] {message}", flush=True)

    if not args.quiet:
        hr("═")
        print("  VEHICLE INTELLIGENCE — BENCHMARK & SCALABILITY SUITE")
        print(f"  suites: {', '.join(suites)}")
        print("  NOTE: ingest/query run against a TEMPORARY sqlite database in a")
        print("        child process. dev.db is never written to.")
        hr("═")

    try:
        report = bench.run_full_suite(
            suites, anpr_cfg, db_cfg, progress=progress, persist=not args.no_persist
        )
    except bench.BenchmarkError as err:
        print(f"\nBenchmark failed: {err}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report, default=str, indent=2))
        return 0

    print_hardware(report.get("hardware", {}))
    m = report.get("measured", {})
    if "anpr" in m:
        print_anpr(m["anpr"])
    if "ingest" in m:
        print_ingest(m["ingest"])
    if "query" in m:
        print_query(m["query"])
    if m.get("storage"):
        print_storage(m["storage"])

    for name, proj in (report.get("projections") or {}).items():
        print_projection(name, proj)

    print_worked_example(report)
    print_provenance(report)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, default=str, indent=2))
        print()
        kv("Extra copy written to", str(out_path))

    print()
    hr("═")
    print("  Serve these numbers: GET /benchmarks/  |  interactive what-if: "
          "GET /benchmarks/projection?cameras=500")
    hr("═")

    # Non-zero exit if a requested suite could not produce numbers — a CI or a
    # pre-demo checklist should notice that, not scroll past it.
    failed = [s for s in suites if s in (report.get("errors") or {})]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
