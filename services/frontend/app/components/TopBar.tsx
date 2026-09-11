"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { API_BASE, getJSON } from "@/lib/api";

/**
 * Two destinations, deliberately. The old build had four pages (dashboard,
 * live, benchmarks, sandbox) and the split meant no single screen ever looked
 * like a working product: cameras on one, an empty trajectory panel on
 * another, static logs on a third. Surveillance is now one live operational
 * view; Sandbox is everything you drive by hand, benchmarks included.
 */
const NAV = [
  { href: "/", label: "Surveillance" },
  { href: "/sandbox", label: "Sandbox" },
];

export default function TopBar() {
  const pathname = usePathname();
  const [online, setOnline] = useState<boolean | null>(null);
  const [clock, setClock] = useState("");

  // Health is polled rather than assumed: during a demo the most useful thing
  // the chrome can tell you is whether the backend is actually reachable.
  useEffect(() => {
    let alive = true;
    const ping = async () => {
      try {
        await getJSON("/");
        if (alive) setOnline(true);
      } catch {
        if (alive) setOnline(false);
      }
    };
    ping();
    const id = setInterval(ping, 10_000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  useEffect(() => {
    const tick = () =>
      setClock(
        new Date().toLocaleTimeString("en-GB", {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
        }),
      );
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <header className="topbar">
      <span className="brand">GODSEYE</span>
      <span className="brand-sub">CITYWIDE VEHICLE INTELLIGENCE</span>

      <nav className="nav">
        {NAV.map((item) => (
          <Link key={item.href} href={item.href}>
            <button
              className={`nav-btn ${pathname === item.href ? "active" : ""}`}
              type="button"
            >
              {item.label}
            </button>
          </Link>
        ))}
      </nav>

      <div className="topbar-right">
        <span>
          <span
            className={`dot ${online === null ? "warn" : online ? "live" : "dead"}`}
            style={{ marginRight: 6 }}
          />
          {online === null ? "CONNECTING" : online ? "API ONLINE" : "API OFFLINE"}
        </span>
        <span className="muted">{API_BASE.replace(/^https?:\/\//, "")}</span>
        <span style={{ color: "var(--text)" }}>{clock}</span>
      </div>
    </header>
  );
}
