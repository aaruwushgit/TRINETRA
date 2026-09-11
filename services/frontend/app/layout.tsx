import type { Metadata } from "next";
import "leaflet/dist/leaflet.css";
import "./globals.css";
import TopBar from "./components/TopBar";

export const metadata: Metadata = {
  title: "GODSEYE — Citywide Vehicle Intelligence",
  description:
    "Citywide ANPR: live camera network, road-snapped trajectories, congestion analytics and enforcement alerts.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="page">
          <TopBar />
          {children}
        </div>
      </body>
    </html>
  );
}
