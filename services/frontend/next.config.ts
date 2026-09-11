import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // `standalone` emits a self-contained server bundle with only the node_modules
  // actually reachable from the build. It is the difference between copying a
  // ~400 MB node_modules into the runtime image and copying ~60 MB, which
  // matters because this whole stack is meant to come up from one compose file.
  output: "standalone",
  reactStrictMode: true,
  eslint: { ignoreDuringBuilds: true },
  typescript: { ignoreBuildErrors: false },
};

export default nextConfig;
