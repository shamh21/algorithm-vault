import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  poweredByHeader: false,
  reactStrictMode: true,
  turbopack: {
    root: process.cwd()
  },
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          { key: "X-Frame-Options", value: "DENY" },
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "same-origin" }
        ]
      }
    ];
  },
  async rewrites() {
    const backendOrigin = process.env.BACKEND_ORIGIN?.replace(/\/$/, "");
    if (!backendOrigin) return [];
    return [
      { source: "/admin/api/:path*", destination: `${backendOrigin}/admin/api/:path*` },
      { source: "/api/:path*", destination: `${backendOrigin}/api/:path*` },
      { source: "/login", destination: `${backendOrigin}/login` },
      { source: "/logout", destination: `${backendOrigin}/logout` },
      { source: "/setup-2fa", destination: `${backendOrigin}/setup-2fa` },
      { source: "/static/:path*", destination: `${backendOrigin}/static/:path*` },
      { source: "/icons/:path*", destination: `${backendOrigin}/icons/:path*` }
    ];
  }
};

export default nextConfig;
