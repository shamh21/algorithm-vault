import type { MetadataRoute } from "next";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "AlgVault Invite Admin",
    short_name: "Invite Admin",
    description: "Invite-code profit-share administration for AlgVault.",
    start_url: "/",
    display: "standalone",
    background_color: "#070807",
    theme_color: "#070807",
    icons: [
      {
        src: "/icon.png",
        sizes: "192x192",
        type: "image/png"
      },
      {
        src: "/apple-icon.png",
        sizes: "180x180",
        type: "image/png"
      }
    ]
  };
}
