import type { Metadata, Viewport } from "next";
import type { ReactNode } from "react";
import "./globals.css";

export const metadata: Metadata = {
  title: "Invite Codes · AlgVault Admin",
  description: "Mobile admin dashboard for invite-code profit-share controls.",
  applicationName: "AlgVault Invite Admin",
  robots: {
    index: false,
    follow: false,
    nocache: true
  },
  appleWebApp: {
    capable: true,
    title: "AV Admin",
    statusBarStyle: "black-translucent",
    startupImage: "/apple-icon.png"
  },
  formatDetection: {
    telephone: false,
    date: false,
    address: false,
    email: false
  },
  icons: {
    icon: [
      { url: "/icon.png", sizes: "192x192", type: "image/png" },
      { url: "/icon-512.png", sizes: "512x512", type: "image/png" }
    ],
    apple: [{ url: "/apple-icon.png", sizes: "180x180", type: "image/png" }]
  }
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
  viewportFit: "cover",
  themeColor: [
    { media: "(prefers-color-scheme: dark)", color: "#030304" },
    { media: "(prefers-color-scheme: light)", color: "#030304" }
  ],
  colorScheme: "dark"
};

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="en" className="bg-[#030304]">
      <body>{children}</body>
    </html>
  );
}
