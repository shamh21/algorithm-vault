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
    title: "Invite Admin",
    statusBarStyle: "black-translucent"
  },
  icons: {
    icon: "/icon.png",
    apple: "/apple-icon.png"
  }
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
  themeColor: "#030304",
  colorScheme: "dark"
};

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
