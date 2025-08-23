import "@/app/globals.css";
import type { Metadata, Viewport } from "next";
import { ClerkProvider } from "@clerk/nextjs";
import { ThemeProvider } from "@/components/contexts/theme-provider";
import { siteConfig } from "../config/site";
import { seoConfig } from "../config/seo";

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 5,
  userScalable: true,
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#ffffff' },
    { media: '(prefers-color-scheme: dark)', color: '#181818' },
  ],
  colorScheme: 'dark',
};

export const metadata: Metadata = {
  ...seoConfig,
  ...siteConfig,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark" style={{ colorScheme: "dark" }}>
      <head>
        {/* Preconnect to CDN styles as in Exein example */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Inter:400,600,700&display=swap" />
        <link rel="stylesheet" href="https://cdn.prod.website-files.com/685fce904fe880c8d21eee20/css/exein.webflow.shared.948934382.min.css" />
        <link rel="stylesheet" href="https://cdn.prod.website-files.com/685fce904fe880c8d21eee20/css/exein.webflow.686baf75259c3dbc3b7bd94a-48711a749.min.css" />
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@splidejs/splide@4.1.4/dist/css/splide.min.css" />
        <link rel="icon" href="/favicon.ico" />
        <link rel="apple-touch-icon" href="/apple-touch-icon.png" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </head>
      <body className="bg-background antialiased dark">
        <ClerkProvider>
          <ThemeProvider>
            <div className="page-wrapper">
              {children}
            </div>
          </ThemeProvider>
        </ClerkProvider>
      </body>
    </html>
  );
}
