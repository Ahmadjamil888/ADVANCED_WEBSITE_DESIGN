import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL("https://zehanx.com"),
  title: {
    default: "zehanx Technologies | Web Development, AI, ML & App Solutions",
    template: "%s | zehanx Technologies",
  },
  description: "zehanx Technologies - 6+ years of excellence in Web Development, AI, Machine Learning, Deep Learning, Neural Networks, Software and App Development. Transform your business with cutting-edge technology solutions.",
  keywords: [
    "web development",
    "AI solutions",
    "machine learning",
    "deep learning",
    "neural networks",
    "software development",
    "app development",
    "mobile apps",
    "artificial intelligence",
    "full-stack development",
    "React",
    "Next.js",
    "Node.js",
    "Python",
    "TensorFlow",
    "PyTorch",
    "enterprise solutions",
    "digital transformation",
    "technology consulting",
  ],
  authors: [{ name: "zehanx Technologies" }],
  creator: "zehanx Technologies",
  publisher: "zehanx Technologies",
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      "max-video-preview": -1,
      "max-image-preview": "large",
      "max-snippet": -1,
    },
  },
  openGraph: {
    type: "website",
    locale: "en_US",
    url: "https://zehanx.com",
    siteName: "zehanx Technologies",
    title: "zehanx Technologies | Web Development, AI, ML & App Solutions",
    description: "Transform your business with cutting-edge Web Development, AI, Machine Learning, and App Development solutions. 6+ years of excellence.",
    images: [
      {
        url: "/logo.png",
        width: 1200,
        height: 630,
        alt: "zehanx Technologies - Web Development, AI, ML & App Solutions",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "zehanx Technologies | Web Development, AI, ML & App Solutions",
    description: "Transform your business with cutting-edge Web Development, AI, Machine Learning, and App Development solutions.",
    images: ["/logo.png"],
    creator: "@zehanxtech",
  },
  alternates: {
    canonical: "https://zehanx.com",
  },
  icons: {
    icon: [
      { url: "/favicon.ico", sizes: "any" },
      { url: "/favicon.ico", type: "image/x-icon" },
    ],
    shortcut: "/favicon.ico",
    apple: "/favicon.ico",
  },
  verification: {
    google: "your-google-verification-code",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className="h-full antialiased"
    >
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
