import "@/app/globals.css";

import type { Metadata } from "next";
import { ThemeProvider } from "@/components/contexts/theme-provider";
import { inter } from "@/lib/fonts";
import { siteConfig } from "../config/site";

export const metadata: Metadata = {
  title: {
    default: "Zehan X Technologies - AI & Web Development Experts",
    template: `%s | Zehan X Technologies`,
  },
  metadataBase: new URL(siteConfig.url),
  description: "Expert AI & web development company. Next.js, machine learning & deep learning solutions. Transform your business with cutting-edge AI technology.",
  keywords: [
    "AI development company",
    "Machine Learning services",
    "Deep Learning solutions",
    "Next.js development agency",
    "Full-stack web development",
    "AI consulting services",
    "Artificial Intelligence solutions",
    "Custom AI models",
    "React development",
    "TypeScript development",
    "AI chatbot development",
    "Data analytics services",
    "Enterprise AI solutions",
    "Predictive analytics",
    "Computer vision",
    "Natural language processing",
    "AI automation",
    "Business intelligence",
    "Zehan X Technologies",
    "Professional AI services",
  ],
  authors: [
    {
      name: "Zehan X Technologies",
      url: "https://zehanx.com",
    },
  ],
  creator: "Zehan X Technologies",
  publisher: "Zehan X Technologies",
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
    url: siteConfig.url,
    title: siteConfig.name,
    description: siteConfig.description,
    siteName: siteConfig.name,
    images: [
      {
        url: siteConfig.ogImage,
        width: 1200,
        height: 630,
        alt: `${siteConfig.name} - AI and Web Development Company`,
        type: "image/jpeg",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: siteConfig.name,
    description: siteConfig.description,
    images: [siteConfig.ogImage],
    creator: "@zehanxtech",
    site: "@zehanxtech",
  },
  icons: {
    icon: [{ url: "/favicon.svg", type: "image/svg+xml" }],
    apple: [{ url: "/apple-touch-icon.svg", sizes: "180x180", type: "image/svg+xml" }],
    shortcut: "/favicon.svg",
  },
  alternates: {
    canonical: siteConfig.url,
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const structuredData = {
    "@context": "https://schema.org",
    "@type": "Organization",
    name: "Zehan X Technologies",
    url: "https://zehanx.com",
    logo: "https://zehanx.com/og.jpg",
    description: siteConfig.description,
    email: "shazabjamildhami@gmail.com",
    foundingDate: "2024",
    industry: "Artificial Intelligence and Web Development",
    numberOfEmployees: "10-50",
    address: {
      "@type": "PostalAddress",
      addressCountry: "Global",
      addressRegion: "Remote",
    },
    contactPoint: {
      "@type": "ContactPoint",
      telephone: "+1-XXX-XXX-XXXX",
      contactType: "customer service",
      email: "shazabjamildhami@gmail.com",
      availableLanguage: "English",
    },
    sameAs: [
      "https://twitter.com/zehanxtech",
      "https://github.com/zehanx",
      "https://linkedin.com/company/zehanx",
    ],
    services: [
      {
        "@type": "Service",
        name: "AI & Machine Learning Development",
        description: "Custom AI solutions, predictive analytics, and intelligent automation",
      },
      {
        "@type": "Service",
        name: "Next.js Development",
        description: "Modern, fast, and scalable web applications built with Next.js",
      },
      {
        "@type": "Service",
        name: "Full-Stack Web Development",
        description: "Complete web solutions from frontend to backend",
      },
      {
        "@type": "Service",
        name: "Deep Learning Solutions",
        description: "Advanced neural networks for complex pattern recognition",
      },
    ],
  };

  const chatbaseScript = `
    (function(){
      if(!window.chatbase || window.chatbase("getState") !== "initialized"){
        window.chatbase = (...args) => {
          if(!window.chatbase.q){window.chatbase.q = []}
          window.chatbase.q.push(args)
        };
        window.chatbase = new Proxy(window.chatbase, {
          get(target, prop) {
            if(prop === "q") return target.q;
            return (...args) => target(prop, ...args);
          }
        });
      }
      const onLoad = function() {
        const script = document.createElement("script");
        script.src = "https://www.chatbase.co/embed.min.js";
        script.id = "fRfzjPpFQjdKijh70A56d";
        script.domain = "www.chatbase.co";
        document.body.appendChild(script);
      };
      if(document.readyState === "complete"){
        onLoad();
      } else {
        window.addEventListener("load", onLoad);
      }
    })();
  `;

  return (
    <html lang="en" style={{ colorScheme: "dark" }} className="dark">
      <head>
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(structuredData) }}
        />
        <script dangerouslySetInnerHTML={{ __html: chatbaseScript }} />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <meta name="theme-color" content="#000000" />
        <meta name="msapplication-TileColor" content="#000000" />

        {/* ✅ Manual favicon link to ensure it's picked up */}
        <link rel="icon" href="/favicon.svg" type="image/svg+xml" />
        <link rel="apple-touch-icon" href="/apple-touch-icon.svg" sizes="180x180" type="image/svg+xml" />
      </head>
      <body className={`${inter.className} bg-background antialiased`}>
        <ThemeProvider>{children}</ThemeProvider>
      </body>
    </html>
  );
}
