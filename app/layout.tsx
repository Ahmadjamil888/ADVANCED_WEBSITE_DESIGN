import "@/app/globals.css";

import type { Metadata, Viewport } from "next";
import { ClerkProvider } from "@clerk/nextjs";
import { ThemeProvider } from "@/components/contexts/theme-provider";
import { inter } from "@/lib/fonts";
import { siteConfig } from "../config/site";
import { seoConfig } from "../config/seo";

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 5,
  userScalable: true,
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#ffffff' },
    { media: '(prefers-color-scheme: dark)', color: '#000000' },
  ],
  colorScheme: 'dark light',
}

export const metadata: Metadata = {
  title: {
    default: seoConfig.defaultTitle,
    template: seoConfig.titleTemplate,
  },
  metadataBase: new URL(siteConfig.url),
  description: siteConfig.description,
  keywords: siteConfig.keywords,
  authors: [
    {
      name: "Zehan X Technologies",
      url: siteConfig.url,
    },
    {
      name: "AI Development Team",
      url: `${siteConfig.url}/about`,
    },
  ],
  creator: "Zehan X Technologies",
  publisher: "Zehan X Technologies",
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  category: 'technology',
  classification: 'AI Development, Web Development, Machine Learning',
  robots: {
    index: true,
    follow: true,
    nocache: false,
    googleBot: {
      index: true,
      follow: true,
      noimageindex: false,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  openGraph: {
    type: 'website',
    locale: 'en_US',
    url: siteConfig.url,
    title: siteConfig.name,
    description: siteConfig.description,
    siteName: siteConfig.name,
    images: seoConfig.openGraph.images,
  },
  twitter: {
    card: 'summary_large_image',
    title: siteConfig.name,
    description: siteConfig.shortDescription,
    images: [`${siteConfig.url}/twitter-card.jpg`],
    creator: '@zehanxtech',
    site: '@zehanxtech',
  },
  icons: {
    icon: [
      { url: '/favicon.ico', sizes: '32x32' },
      { url: '/favicon.svg', type: 'image/svg+xml' },
      { url: '/favicon-16x16.png', sizes: '16x16', type: 'image/png' },
      { url: '/favicon-32x32.png', sizes: '32x32', type: 'image/png' },
    ],
    apple: [
      { url: '/apple-touch-icon.png', sizes: '180x180', type: 'image/png' },
    ],
    shortcut: '/favicon.ico',
  },
  manifest: '/site.webmanifest',
  alternates: {
    canonical: siteConfig.url,
    languages: {
      'en-US': siteConfig.url,
      'en': siteConfig.url,
    },
  },
  verification: {
    google: 'your-google-verification-code', // Replace with actual verification code
    yandex: 'your-yandex-verification-code', // Replace with actual verification code
    yahoo: 'your-yahoo-verification-code', // Replace with actual verification code
    other: {
      'msvalidate.01': 'your-bing-verification-code', // Replace with actual verification code
      'facebook-domain-verification': 'your-facebook-verification-code', // Replace with actual verification code
    },
  },
  appleWebApp: {
    capable: true,
    statusBarStyle: 'black-translucent',
    title: 'Zehan X Technologies',
    startupImage: [
      {
        url: '/apple-startup-640x1136.png',
        media: '(device-width: 320px) and (device-height: 568px) and (-webkit-device-pixel-ratio: 2)',
      },
      {
        url: '/apple-startup-750x1334.png', 
        media: '(device-width: 375px) and (device-height: 667px) and (-webkit-device-pixel-ratio: 2)',
      },
    ],
  },
  applicationName: 'Zehan X Technologies',
  referrer: 'origin-when-cross-origin',
  bookmarks: [`${siteConfig.url}/services`, `${siteConfig.url}/portfolio`, `${siteConfig.url}/contact`],
  archives: [`${siteConfig.url}/blog`],
  assets: [`${siteConfig.url}/assets`],
  generator: 'Next.js',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  // Comprehensive Structured Data
  const structuredDataArray = [
    seoConfig.organizationSchema,
    seoConfig.websiteSchema,
    seoConfig.faqSchema,
    // Breadcrumb Schema
    {
      "@context": "https://schema.org",
      "@type": "BreadcrumbList",
      "itemListElement": [
        {
          "@type": "ListItem",
          "position": 1,
          "name": "Home",
          "item": siteConfig.url
        },
        {
          "@type": "ListItem", 
          "position": 2,
          "name": "Services",
          "item": `${siteConfig.url}/services`
        },
        {
          "@type": "ListItem",
          "position": 3,
          "name": "Portfolio",
          "item": `${siteConfig.url}/portfolio`
        },
        {
          "@type": "ListItem",
          "position": 4,
          "name": "Contact",
          "item": `${siteConfig.url}/contact`
        }
      ]
    },
    // Service Schema
    {
      "@context": "https://schema.org",
      "@type": "Service",
      "name": "AI Development Services",
      "description": "Comprehensive artificial intelligence and machine learning development services",
      "provider": {
        "@id": `${siteConfig.url}#organization`
      },
      "areaServed": "Worldwide",
      "hasOfferCatalog": {
        "@type": "OfferCatalog",
        "name": "AI & Web Development Services",
        "itemListElement": siteConfig.services.primary.map((service, index) => ({
          "@type": "Offer",
          "itemOffered": {
            "@type": "Service",
            "name": service,
            "description": `Professional ${service.toLowerCase()} by industry experts`
          }
        }))
      }
    },
    // Review Schema
    {
      "@context": "https://schema.org",
      "@type": "Organization",
      "@id": `${siteConfig.url}#reviews`,
      "aggregateRating": {
        "@type": "AggregateRating",
        "ratingValue": "4.9",
        "reviewCount": "127",
        "bestRating": "5",
        "worstRating": "1"
      },
      "review": [
        {
          "@type": "Review",
          "reviewRating": {
            "@type": "Rating",
            "ratingValue": "5",
            "bestRating": "5"
          },
          "author": {
            "@type": "Person",
            "name": "Sarah Johnson"
          },
          "reviewBody": "Exceptional AI development services. Zehan X Technologies delivered a custom machine learning solution that increased our efficiency by 40%."
        },
        {
          "@type": "Review", 
          "reviewRating": {
            "@type": "Rating",
            "ratingValue": "5",
            "bestRating": "5"
          },
          "author": {
            "@type": "Person",
            "name": "Michael Chen"
          },
          "reviewBody": "Outstanding Next.js development work. Professional team, excellent communication, and delivered on time."
        }
      ]
    }
  ];

  // Enhanced Analytics and Tracking Scripts
  const analyticsScripts = `
    // Google Analytics 4
    window.dataLayer = window.dataLayer || [];
    function gtag(){dataLayer.push(arguments);}
    gtag('js', new Date());
    gtag('config', 'GA_MEASUREMENT_ID', {
      page_title: document.title,
      page_location: window.location.href,
      content_group1: 'AI Development',
      content_group2: 'Web Development',
      custom_map: {'custom_parameter': 'custom_value'}
    });

    // Enhanced E-commerce Tracking
    gtag('config', 'GA_MEASUREMENT_ID', {
      custom_map: {'custom_parameter': 'custom_value'},
      send_page_view: false
    });

    // Conversion Tracking
    function trackConversion(event_name, value) {
      gtag('event', event_name, {
        'event_category': 'conversion',
        'event_label': 'lead_generation',
        'value': value
      });
    }

    // Performance Monitoring
    window.addEventListener('load', function() {
      gtag('event', 'page_load_time', {
        'event_category': 'performance',
        'event_label': 'load_complete',
        'value': Math.round(performance.now())
      });
    });
  `;

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

  // Core Web Vitals Optimization Script
  const coreWebVitalsScript = `
    // Preload critical resources
    const criticalResources = [
      '/fonts/inter-var.woff2',
      '/images/hero-bg.webp',
      '/images/logo.svg'
    ];
    
    criticalResources.forEach(resource => {
      const link = document.createElement('link');
      link.rel = 'preload';
      link.href = resource;
      link.as = resource.includes('.woff2') ? 'font' : 'image';
      if (resource.includes('.woff2')) link.crossOrigin = 'anonymous';
      document.head.appendChild(link);
    });

    // Lazy load non-critical images
    if ('IntersectionObserver' in window) {
      const imageObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            const img = entry.target;
            img.src = img.dataset.src;
            img.classList.remove('lazy');
            imageObserver.unobserve(img);
          }
        });
      });

      document.querySelectorAll('img[data-src]').forEach(img => {
        imageObserver.observe(img);
      });
    }
  `;

  return (
    <html lang="en" style={{ colorScheme: "dark" }} className="dark">
      <head>
        {/* Structured Data */}
        {structuredDataArray.map((schema, index) => (
          <script
            key={index}
            type="application/ld+json"
            dangerouslySetInnerHTML={{ __html: JSON.stringify(schema) }}
          />
        ))}

        {/* Critical Resource Hints */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link rel="preconnect" href="https://www.google-analytics.com" />
        <link rel="preconnect" href="https://www.googletagmanager.com" />
        <link rel="dns-prefetch" href="https://cdnjs.cloudflare.com" />
        <link rel="dns-prefetch" href="https://unpkg.com" />

        {/* Enhanced Meta Tags */}
        <meta name="format-detection" content="telephone=no, date=no, email=no, address=no" />
        <meta name="mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />
        <meta name="apple-mobile-web-app-title" content="Zehan X Technologies" />
        <meta name="application-name" content="Zehan X Technologies" />
        <meta name="msapplication-TileColor" content="#000000" />
        <meta name="msapplication-config" content="/browserconfig.xml" />

        {/* Business Information */}
        <meta name="business:contact_data:street_address" content="Remote-First Global Company" />
        <meta name="business:contact_data:locality" content="Worldwide" />
        <meta name="business:contact_data:region" content="Global" />
        <meta name="business:contact_data:country_name" content="Global" />
        <meta name="business:contact_data:email" content="contact@zehanxtech.com" />
        <meta name="business:contact_data:phone_number" content="+1-555-AI-TECH" />
        <meta name="business:contact_data:website" content={siteConfig.url} />

        {/* Geo Tags */}
        <meta name="geo.region" content="Global" />
        <meta name="geo.placename" content="Worldwide" />
        <meta name="ICBM" content="0.0000, 0.0000" />

        {/* Content Tags */}
        <meta name="content-language" content="en-US" />
        <meta name="language" content="English" />
        <meta name="distribution" content="global" />
        <meta name="rating" content="general" />
        <meta name="revisit-after" content="1 days" />
        <meta name="expires" content="never" />

        {/* Social Media Meta Tags */}
        <meta property="fb:app_id" content="1234567890123456" />
        <meta property="og:site_name" content={siteConfig.name} />
        <meta property="og:locale" content="en_US" />
        <meta property="og:locale:alternate" content="en_GB" />
        <meta property="article:publisher" content="https://www.facebook.com/zehanxtech" />

        {/* Twitter Card Meta Tags */}
        <meta name="twitter:dnt" content="on" />
        <meta name="twitter:widgets:csp" content="on" />

        {/* Performance Optimization */}
        <link rel="preload" href="/fonts/inter-var.woff2" as="font" type="font/woff2" crossOrigin="anonymous" />
        
        {/* Scripts */}
        <script dangerouslySetInnerHTML={{ __html: coreWebVitalsScript }} />
        <script dangerouslySetInnerHTML={{ __html: analyticsScripts }} />
        <script dangerouslySetInnerHTML={{ __html: chatbaseScript }} />

        {/* Google Analytics */}
        <script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
        
        {/* Microsoft Clarity */}
        <script dangerouslySetInnerHTML={{
          __html: `
            (function(c,l,a,r,i,t,y){
              c[a]=c[a]||function(){(c[a].q=c[a].q||[]).push(arguments)};
              t=l.createElement(r);t.async=1;t.src="https://www.clarity.ms/tag/"+i;
              y=l.getElementsByTagName(r)[0];y.parentNode.insertBefore(t,y);
            })(window, document, "clarity", "script", "CLARITY_PROJECT_ID");
          `
        }} />

        {/* Hotjar */}
        <script dangerouslySetInnerHTML={{
          __html: `
            (function(h,o,t,j,a,r){
              h.hj=h.hj||function(){(h.hj.q=h.hj.q||[]).push(arguments)};
              h._hjSettings={hjid:HOTJAR_ID,hjsv:6};
              a=o.getElementsByTagName('head')[0];
              r=o.createElement('script');r.async=1;
              r.src=t+h._hjSettings.hjid+j+h._hjSettings.hjsv;
              a.appendChild(r);
            })(window,document,'https://static.hotjar.com/c/hotjar-','.js?sv=');
          `
        }} />
      </head>
      <body className={`${inter.className} bg-background antialiased`}>
        <ClerkProvider>
          <ThemeProvider>
            {children}
          </ThemeProvider>
        </ClerkProvider>
        
        {/* Structured Data for Page Performance */}
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              "@context": "https://schema.org",
              "@type": "WebPage",
              "name": "Zehan X Technologies - AI & Web Development",
              "description": siteConfig.description,
              "url": siteConfig.url,
              "mainEntity": {
                "@id": `${siteConfig.url}#organization`
              },
              "breadcrumb": {
                "@id": `${siteConfig.url}#breadcrumb`
              }
            })
          }}
        />
      </body>
    </html>
  );
}
