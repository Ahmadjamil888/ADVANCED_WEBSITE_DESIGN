import "@/once-ui/styles/index.scss";
import "@/once-ui/tokens/index.scss";

import classNames from "classnames";

import { Footer, Header, RouteGuard } from "@/components";
import { baseURL, effects, style, font, home } from "@/app/resources";

import { Background, Column, Flex, ThemeProvider, ToastProvider } from "@/once-ui/components";
import { opacity, SpacingToken } from "@/once-ui/types";
import { Meta } from "@/once-ui/modules";
import { AuthProvider } from "@/contexts/AuthContext";

export async function generateMetadata() {
  return {
    title: {
      default: 'zehanxtech - AI Systems Development & Web Solutions',
      template: '%s | zehanxtech'
    },
    description: 'zehanxtech specializes in AI systems development, machine learning solutions, and modern web applications. Building AI for the betterment of humanity from Gujranwala, Pakistan.',
    keywords: [
      'AI development',
      'artificial intelligence',
      'machine learning',
      'web development',
      'AI systems',
      'Pakistan AI company',
      'Gujranwala tech',
      'zehanxtech',
      'custom AI solutions',
      'intelligent automation',
      'Next.js development',
      'React applications',
      'AI consulting'
    ],
    authors: [{ name: 'zehanxtech Team' }],
    creator: 'zehanxtech',
    publisher: 'zehanxtech',
    formatDetection: {
      email: false,
      address: false,
      telephone: false,
    },
    metadataBase: new URL(baseURL),
    alternates: {
      canonical: '/',
    },
    openGraph: {
      title: 'zehanxtech - AI Systems Development & Web Solutions',
      description: 'Building AI for the betterment of humanity. Custom AI systems, machine learning models, and modern web applications.',
      url: baseURL,
      siteName: 'zehanxtech',
      images: [
        {
          url: '/og-image.jpg',
          width: 1200,
          height: 630,
          alt: 'zehanxtech - AI Systems Development Company',
        },
      ],
      locale: 'en_US',
      type: 'website',
    },
    twitter: {
      card: 'summary_large_image',
      title: 'zehanxtech - AI Systems Development & Web Solutions',
      description: 'Building AI for the betterment of humanity. Custom AI systems and web applications.',
      images: ['/og-image.jpg'],
    },
    robots: {
      index: true,
      follow: true,
      googleBot: {
        index: true,
        follow: true,
        'max-video-preview': -1,
        'max-image-preview': 'large',
        'max-snippet': -1,
      },
    },
    verification: {
      google: 'your-google-verification-code',
    },
  };
}

interface RootLayoutProps {
  children: React.ReactNode;
}

export default async function RootLayout({ children }: RootLayoutProps) {
  return (
    <Flex
      suppressHydrationWarning
      as="html"
      lang="en"
      background="page"
      data-neutral={style.neutral}
      data-brand={style.brand}
      data-accent={style.accent}
      data-solid={style.solid}
      data-solid-style={style.solidStyle}
      data-border={style.border}
      data-surface={style.surface}
      data-transition={style.transition}
      className={classNames(
        font.primary.variable,
        font.secondary.variable,
        font.tertiary.variable,
        font.code.variable,
      )}
    >
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="canonical" href={baseURL} />
        <meta name="geo.region" content="PK-PB" />
        <meta name="geo.placename" content="Gujranwala" />
        <meta name="geo.position" content="32.1877;74.1945" />
        <meta name="ICBM" content="32.1877, 74.1945" />
        
        {/* Structured Data for Organization */}
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              "@context": "https://schema.org",
              "@type": "Organization",
              "name": "zehanxtech",
              "alternateName": "zehanx technologies",
              "url": baseURL,
              "logo": `${baseURL}/logo.jpg`,
              "description": "AI systems development and web solutions company building AI for the betterment of humanity",
              "foundingDate": "2024",
              "address": {
                "@type": "PostalAddress",
                "addressLocality": "Gujranwala",
                "addressRegion": "Punjab",
                "addressCountry": "Pakistan"
              },
              "contactPoint": {
                "@type": "ContactPoint",
                "telephone": "+92-344-2693910",
                "contactType": "customer service",
                "email": "zehanxtech@gmail.com",
                "availableLanguage": ["English", "Urdu"]
              },
              "sameAs": [
                "https://github.com/Ahmadjamil888",
                "https://www.linkedin.com/company/zehanxtech",
                "https://www.youtube.com/@zehanxtech"
              ],
              "serviceArea": {
                "@type": "Place",
                "name": "Worldwide"
              },
              "hasOfferCatalog": {
                "@type": "OfferCatalog",
                "name": "AI and Web Development Services",
                "itemListElement": [
                  {
                    "@type": "Offer",
                    "itemOffered": {
                      "@type": "Service",
                      "name": "AI Systems Development",
                      "description": "Custom AI systems, machine learning models, and intelligent automation solutions"
                    }
                  },
                  {
                    "@type": "Offer",
                    "itemOffered": {
                      "@type": "Service",
                      "name": "Web Development",
                      "description": "Modern web applications, responsive design, and full-stack development"
                    }
                  },
                  {
                    "@type": "Offer",
                    "itemOffered": {
                      "@type": "Service",
                      "name": "AI Consulting",
                      "description": "Technology consulting, AI strategy, and digital transformation guidance"
                    }
                  }
                ]
              }
            })
          }}
        />

        {/* Theme Script */}
        <script
          dangerouslySetInnerHTML={{
            __html: `
              (function() {
                try {
                  const theme = localStorage.getItem('theme') || 'system';
                  const root = document.documentElement;
                  if (theme === 'system') {
                    const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
                    root.setAttribute('data-theme', isDark ? 'dark' : 'light');
                  } else {
                    root.setAttribute('data-theme', theme);
                  }
                } catch (e) {
                  document.documentElement.setAttribute('data-theme', 'dark');
                }
              })();
            `,
          }}
        />
      </head>
      <ThemeProvider>
        <ToastProvider>
          <AuthProvider>
            <Column style={{ minHeight: "100vh" }} as="body" fillWidth margin="0" padding="0">
            <Background
              position="fixed"
              mask={{
                x: effects.mask.x,
                y: effects.mask.y,
                radius: effects.mask.radius,
                cursor: effects.mask.cursor
              }}
              gradient={{
                display: effects.gradient.display,
                opacity: effects.gradient.opacity as opacity,
                x: effects.gradient.x,
                y: effects.gradient.y,
                width: effects.gradient.width,
                height: effects.gradient.height,
                tilt: effects.gradient.tilt,
                colorStart: effects.gradient.colorStart,
                colorEnd: effects.gradient.colorEnd,
              }}
              dots={{
                display: effects.dots.display,
                opacity: effects.dots.opacity as opacity,
                size: effects.dots.size as SpacingToken,
                color: effects.dots.color,
              }}
              grid={{
                display: effects.grid.display,
                opacity: effects.grid.opacity as opacity,
                color: effects.grid.color,
                width: effects.grid.width,
                height: effects.grid.height,
              }}
              lines={{
                display: effects.lines.display,
                opacity: effects.lines.opacity as opacity,
                size: effects.lines.size as SpacingToken,
                thickness: effects.lines.thickness,
                angle: effects.lines.angle,
                color: effects.lines.color,
              }}
            />
            <Flex fillWidth minHeight="16" hide="s"></Flex>
            <Header />
            <Flex
              zIndex={0}
              fillWidth
              paddingY="l"
              paddingX="l"
              horizontal="center"
              flex={1}
            >
              <Flex horizontal="center" fillWidth minHeight="0">
                <RouteGuard>{children}</RouteGuard>
              </Flex>
            </Flex>
            <Footer />
          </Column>
          </AuthProvider>
        </ToastProvider>
      </ThemeProvider>
    </Flex>
  );
}
