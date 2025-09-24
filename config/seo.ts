import { siteConfig } from './site';

export const seoConfig = {
  // Core SEO Settings
  defaultTitle: "Zehan X Technologies - Leading AI & Web Development Company | 150+ Projects Delivered",
  titleTemplate: "%s | Zehan X Technologies - AI & Web Development Experts",
  defaultDescription: siteConfig.description,
  
  // Advanced Meta Tags
  additionalMetaTags: [
    {
      name: 'viewport',
      content: 'width=device-width, initial-scale=1, maximum-scale=5, user-scalable=yes'
    },
    {
      name: 'format-detection',
      content: 'telephone=no'
    },
    {
      name: 'mobile-web-app-capable',
      content: 'yes'
    },
    {
      name: 'apple-mobile-web-app-capable',
      content: 'yes'
    },
    {
      name: 'apple-mobile-web-app-status-bar-style',
      content: 'black-translucent'
    },
    {
      name: 'apple-mobile-web-app-title',
      content: 'Zehan X Technologies'
    },
    {
      name: 'application-name',
      content: 'Zehan X Technologies'
    },
    {
      name: 'msapplication-TileColor',
      content: '#000000'
    },
    {
      name: 'msapplication-config',
      content: '/browserconfig.xml'
    },
    {
      name: 'theme-color',
      content: '#000000'
    },
    {
      name: 'color-scheme',
      content: 'dark light'
    },
    // Business Information
    {
      name: 'business:contact_data:street_address',
      content: 'Remote-First Global Company'
    },
    {
      name: 'business:contact_data:locality',
      content: 'Worldwide'
    },
    {
      name: 'business:contact_data:region',
      content: 'Global'
    },
    {
      name: 'business:contact_data:postal_code',
      content: '00000'
    },
    {
      name: 'business:contact_data:country_name',
      content: 'Global'
    },
    {
      name: 'business:contact_data:email',
      content: 'contact@zehanxtech.com'
    },
    {
      name: 'business:contact_data:phone_number',
      content: '+1-555-AI-TECH'
    },
    {
      name: 'business:contact_data:website',
      content: siteConfig.url
    },
    // Geo Tags
    {
      name: 'geo.region',
      content: 'Global'
    },
    {
      name: 'geo.placename',
      content: 'Worldwide'
    },
    {
      name: 'ICBM',
      content: '0.0000, 0.0000'
    },
    // Content Tags
    {
      name: 'content-language',
      content: 'en-US'
    },
    {
      name: 'language',
      content: 'English'
    },
    {
      name: 'distribution',
      content: 'global'
    },
    {
      name: 'rating',
      content: 'general'
    },
    {
      name: 'robots',
      content: 'index, follow, max-image-preview:large, max-snippet:-1, max-video-preview:-1'
    },
    {
      name: 'googlebot',
      content: 'index, follow, max-image-preview:large, max-snippet:-1, max-video-preview:-1'
    },
    {
      name: 'bingbot',
      content: 'index, follow'
    },
    // Performance & Technical
    {
      name: 'referrer',
      content: 'origin-when-cross-origin'
    },
    {
      name: 'format-detection',
      content: 'telephone=no, date=no, email=no, address=no'
    }
  ],

  // Open Graph Configuration
  openGraph: {
    type: 'website',
    locale: 'en_US',
    url: siteConfig.url,
    siteName: siteConfig.name,
    title: siteConfig.name,
    description: siteConfig.description,
    images: [
      {
        url: `${siteConfig.url}/og-main.jpg`,
        width: 1200,
        height: 630,
        alt: `${siteConfig.name} - Leading AI & Web Development Company`,
        type: 'image/jpeg',
      },
      {
        url: `${siteConfig.url}/og-square.jpg`,
        width: 1200,
        height: 1200,
        alt: `${siteConfig.name} - AI Development Services`,
        type: 'image/jpeg',
      },
      {
        url: `${siteConfig.url}/og-wide.jpg`,
        width: 1920,
        height: 1080,
        alt: `${siteConfig.name} - Machine Learning Solutions`,
        type: 'image/jpeg',
      }
    ],
  },

  // Twitter Configuration
  twitter: {
    handle: '@zehanxtech',
    site: '@zehanxtech',
    cardType: 'summary_large_image',
    title: siteConfig.name,
    description: siteConfig.shortDescription,
    image: `${siteConfig.url}/twitter-card.jpg`,
  },

  // Additional Social Media
  facebook: {
    appId: '1234567890123456', // Replace with actual Facebook App ID
  },

  // Structured Data Templates
  organizationSchema: {
    "@context": "https://schema.org",
    "@type": "Organization",
    "@id": `${siteConfig.url}#organization`,
    name: siteConfig.name,
    url: siteConfig.url,
    logo: {
      "@type": "ImageObject",
      url: `${siteConfig.url}/logo.png`,
      width: 512,
      height: 512
    },
    image: `${siteConfig.url}/og-main.jpg`,
    description: siteConfig.longDescription,
    email: "contact@zehanxtech.com",
    telephone: "+1-555-AI-TECH",
    foundingDate: siteConfig.company.founded,
    numberOfEmployees: siteConfig.company.employees,
    address: {
      "@type": "PostalAddress",
      addressCountry: "Global",
      addressRegion: "Worldwide",
      addressLocality: "Remote-First"
    },
    contactPoint: [
      {
        "@type": "ContactPoint",
        telephone: "+1-555-AI-TECH",
        contactType: "customer service",
        email: "contact@zehanxtech.com",
        availableLanguage: ["English"],
        areaServed: "Worldwide"
      },
      {
        "@type": "ContactPoint",
        telephone: "+1-555-AI-TECH",
        contactType: "technical support",
        email: "support@zehanxtech.com",
        availableLanguage: ["English"],
        areaServed: "Worldwide"
      },
      {
        "@type": "ContactPoint",
        telephone: "+1-555-AI-TECH",
        contactType: "sales",
        email: "sales@zehanxtech.com",
        availableLanguage: ["English"],
        areaServed: "Worldwide"
      }
    ],
    sameAs: [
      siteConfig.links.twitter,
      siteConfig.links.linkedin,
      siteConfig.links.youtube,
      siteConfig.links.github,
      "https://www.crunchbase.com/organization/zehan-x-technologies",
      "https://www.facebook.com/zehanxtech",
      "https://www.instagram.com/zehanxtech"
    ],
    hasOfferCatalog: {
      "@type": "OfferCatalog",
      name: "AI & Web Development Services",
      itemListElement: siteConfig.services.primary.map((service, index) => ({
        "@type": "Offer",
        itemOffered: {
          "@type": "Service",
          name: service,
          description: `Professional ${service.toLowerCase()} services by Zehan X Technologies`
        }
      }))
    },
    award: siteConfig.company.awards,
    knowsAbout: [
      "Artificial Intelligence",
      "Machine Learning",
      "Deep Learning",
      "Next.js Development",
      "React Development",
      "TypeScript",
      "Full-Stack Development",
      "Data Science",
      "Computer Vision",
      "Natural Language Processing",
      "Predictive Analytics",
      "AI Automation",
      "MLOps",
      "Cloud Computing",
      "API Development"
    ],
    memberOf: {
      "@type": "Organization",
      name: "AI Development Industry Association"
    }
  },

  websiteSchema: {
    "@context": "https://schema.org",
    "@type": "WebSite",
    "@id": `${siteConfig.url}#website`,
    url: siteConfig.url,
    name: siteConfig.name,
    description: siteConfig.description,
    publisher: {
      "@id": `${siteConfig.url}#organization`
    },
    potentialAction: [
      {
        "@type": "SearchAction",
        target: {
          "@type": "EntryPoint",
          urlTemplate: `${siteConfig.url}/search?q={search_term_string}`
        },
        "query-input": "required name=search_term_string"
      }
    ],
    inLanguage: "en-US"
  },

  // FAQ Schema for common questions
  faqSchema: {
    "@context": "https://schema.org",
    "@type": "FAQPage",
    mainEntity: [
      {
        "@type": "Question",
        name: "What AI services does Zehan X Technologies offer?",
        acceptedAnswer: {
          "@type": "Answer",
          text: "We offer comprehensive AI services including custom machine learning model development, deep learning solutions, computer vision, natural language processing, predictive analytics, AI chatbot development, and AI automation systems."
        }
      },
      {
        "@type": "Question", 
        name: "How much does AI development cost?",
        acceptedAnswer: {
          "@type": "Answer",
          text: "AI development costs vary based on project complexity, data requirements, and implementation scope. We offer competitive pricing starting from $10,000 for basic AI solutions to $100,000+ for enterprise-level implementations. Contact us for a detailed quote."
        }
      },
      {
        "@type": "Question",
        name: "How long does it take to develop an AI solution?",
        acceptedAnswer: {
          "@type": "Answer",
          text: "Development timelines range from 4-12 weeks for standard AI solutions to 6+ months for complex enterprise systems. We provide detailed project timelines during our initial consultation."
        }
      },
      {
        "@type": "Question",
        name: "Do you provide ongoing AI model maintenance?",
        acceptedAnswer: {
          "@type": "Answer",
          text: "Yes, we offer comprehensive maintenance and support services including model monitoring, performance optimization, data pipeline management, and regular updates to ensure your AI systems continue performing optimally."
        }
      }
    ]
  }
};

export type SEOConfig = typeof seoConfig;