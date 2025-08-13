'use client'

import { useEffect } from 'react'

interface StructuredDataProps {
  data: object | object[]
}

export function StructuredData({ data }: StructuredDataProps) {
  useEffect(() => {
    // Inject structured data into the page
    const script = document.createElement('script')
    script.type = 'application/ld+json'
    script.text = JSON.stringify(Array.isArray(data) ? data : [data])
    document.head.appendChild(script)

    return () => {
      // Cleanup on unmount
      if (script.parentNode) {
        script.parentNode.removeChild(script)
      }
    }
  }, [data])

  return null
}

// Specific structured data components
export function OrganizationStructuredData() {
  const organizationData = {
    "@context": "https://schema.org",
    "@type": "Organization",
    "@id": "https://zehanxtech.com#organization",
    "name": "Zehan X Technologies",
    "url": "https://zehanxtech.com",
    "logo": {
      "@type": "ImageObject",
      "url": "https://zehanxtech.com/logo.png",
      "width": 512,
      "height": 512
    },
    "description": "Leading AI & Web Development Company specializing in Next.js, Machine Learning, and Deep Learning solutions.",
    "foundingDate": "2019",
    "numberOfEmployees": "10-50",
    "industry": "Artificial Intelligence and Web Development",
    "address": {
      "@type": "PostalAddress",
      "addressCountry": "Global",
      "addressRegion": "Worldwide"
    },
    "contactPoint": [
      {
        "@type": "ContactPoint",
        "telephone": "+1-555-AI-TECH",
        "contactType": "customer service",
        "email": "contact@zehanxtech.com",
        "availableLanguage": ["English"],
        "areaServed": "Worldwide"
      }
    ],
    "sameAs": [
      "https://twitter.com/zehanxtech",
      "https://linkedin.com/company/zehanx-technologies",
      "https://github.com/zehanx",
      "https://facebook.com/zehanxtech"
    ],
    "hasOfferCatalog": {
      "@type": "OfferCatalog",
      "name": "AI & Web Development Services",
      "itemListElement": [
        {
          "@type": "Offer",
          "itemOffered": {
            "@type": "Service",
            "name": "AI & Machine Learning Development",
            "description": "Custom AI solutions, predictive analytics, and intelligent automation"
          }
        },
        {
          "@type": "Offer",
          "itemOffered": {
            "@type": "Service",
            "name": "Next.js Development",
            "description": "Modern, fast, and scalable web applications built with Next.js"
          }
        }
      ]
    },
    "aggregateRating": {
      "@type": "AggregateRating",
      "ratingValue": "4.9",
      "reviewCount": "127",
      "bestRating": "5"
    }
  }

  return <StructuredData data={organizationData} />
}

export function WebsiteStructuredData() {
  const websiteData = {
    "@context": "https://schema.org",
    "@type": "WebSite",
    "@id": "https://zehanxtech.com#website",
    "url": "https://zehanxtech.com",
    "name": "Zehan X Technologies",
    "description": "Leading AI & Web Development Company",
    "publisher": {
      "@id": "https://zehanxtech.com#organization"
    },
    "potentialAction": {
      "@type": "SearchAction",
      "target": {
        "@type": "EntryPoint",
        "urlTemplate": "https://zehanxtech.com/search?q={search_term_string}"
      },
      "query-input": "required name=search_term_string"
    }
  }

  return <StructuredData data={websiteData} />
}

export function ServiceStructuredData({ 
  serviceName, 
  description, 
  url, 
  price 
}: {
  serviceName: string
  description: string
  url: string
  price?: string
}) {
  const serviceData = {
    "@context": "https://schema.org",
    "@type": "Service",
    "name": serviceName,
    "description": description,
    "url": url,
    "provider": {
      "@id": "https://zehanxtech.com#organization"
    },
    "areaServed": "Worldwide",
    "serviceType": serviceName,
    ...(price && {
      "offers": {
        "@type": "Offer",
        "price": price,
        "priceCurrency": "USD",
        "availability": "https://schema.org/InStock"
      }
    })
  }

  return <StructuredData data={serviceData} />
}

export function FAQStructuredData({ faqs }: { 
  faqs: Array<{ question: string; answer: string }> 
}) {
  const faqData = {
    "@context": "https://schema.org",
    "@type": "FAQPage",
    "mainEntity": faqs.map(faq => ({
      "@type": "Question",
      "name": faq.question,
      "acceptedAnswer": {
        "@type": "Answer",
        "text": faq.answer
      }
    }))
  }

  return <StructuredData data={faqData} />
}

export function BreadcrumbStructuredData({ 
  items 
}: { 
  items: Array<{ name: string; url: string }> 
}) {
  const breadcrumbData = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    "itemListElement": items.map((item, index) => ({
      "@type": "ListItem",
      "position": index + 1,
      "name": item.name,
      "item": item.url
    }))
  }

  return <StructuredData data={breadcrumbData} />
}

export function ArticleStructuredData({
  title,
  description,
  url,
  image,
  publishedTime,
  modifiedTime,
  authors = ['Zehan X Technologies']
}: {
  title: string
  description: string
  url: string
  image?: string
  publishedTime?: string
  modifiedTime?: string
  authors?: string[]
}) {
  const articleData = {
    "@context": "https://schema.org",
    "@type": "Article",
    "headline": title,
    "description": description,
    "url": url,
    "image": image || "https://zehanxtech.com/og-main.jpg",
    "datePublished": publishedTime,
    "dateModified": modifiedTime || publishedTime,
    "author": authors.map(author => ({
      "@type": "Person",
      "name": author
    })),
    "publisher": {
      "@id": "https://zehanxtech.com#organization"
    },
    "mainEntityOfPage": {
      "@type": "WebPage",
      "@id": url
    }
  }

  return <StructuredData data={articleData} />
}