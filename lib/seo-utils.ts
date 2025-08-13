import { Metadata } from 'next'
import { siteConfig } from '@/config/site'
import { seoConfig } from '@/config/seo'

interface SEOProps {
  title?: string
  description?: string
  keywords?: string[]
  image?: string
  url?: string
  type?: 'website' | 'article' | 'profile'
  publishedTime?: string
  modifiedTime?: string
  authors?: string[]
  section?: string
  tags?: string[]
  noIndex?: boolean
  canonical?: string
}

export function generateSEO({
  title,
  description,
  keywords = [],
  image,
  url,
  type = 'website',
  publishedTime,
  modifiedTime,
  authors = [],
  section,
  tags = [],
  noIndex = false,
  canonical
}: SEOProps = {}): Metadata {
  const seoTitle = title 
    ? `${title} | ${siteConfig.name}`
    : seoConfig.defaultTitle

  const seoDescription = description || siteConfig.description
  const seoImage = image || `${siteConfig.url}/og-main.jpg`
  const seoUrl = url || siteConfig.url
  const seoCanonical = canonical || seoUrl

  // Combine keywords
  const allKeywords = [
    ...siteConfig.keywords,
    ...keywords,
    ...tags
  ].filter((keyword, index, array) => array.indexOf(keyword) === index)

  const metadata: Metadata = {
    title: seoTitle,
    description: seoDescription,
    keywords: allKeywords,
    authors: authors.length > 0 
      ? authors.map(author => ({ name: author }))
      : [{ name: siteConfig.name, url: siteConfig.url }],
    creator: siteConfig.name,
    publisher: siteConfig.name,
    robots: noIndex 
      ? { index: false, follow: false }
      : {
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
    openGraph: {
      type,
      locale: 'en_US',
      url: seoUrl,
      title: seoTitle,
      description: seoDescription,
      siteName: siteConfig.name,
      images: [
        {
          url: seoImage,
          width: 1200,
          height: 630,
          alt: seoTitle,
          type: 'image/jpeg',
        },
      ],
      ...(publishedTime && { publishedTime }),
      ...(modifiedTime && { modifiedTime }),
      ...(authors.length > 0 && { authors }),
      ...(section && { section }),
      ...(tags.length > 0 && { tags }),
    },
    twitter: {
      card: 'summary_large_image',
      title: seoTitle,
      description: seoDescription,
      images: [seoImage],
      creator: '@zehanxtech',
      site: '@zehanxtech',
    },
    alternates: {
      canonical: seoCanonical,
    },
  }

  return metadata
}

export function generateArticleSEO({
  title,
  description,
  keywords = [],
  image,
  url,
  publishedTime,
  modifiedTime,
  authors = [],
  section = 'Technology',
  tags = []
}: Omit<SEOProps, 'type'>) {
  return generateSEO({
    title,
    description,
    keywords,
    image,
    url,
    type: 'article',
    publishedTime,
    modifiedTime,
    authors,
    section,
    tags
  })
}

export function generateServiceSEO({
  serviceName,
  description,
  keywords = [],
  image,
  url
}: {
  serviceName: string
  description?: string
  keywords?: string[]
  image?: string
  url?: string
}) {
  const serviceKeywords = [
    `${serviceName} services`,
    `professional ${serviceName.toLowerCase()}`,
    `${serviceName} company`,
    `${serviceName} experts`,
    `${serviceName} solutions`,
    `custom ${serviceName.toLowerCase()}`,
    `enterprise ${serviceName.toLowerCase()}`,
    ...keywords
  ]

  return generateSEO({
    title: `${serviceName} Services - Professional ${serviceName} Solutions`,
    description: description || `Professional ${serviceName.toLowerCase()} services by Zehan X Technologies. Expert solutions, proven results, and cutting-edge technology.`,
    keywords: serviceKeywords,
    image,
    url,
    type: 'website'
  })
}

export function generateBlogSEO({
  title,
  description,
  keywords = [],
  image,
  slug,
  publishedTime,
  modifiedTime,
  authors = ['Zehan X Technologies'],
  tags = []
}: {
  title: string
  description?: string
  keywords?: string[]
  image?: string
  slug: string
  publishedTime?: string
  modifiedTime?: string
  authors?: string[]
  tags?: string[]
}) {
  const blogKeywords = [
    'AI blog',
    'machine learning insights',
    'web development tips',
    'technology trends',
    'AI tutorials',
    'development best practices',
    ...keywords,
    ...tags
  ]

  return generateArticleSEO({
    title,
    description: description || `${title} - Expert insights and tutorials from Zehan X Technologies on AI, machine learning, and web development.`,
    keywords: blogKeywords,
    image,
    url: `${siteConfig.url}/blog/${slug}`,
    publishedTime,
    modifiedTime,
    authors,
    section: 'Technology Blog',
    tags
  })
}

// Schema.org structured data generators
export function generateOrganizationSchema() {
  return seoConfig.organizationSchema
}

export function generateWebsiteSchema() {
  return seoConfig.websiteSchema
}

export function generateArticleSchema({
  title,
  description,
  url,
  image,
  publishedTime,
  modifiedTime,
  authors = ['Zehan X Technologies'],
  section = 'Technology'
}: {
  title: string
  description: string
  url: string
  image?: string
  publishedTime?: string
  modifiedTime?: string
  authors?: string[]
  section?: string
}) {
  return {
    "@context": "https://schema.org",
    "@type": "Article",
    "headline": title,
    "description": description,
    "url": url,
    "image": image || `${siteConfig.url}/og-main.jpg`,
    "datePublished": publishedTime,
    "dateModified": modifiedTime || publishedTime,
    "author": authors.map(author => ({
      "@type": "Person",
      "name": author
    })),
    "publisher": {
      "@id": `${siteConfig.url}#organization`
    },
    "mainEntityOfPage": {
      "@type": "WebPage",
      "@id": url
    },
    "articleSection": section,
    "inLanguage": "en-US"
  }
}

export function generateServiceSchema({
  serviceName,
  description,
  url,
  image
}: {
  serviceName: string
  description: string
  url: string
  image?: string
}) {
  return {
    "@context": "https://schema.org",
    "@type": "Service",
    "name": serviceName,
    "description": description,
    "url": url,
    "image": image || `${siteConfig.url}/og-main.jpg`,
    "provider": {
      "@id": `${siteConfig.url}#organization`
    },
    "areaServed": "Worldwide",
    "serviceType": serviceName,
    "category": "Technology Services"
  }
}

export function generateBreadcrumbSchema(items: Array<{ name: string; url: string }>) {
  return {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    "itemListElement": items.map((item, index) => ({
      "@type": "ListItem",
      "position": index + 1,
      "name": item.name,
      "item": item.url
    }))
  }
}

// Performance optimization utilities
export function preloadCriticalResources() {
  if (typeof window !== 'undefined') {
    const criticalResources = [
      '/fonts/inter-var.woff2',
      '/images/hero-bg.webp',
      '/images/logo.svg',
      '/og-main.jpg'
    ]

    criticalResources.forEach(resource => {
      const link = document.createElement('link')
      link.rel = 'preload'
      link.href = resource
      link.as = resource.includes('.woff2') ? 'font' : 'image'
      if (resource.includes('.woff2')) {
        link.crossOrigin = 'anonymous'
      }
      document.head.appendChild(link)
    })
  }
}

// Core Web Vitals optimization
export function optimizeImages() {
  if (typeof window !== 'undefined' && 'IntersectionObserver' in window) {
    const imageObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const img = entry.target as HTMLImageElement
          if (img.dataset.src) {
            img.src = img.dataset.src
            img.classList.remove('lazy')
            imageObserver.unobserve(img)
          }
        }
      })
    })

    document.querySelectorAll('img[data-src]').forEach(img => {
      imageObserver.observe(img)
    })
  }
}