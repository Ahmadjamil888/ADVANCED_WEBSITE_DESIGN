'use client'

import Head from 'next/head'
import { siteConfig } from '@/config/site'

interface SEOHeadProps {
  title?: string
  description?: string
  keywords?: string[]
  image?: string
  url?: string
  type?: 'website' | 'article' | 'product'
  noIndex?: boolean
  canonical?: string
  publishedTime?: string
  modifiedTime?: string
  authors?: string[]
}

export function SEOHead({
  title,
  description,
  keywords = [],
  image,
  url,
  type = 'website',
  noIndex = false,
  canonical,
  publishedTime,
  modifiedTime,
  authors = []
}: SEOHeadProps) {
  const seoTitle = title 
    ? `${title} | ${siteConfig.name}`
    : `${siteConfig.name} - Leading AI & Web Development Company`
  
  const seoDescription = description || siteConfig.description
  const seoImage = image || `${siteConfig.url}/og-main.jpg`
  const seoUrl = url || siteConfig.url
  const seoCanonical = canonical || seoUrl

  const allKeywords = [
    ...siteConfig.keywords,
    ...keywords
  ].join(', ')

  return (
    <Head>
      {/* Basic Meta Tags */}
      <title>{seoTitle}</title>
      <meta name="description" content={seoDescription} />
      <meta name="keywords" content={allKeywords} />
      <meta name="author" content={authors.join(', ') || siteConfig.name} />
      <meta name="creator" content={siteConfig.name} />
      <meta name="publisher" content={siteConfig.name} />
      
      {/* Canonical URL */}
      <link rel="canonical" href={seoCanonical} />
      
      {/* Robots */}
      <meta 
        name="robots" 
        content={noIndex ? 'noindex,nofollow' : 'index,follow,max-image-preview:large,max-snippet:-1,max-video-preview:-1'} 
      />
      <meta 
        name="googlebot" 
        content={noIndex ? 'noindex,nofollow' : 'index,follow,max-image-preview:large,max-snippet:-1,max-video-preview:-1'} 
      />
      
      {/* Open Graph */}
      <meta property="og:type" content={type} />
      <meta property="og:title" content={seoTitle} />
      <meta property="og:description" content={seoDescription} />
      <meta property="og:url" content={seoUrl} />
      <meta property="og:site_name" content={siteConfig.name} />
      <meta property="og:image" content={seoImage} />
      <meta property="og:image:width" content="1200" />
      <meta property="og:image:height" content="630" />
      <meta property="og:image:alt" content={seoTitle} />
      <meta property="og:locale" content="en_US" />
      
      {/* Article specific Open Graph */}
      {type === 'article' && publishedTime && (
        <meta property="article:published_time" content={publishedTime} />
      )}
      {type === 'article' && modifiedTime && (
        <meta property="article:modified_time" content={modifiedTime} />
      )}
      {type === 'article' && authors.length > 0 && 
        authors.map((author, index) => (
          <meta key={index} property="article:author" content={author} />
        ))
      }
      
      {/* Twitter Card */}
      <meta name="twitter:card" content="summary_large_image" />
      <meta name="twitter:site" content="@zehanxtech" />
      <meta name="twitter:creator" content="@zehanxtech" />
      <meta name="twitter:title" content={seoTitle} />
      <meta name="twitter:description" content={seoDescription} />
      <meta name="twitter:image" content={seoImage} />
      
      {/* Additional Meta Tags */}
      <meta name="theme-color" content="#000000" />
      <meta name="msapplication-TileColor" content="#000000" />
      <meta name="format-detection" content="telephone=no" />
      
      {/* Preconnect to external domains */}
      <link rel="preconnect" href="https://fonts.googleapis.com" />
      <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      <link rel="preconnect" href="https://www.google-analytics.com" />
      <link rel="preconnect" href="https://www.googletagmanager.com" />
      
      {/* DNS Prefetch */}
      <link rel="dns-prefetch" href="https://cdnjs.cloudflare.com" />
      <link rel="dns-prefetch" href="https://unpkg.com" />
      
      {/* Preload critical resources */}
      <link 
        rel="preload" 
        href="/fonts/inter-var.woff2" 
        as="font" 
        type="font/woff2" 
        crossOrigin="anonymous" 
      />
    </Head>
  )
}