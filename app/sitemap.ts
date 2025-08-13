import { MetadataRoute } from 'next'
import { siteConfig } from '@/config/site'

export default function sitemap(): MetadataRoute.Sitemap {
  const baseUrl = siteConfig.url
  const currentDate = new Date()
  
  // Static pages with high priority
  const staticPages = [
    {
      url: baseUrl,
      lastModified: currentDate,
      changeFrequency: 'daily' as const,
      priority: 1.0,
    },
    {
      url: `${baseUrl}/services`,
      lastModified: currentDate,
      changeFrequency: 'weekly' as const,
      priority: 0.9,
    },
    {
      url: `${baseUrl}/about`,
      lastModified: currentDate,
      changeFrequency: 'monthly' as const,
      priority: 0.8,
    },
    {
      url: `${baseUrl}/portfolio`,
      lastModified: currentDate,
      changeFrequency: 'weekly' as const,
      priority: 0.8,
    },
    {
      url: `${baseUrl}/contact`,
      lastModified: currentDate,
      changeFrequency: 'monthly' as const,
      priority: 0.7,
    },
    {
      url: `${baseUrl}/pricing`,
      lastModified: currentDate,
      changeFrequency: 'weekly' as const,
      priority: 0.7,
    },
    {
      url: `${baseUrl}/blog`,
      lastModified: currentDate,
      changeFrequency: 'daily' as const,
      priority: 0.8,
    },
    {
      url: `${baseUrl}/zehan`,
      lastModified: currentDate,
      changeFrequency: 'weekly' as const,
      priority: 0.6,
    },
    {
      url: `${baseUrl}/journey`,
      lastModified: currentDate,
      changeFrequency: 'monthly' as const,
      priority: 0.5,
    },
    {
      url: `${baseUrl}/privacy`,
      lastModified: currentDate,
      changeFrequency: 'yearly' as const,
      priority: 0.3,
    },
    {
      url: `${baseUrl}/terms`,
      lastModified: currentDate,
      changeFrequency: 'yearly' as const,
      priority: 0.3,
    },
  ]

  // Service pages with high priority
  const servicePages = [
    'ai-machine-learning',
    'nextjs-development', 
    'fullstack-web-development',
    'deep-learning',
    'ai-consulting',
    'data-analytics',
    'ai-chatbots',
    'enterprise-solutions'
  ].map(service => ({
    url: `${baseUrl}/services/${service}`,
    lastModified: currentDate,
    changeFrequency: 'weekly' as const,
    priority: 0.8,
  }))

  // Blog posts (you can dynamically generate these from your blog data)
  const blogPosts = [
    'ai-automation-business-processes',
    'ai-chatbots-conversational-interfaces', 
    'deep-learning-neural-networks-guide',
    'machine-learning-business-applications',
    'nextjs-performance-optimization',
    'ai-ethics-responsible-development'
  ].map(slug => ({
    url: `${baseUrl}/blog/${slug}`,
    lastModified: currentDate,
    changeFrequency: 'monthly' as const,
    priority: 0.6,
  }))

  // Industry-specific landing pages
  const industryPages = siteConfig.industries.map(industry => ({
    url: `${baseUrl}/industries/${industry.toLowerCase().replace(/\s+/g, '-').replace(/&/g, 'and')}`,
    lastModified: currentDate,
    changeFrequency: 'monthly' as const,
    priority: 0.7,
  }))

  // Technology-specific pages
  const technologyPages = [
    'tensorflow',
    'pytorch',
    'openai',
    'aws-ai',
    'google-cloud-ai',
    'azure-ai',
    'react',
    'nextjs',
    'typescript',
    'python',
    'nodejs'
  ].map(tech => ({
    url: `${baseUrl}/technologies/${tech}`,
    lastModified: currentDate,
    changeFrequency: 'monthly' as const,
    priority: 0.6,
  }))

  // Case studies and portfolio items
  const portfolioPages = [
    'healthcare-ai-diagnosis',
    'fintech-fraud-detection',
    'ecommerce-recommendation-engine',
    'manufacturing-predictive-maintenance',
    'education-personalized-learning'
  ].map(project => ({
    url: `${baseUrl}/portfolio/${project}`,
    lastModified: currentDate,
    changeFrequency: 'monthly' as const,
    priority: 0.7,
  }))

  return [
    ...staticPages,
    ...servicePages,
    ...blogPosts,
    ...industryPages,
    ...technologyPages,
    ...portfolioPages
  ]
}