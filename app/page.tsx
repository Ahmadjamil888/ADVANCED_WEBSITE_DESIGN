import { Metadata } from "next";
import { generateSEO } from "@/lib/seo-utils";
import { siteConfig } from "@/config/site";

import CTA from "../components/sections/cta/default";
import FAQ from "../components/sections/faq/default";
import Footer from "../components/sections/footer/default";
import Hero from "../components/sections/hero/zehan-hero";
import Items from "../components/sections/items/default";
import Logos from "../components/sections/logos/default";
import CompanyMarquee from "../components/sections/logos/company-marquee";
import Navbar from "../components/sections/navbar/default";
import Stats from "../components/sections/stats/default";
import SocialShare from "../components/ui/social-share";
import { 
  OrganizationStructuredData, 
  WebsiteStructuredData, 
  FAQStructuredData,
  BreadcrumbStructuredData 
} from "../components/seo/structured-data";

export const metadata: Metadata = generateSEO({
  title: "Best AI & Web Development Company | SEO-First Next.js Agency | Zehan X Technologies",
  description: "AI development and SEO-first Next.js web development from a results-driven agency. Enterprise machine learning, deep learning, and production-grade web apps that rank and convert.",
  keywords: [
    // Primary high-volume keywords
    "AI development company",
    "web development company",
    "Next.js development agency",
    "enterprise SEO web development",
    "artificial intelligence consulting",
    "deep learning solutions",
    "technical SEO for Next.js",
    "Core Web Vitals optimization",
    "AI website development",
    "best AI company",
    
    // Long-tail keywords with high intent
    "custom AI model development services",
    "enterprise machine learning solutions",
    "AI chatbot development company",
    "predictive analytics consulting",
    "computer vision AI development",
    "natural language processing services",
    "AI automation business solutions",
    "full-stack web development agency",
    "React TypeScript development services",
    
    // Location-based keywords
    "AI development company USA",
    "machine learning services worldwide",
    "remote AI development team",
    "global AI consulting services",
    
    // Industry-specific keywords
    "healthcare AI solutions",
    "fintech machine learning",
    "e-commerce AI development",
    "manufacturing AI automation",
    "education AI platforms",
    
    // Technology-specific keywords
    "TensorFlow development services",
    "PyTorch machine learning",
    "OpenAI API integration",
    "AWS AI services",
    "Google Cloud AI development",
    "Azure AI solutions",
    
    // Service-specific keywords
    "AI model training services",
    "data science consulting",
    "MLOps implementation",
    "AI strategy development",
    "business intelligence solutions",
    "intelligent automation systems",
    
    // Competitive keywords
    "best AI development company",
    "top machine learning agency",
    "leading AI consulting firm",
    "professional AI services",
    "expert AI developers",
    "trusted AI partner",
    
    // Problem-solving keywords
    "AI business transformation",
    "increase efficiency with AI",
    "reduce costs with automation",
    "AI competitive advantage",
    "digital transformation AI",
    "AI ROI optimization"
  ],
  url: siteConfig.url,
  type: "website"
});

export default function Home() {
  return (
    <main className="min-h-screen w-full overflow-hidden bg-background text-foreground">
      <Navbar />
      <Hero />
      <CompanyMarquee className="-mt-8 sm:-mt-16" />
      <Logos />
      <Items />
      <Stats />
      
      {/* SEO Content Section */}
      <section className="py-20 bg-muted/30 section-divider-professional">
        <div className="max-w-container mx-auto px-4">
          <div className="max-w-4xl mx-auto">
            <h2 className="text-4xl sm:text-5xl heading-professional text-center mb-6">
              Elevate Your Business with Expert AI & Next.js Delivery
            </h2>
            <p className="text-muted-foreground text-center max-w-3xl mx-auto mb-12">
              From strategy to production, we build performant, secure and scalable solutions that deliver measurable outcomes.
            </p>
            
            <div className="grid md:grid-cols-2 gap-8 mb-12">
              <div>
                <h3 className="text-2xl font-semibold mb-4">Artificial Intelligence Expertise</h3>
                <p className="text-muted-foreground mb-4">
                  Our team specializes in cutting-edge artificial intelligence and machine learning solutions. 
                  We develop custom AI models, implement deep learning algorithms, and create intelligent 
                  automation systems that transform how businesses operate. From computer vision to natural 
                  language processing, we deliver AI solutions that drive real business value.
                </p>
                <p className="text-muted-foreground">
                  Our AI services include predictive analytics, recommendation systems, chatbot development, 
                  and automated decision-making platforms. We work with businesses across industries to 
                  implement AI strategies that increase efficiency, reduce costs, and unlock new opportunities.
                </p>
              </div>
              
              <div>
                <h3 className="text-2xl font-semibold mb-4">Modern Web Development</h3>
                <p className="text-muted-foreground mb-4">
                  We excel in modern web development using Next.js, React, and TypeScript. Our full-stack 
                  development approach ensures scalable, performant, and user-friendly web applications. 
                  We build everything from simple business websites to complex enterprise applications 
                  with advanced functionality and seamless user experiences.
                </p>
                <p className="text-muted-foreground">
                  Our web development services include responsive design, API development, database 
                  integration, and performance optimization. We follow industry best practices for 
                  security, accessibility, and SEO to ensure your web presence drives business growth.
                </p>
              </div>
            </div>
            
            <div className="text-center">
              <h3 className="text-2xl font-semibold mb-4">Experience Our AI Technology — Try Zehan AI</h3>
              <p className="text-muted-foreground max-w-3xl mx-auto mb-8">
                Zehan X Technologies began as a small web development agency and evolved into a leading 
                AI and machine learning company. Our journey from creating simple websites to developing 
                sophisticated AI systems reflects our commitment to innovation and continuous learning. 
                Today, we combine our web development expertise with advanced AI capabilities to deliver 
                comprehensive digital transformation solutions for businesses worldwide.
              </p>
              
              <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-8">
                <a 
                  href="/zehan" 
                  className="btn-gradient-primary text-white px-6 py-3 rounded-lg font-semibold hover-lift-professional transition-all duration-300 inline-flex items-center gap-2"
                >
                   Try Zehan AI Now
                  <span className="text-sm">→</span>
                </a>
                <SocialShare />
              </div>
            </div>
          </div>
        </div>
      </section>
      
      <FAQ />
      <CTA />
      <Footer />
    </main>
  );
}
