import { Metadata } from "next";
import Link from "next/link";
import { generateSEO } from "@/lib/seo-utils";
import { siteConfig } from "@/config/site";

import CTA from "../components/sections/cta/default";
import FAQ from "../components/sections/faq/default";
import Footer from "../components/sections/footer/default";
import ModernHero from "../components/sections/hero/modern-hero";
import Items from "../components/sections/items/default";
import Logos from "../components/sections/logos/default";
import CompanyMarquee from "../components/sections/logos/company-marquee";
import ProfessionalNavbar from "../components/ui/professional-navbar";
import Stats from "../components/sections/stats/default";
import TeamSection from "../components/sections/team/team-section";
import PricingSection from "../components/sections/pricing/pricing-section";
import ModernTestimonials from "../components/sections/testimonials/modern-testimonials";
import SmoothScroll from "../components/ui/smooth-scroll";
import SocialShare from "../components/ui/social-share";
// Structured data imports removed as they're not currently used
// import { 
//   OrganizationStructuredData, 
//   WebsiteStructuredData, 
//   FAQStructuredData,
//   BreadcrumbStructuredData 
// } from "../components/seo/structured-data";

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
    <main className="min-h-screen w-full overflow-hidden bg-black text-white relative">
      <SmoothScroll />
      <div className="fixed top-0 left-0 right-0 z-50">
        <ProfessionalNavbar />
      </div>
      <ModernHero />
      <CompanyMarquee className="-mt-8 sm:-mt-16" />
      <Logos />
      <Items />
      <Stats />
      <TeamSection />
      <PricingSection />
      <ModernTestimonials />
      
      {/* SEO Content Section */}
      <section className="py-20 bg-gradient-to-br from-gray-900 via-black to-gray-900 relative overflow-hidden">
        <div className="absolute inset-0">
          <div className="absolute top-0 left-0 w-full h-px bg-gradient-to-r from-transparent via-white/20 to-transparent" />
          <div className="absolute bottom-0 left-0 w-full h-px bg-gradient-to-r from-transparent via-white/20 to-transparent" />
        </div>
        
        <div className="max-w-container mx-auto px-4 relative z-10">
          <div className="max-w-4xl mx-auto">
            <h2 className="text-4xl sm:text-5xl font-bold text-center mb-6 text-white">
              Elevate Your Business with Expert <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-400">AI & Web Development</span>
            </h2>
            <p className="text-gray-300 text-center max-w-3xl mx-auto mb-12 text-lg">
              From strategy to production, we build performant, secure and scalable solutions that deliver measurable outcomes.
            </p>
            
            <div className="grid md:grid-cols-2 gap-8 mb-12">
              <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-2xl p-8 border border-white/30">
                <h3 className="text-2xl font-semibold mb-4 text-white">Artificial Intelligence Expertise</h3>
                <p className="text-gray-300 mb-4">
                  Our team specializes in cutting-edge artificial intelligence and machine learning solutions. 
                  We develop custom AI models, implement deep learning algorithms, and create intelligent 
                  automation systems that transform how businesses operate.
                </p>
                <p className="text-gray-300">
                  Our AI services include predictive analytics, recommendation systems, chatbot development, 
                  and automated decision-making platforms that increase efficiency and reduce costs.
                </p>
              </div>
              
              <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-2xl p-8 border border-white/30">
                <h3 className="text-2xl font-semibold mb-4 text-white">Modern Web Development</h3>
                <p className="text-gray-300 mb-4">
                  We excel in modern web development using Next.js, React, and TypeScript. Our full-stack 
                  development approach ensures scalable, performant, and user-friendly web applications.
                </p>
                <p className="text-gray-300">
                  Our web development services include responsive design, API development, database 
                  integration, and performance optimization following industry best practices.
                </p>
              </div>
            </div>
            
            <div className="text-center">
              <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-8">
                <a 
                  href="/zehan" 
                  className="bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white px-8 py-3 rounded-full font-semibold transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl inline-flex items-center gap-2"
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
