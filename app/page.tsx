import { Metadata } from "next";
import Link from "next/link";
import Script from "next/script";
import { generateSEO } from "@/lib/seo-utils";
import { siteConfig } from "@/config/site";

import CTA from "../components/sections/cta/default";
import FAQ from "../components/sections/faq/default";
import Footer from "../components/sections/footer/default";
import ModernHero from "../components/sections/hero/modern-hero";
import Items from "../components/sections/items/default";
import Logos from "../components/sections/logos/default";
import CompanyMarquee from "../components/sections/logos/company-marquee";
import Navbar from "../components/sections/navbar/default";
import Stats from "../components/sections/stats/default";
import AIInnovationSection from "../components/sections/corporate/ai-innovation-section";
import EnterpriseSolutionsSection from "../components/sections/corporate/enterprise-solutions-section";
import DigitalTransformationSection from "../components/sections/corporate/digital-transformation-section";
import InnovationLabSection from "../components/sections/corporate/innovation-lab-section";
import ModernTestimonials from "../components/sections/testimonials/modern-testimonials";
import SmoothScroll from "../components/ui/smooth-scroll";
import SocialShare from "../components/ui/social-share";
import BookMeetingButton from "../components/ui/book-meeting-button";
import ChatbotTrigger from "../components/ui/chatbot-trigger";
// Structured data imports removed as they're not currently used
// import { 
//   OrganizationStructuredData, 
//   WebsiteStructuredData, 
//   FAQStructuredData,
//   BreadcrumbStructuredData 
// } from "../components/seo/structured-data";

export const metadata: Metadata = generateSEO({
  title: "Creative AI + Web Development Agency | Zehan X Technologies",
  description: "Full-service creative agency specializing in AI solutions, web development, digital marketing, video editing, graphic design, and content writing. Transform your vision into stunning digital experiences.",
  keywords: [
    // Primary high-volume keywords
    "web development company",
    "creative agency",
    "Next.js development agency",
    "enterprise SEO web development",
    "digital solutions consulting",
    "custom web development",
    "technical SEO for Next.js",
    "Core Web Vitals optimization",
    "modern website development",
    "best web development company",
    
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
      <Navbar />
      <ModernHero />
      <CompanyMarquee className="-mt-8 sm:-mt-16" />
      <Logos />
      <Items />
      
      {/* Creative Services Showcase */}
      <section className="py-20 bg-gradient-to-br from-gray-900 via-black to-gray-900 relative overflow-hidden">
        <div className="absolute inset-0">
          <div className="absolute top-0 left-0 w-full h-px bg-gradient-to-r from-transparent via-white/20 to-transparent" />
          <div className="absolute bottom-0 left-0 w-full h-px bg-gradient-to-r from-transparent via-white/20 to-transparent" />
        </div>
        
        <div className="max-w-container mx-auto px-4 relative z-10">
          <div className="text-center mb-16">
            <h2 className="text-4xl sm:text-5xl font-bold mb-6 text-white animate-fade-in-up">
              Transform Your Vision Into <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-400">Digital Reality</span>
            </h2>
            <p className="text-gray-300 text-lg max-w-3xl mx-auto animate-fade-in-up" style={{animationDelay: '0.2s'}}>
              From AI-powered solutions to stunning visual designs, we bring your ideas to life with cutting-edge technology and creative excellence
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {/* Technology Solutions */}
            <div className="group bg-black rounded-2xl p-8 border border-blue-500/30 hover:border-blue-500 transition-all duration-300 hover:transform hover:scale-105 animate-fade-in-up">
              <div className="w-16 h-16 bg-blue-500/20 rounded-xl flex items-center justify-center mb-6 group-hover:bg-blue-500/30 transition-colors">
                <svg className="w-8 h-8 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold mb-4 text-white">Technology Solutions</h3>
              <p className="text-gray-300 mb-4">Custom software solutions, automation systems, and intelligent applications that transform business operations.</p>
              <a href="/services/enterprise-solutions" className="text-blue-400 hover:text-blue-300 font-medium">Learn More →</a>
            </div>
            
            {/* Web Development */}
            <div className="group bg-black rounded-2xl p-8 border border-blue-500/30 hover:border-blue-500 transition-all duration-300 hover:transform hover:scale-105 animate-fade-in-up" style={{animationDelay: '0.1s'}}>
              <div className="w-16 h-16 bg-blue-500/20 rounded-xl flex items-center justify-center mb-6 group-hover:bg-blue-500/30 transition-colors">
                <svg className="w-8 h-8 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold mb-4 text-white">Web Development</h3>
              <p className="text-gray-300 mb-4">Modern, responsive websites and web applications built with Next.js, React, and cutting-edge technologies.</p>
              <a href="/services/fullstack-web-development" className="text-blue-400 hover:text-blue-300 font-medium">Learn More →</a>
            </div>
            
            {/* Digital Marketing */}
            <div className="group bg-black rounded-2xl p-8 border border-blue-500/30 hover:border-blue-500 transition-all duration-300 hover:transform hover:scale-105 animate-fade-in-up" style={{animationDelay: '0.2s'}}>
              <div className="w-16 h-16 bg-blue-500/20 rounded-xl flex items-center justify-center mb-6 group-hover:bg-blue-500/30 transition-colors">
                <svg className="w-8 h-8 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold mb-4 text-white">Digital Marketing</h3>
              <p className="text-gray-300 mb-4">Strategic digital marketing campaigns that drive traffic, leads, and conversions for your business.</p>
              <a href="/services/digital-marketing" className="text-blue-400 hover:text-blue-300 font-medium">Learn More →</a>
            </div>
            
            {/* Video Editing */}
            <div className="group bg-black rounded-2xl p-8 border border-blue-500/30 hover:border-blue-500 transition-all duration-300 hover:transform hover:scale-105 animate-fade-in-up" style={{animationDelay: '0.3s'}}>
              <div className="w-16 h-16 bg-blue-500/20 rounded-xl flex items-center justify-center mb-6 group-hover:bg-blue-500/30 transition-colors">
                <svg className="w-8 h-8 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold mb-4 text-white">Video Editing</h3>
              <p className="text-gray-300 mb-4">Professional video production, editing, and post-production services that bring your vision to life.</p>
              <a href="/services/video-editing" className="text-blue-400 hover:text-blue-300 font-medium">Learn More →</a>
            </div>
            
            {/* Graphic Design */}
            <div className="group bg-black rounded-2xl p-8 border border-blue-500/30 hover:border-blue-500 transition-all duration-300 hover:transform hover:scale-105 animate-fade-in-up" style={{animationDelay: '0.4s'}}>
              <div className="w-16 h-16 bg-blue-500/20 rounded-xl flex items-center justify-center mb-6 group-hover:bg-blue-500/30 transition-colors">
                <svg className="w-8 h-8 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zM21 5a2 2 0 00-2-2h-4a2 2 0 00-2 2v12a4 4 0 004 4h4a2 2 0 002-2V5z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold mb-4 text-white">Graphic Design</h3>
              <p className="text-gray-300 mb-4">Creative visual design for branding, marketing materials, and digital assets that make your brand stand out.</p>
              <a href="/services/graphic-design" className="text-blue-400 hover:text-blue-300 font-medium">Learn More →</a>
            </div>
            
            {/* Content Writing */}
            <div className="group bg-black rounded-2xl p-8 border border-blue-500/30 hover:border-blue-500 transition-all duration-300 hover:transform hover:scale-105 animate-fade-in-up" style={{animationDelay: '0.5s'}}>
              <div className="w-16 h-16 bg-blue-500/20 rounded-xl flex items-center justify-center mb-6 group-hover:bg-blue-500/30 transition-colors">
                <svg className="w-8 h-8 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold mb-4 text-white">Content Writing</h3>
              <p className="text-gray-300 mb-4">Compelling content creation for websites, blogs, and marketing materials that engage and convert.</p>
              <a href="/services/content-writing" className="text-blue-400 hover:text-blue-300 font-medium">Learn More →</a>
            </div>
          </div>
        </div>
      </section>
      <Stats />
      <AIInnovationSection />
      <EnterpriseSolutionsSection />
      <DigitalTransformationSection />
      <InnovationLabSection />
      <ModernTestimonials />
      
      {/* Custom Orders Section */}
      <section className="py-20 bg-gradient-to-br from-gray-900 via-black to-gray-900 relative overflow-hidden">
        <div className="absolute inset-0">
          <div className="absolute top-0 left-0 w-full h-px bg-gradient-to-r from-transparent via-white/20 to-transparent" />
          <div className="absolute bottom-0 left-0 w-full h-px bg-gradient-to-r from-transparent via-white/20 to-transparent" />
        </div>
        
        <div className="max-w-container mx-auto px-4 relative z-10">
          <div className="max-w-4xl mx-auto text-center">
            <h2 className="text-4xl sm:text-5xl font-bold mb-6 text-white animate-fade-in-up">
              Ready to Start Your <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-400 animate-pulse">Custom Project?</span>
            </h2>
            <p className="text-gray-300 text-lg max-w-3xl mx-auto mb-12 animate-fade-in-up" style={{animationDelay: '0.2s'}}>
              Every project is unique. Contact us directly to discuss your specific requirements and get a personalized quote tailored to your needs.
            </p>
            
            <div className="bg-black rounded-2xl p-8 border border-blue-500/30 hover:border-blue-500 transition-all duration-300 max-w-2xl mx-auto animate-fade-in-up" style={{animationDelay: '0.4s'}}>
              <div className="flex items-center justify-center mb-6">
                <div className="w-16 h-16 bg-blue-500/20 rounded-xl flex items-center justify-center animate-pulse">
                  <svg className="w-8 h-8 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 4.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                  </svg>
                </div>
              </div>
              <h3 className="text-2xl font-semibold mb-4 text-white">Custom Orders & Consultations</h3>
              <p className="text-gray-300 mb-6">
                Get in touch with our team to discuss your project requirements, timeline, and budget. 
                We'll provide a detailed proposal tailored to your specific needs.
              </p>
              <Link 
                href="/contact" 
                className="inline-flex items-center justify-center rounded-md bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 text-sm font-medium shadow-lg hover:shadow-xl transition-all duration-300 transform hover:scale-105"
              >
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 4.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
                Contact Us for Custom Quote
              </Link>
            </div>
          </div>
        </div>
      </section>
      
      {/* Team Section */}
      <section className="py-20 bg-gradient-to-br from-black via-gray-900 to-black relative overflow-hidden">
        <div className="absolute inset-0">
          <div className="absolute top-0 left-0 w-full h-px bg-gradient-to-r from-transparent via-white/20 to-transparent" />
          <div className="absolute bottom-0 left-0 w-full h-px bg-gradient-to-r from-transparent via-white/20 to-transparent" />
        </div>
        
        <div className="max-w-container mx-auto px-4 relative z-10">
          <div className="text-center mb-16">
            <h2 className="text-4xl sm:text-5xl font-bold mb-6 text-white animate-fade-in-up">
              Meet The <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-400">Visionaries</span>
            </h2>
            <p className="text-gray-300 text-lg max-w-3xl mx-auto animate-fade-in-up" style={{animationDelay: '0.2s'}}>
              The brilliant minds behind Zehan X Technologies, transforming ideas into extraordinary digital experiences
            </p>
          </div>
          
          <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
            {/* Ahmad Jamil - Founder */}
            <div className="group bg-black rounded-2xl p-8 border border-blue-500/30 hover:border-blue-500 transition-all duration-300 hover:transform hover:scale-105 text-center animate-fade-in-up">
              <div className="w-24 h-24 bg-blue-500/20 rounded-full flex items-center justify-center mx-auto mb-6 group-hover:bg-blue-500/30 transition-colors">
                <span className="text-2xl font-bold text-blue-400">AJ</span>
              </div>
              <h3 className="text-xl font-semibold mb-2 text-white">Ahmad Jamil</h3>
              <p className="text-blue-400 font-medium mb-4">Founder</p>
              <p className="text-gray-300 text-sm">
                Visionary leader and strategic architect driving innovation in AI and web development solutions.
              </p>
            </div>
            
            {/* Humayl - Co-founder */}
            <div className="group bg-black rounded-2xl p-8 border border-blue-500/30 hover:border-blue-500 transition-all duration-300 hover:transform hover:scale-105 text-center animate-fade-in-up" style={{animationDelay: '0.2s'}}>
              <div className="w-24 h-24 bg-blue-500/20 rounded-full flex items-center justify-center mx-auto mb-6 group-hover:bg-blue-500/30 transition-colors">
                <span className="text-2xl font-bold text-blue-400">H</span>
              </div>
              <h3 className="text-xl font-semibold mb-2 text-white">Humayl</h3>
              <p className="text-blue-400 font-medium mb-4">Co-founder</p>
              <p className="text-gray-300 text-sm">
                Operations expert ensuring smooth project delivery and maintaining the highest quality standards.
              </p>
            </div>
            
            {/* Ahmad Ibrahim - Co-founder */}
            <div className="group bg-black rounded-2xl p-8 border border-blue-500/30 hover:border-blue-500 transition-all duration-300 hover:transform hover:scale-105 text-center animate-fade-in-up" style={{animationDelay: '0.4s'}}>
              <div className="w-24 h-24 bg-blue-500/20 rounded-full flex items-center justify-center mx-auto mb-6 group-hover:bg-blue-500/30 transition-colors">
                <span className="text-2xl font-bold text-blue-400">AI</span>
              </div>
              <h3 className="text-xl font-semibold mb-2 text-white">Ahmad Ibrahim</h3>
              <p className="text-blue-400 font-medium mb-4">Co-founder</p>
              <p className="text-gray-300 text-sm">
                Strategic leader overseeing business operations and driving growth in the creative technology space.
              </p>
            </div>
          </div>
        </div>
      </section>
      
      {/* SEO Content Section */}
      <section className="py-20 bg-gradient-to-br from-gray-900 via-black to-gray-900 relative overflow-hidden">
        <div className="absolute inset-0">
          <div className="absolute top-0 left-0 w-full h-px bg-gradient-to-r from-transparent via-white/20 to-transparent" />
          <div className="absolute bottom-0 left-0 w-full h-px bg-gradient-to-r from-transparent via-white/20 to-transparent" />
        </div>
        
        <div className="max-w-container mx-auto px-4 relative z-10">
          <div className="max-w-4xl mx-auto">
            <h2 className="text-4xl sm:text-5xl font-bold text-center mb-6 text-white">
              Elevate Your Brand with Creative <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-400">AI & Digital Solutions</span>
            </h2>
            <p className="text-gray-300 text-center max-w-3xl mx-auto mb-12 text-lg">
              From AI-powered applications to stunning visual designs, we create comprehensive digital experiences that captivate audiences and drive business growth.
            </p>
            
            <div className="grid md:grid-cols-2 gap-8 mb-12">
              <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-2xl p-8 border border-white/30">
                <h3 className="text-2xl font-semibold mb-4 text-white">Technology Solutions</h3>
                <p className="text-gray-300 mb-4">
                  We specialize in cutting-edge web development and modern technology solutions. 
                  Our services include custom applications, automation systems, and enterprise solutions that transform business operations.
                </p>
                <p className="text-gray-300">
                  Our technology stack includes Next.js, React, TypeScript, and modern frameworks, 
                  ensuring scalable and robust applications that drive results.
                </p>
              </div>
              
              <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-2xl p-8 border border-white/30">
                <h3 className="text-2xl font-semibold mb-4 text-white">Creative & Marketing Services</h3>
                <p className="text-gray-300 mb-4">
                  We provide comprehensive creative services including digital marketing, video editing, 
                  graphic design, and content writing. Our creative team brings your brand to life with 
                  stunning visuals and compelling content.
                </p>
                <p className="text-gray-300">
                  From social media campaigns to professional video production, we create engaging 
                  content that captivates your audience and drives meaningful engagement.
                </p>
              </div>
            </div>
            
            <div className="text-center">
              <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-8">
                <Link 
                  href="/contact" 
                  className="bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white px-8 py-3 rounded-full font-semibold transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl inline-flex items-center gap-2"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 4.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                  </svg>
                  Get Custom Quote
                  <span className="text-sm">→</span>
                </Link>
                <SocialShare />
              </div>
            </div>
          </div>
        </div>
      </section>
      
      <FAQ />
      <CTA />
      <Footer />
      <BookMeetingButton />
      <ChatbotTrigger />
      
      {/* Botpress Chatbot Scripts */}
      <Script 
        src="https://cdn.botpress.cloud/webchat/v3.3/inject.js" 
        strategy="afterInteractive"
      />
      <Script 
        src="https://files.bpcontent.cloud/2025/09/19/13/20250919130112-NCQJ5BHI.js" 
        strategy="lazyOnload"
      />
    </main>
  );
}
