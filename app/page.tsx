import { Metadata } from "next";

import CTA from "../components/sections/cta/default";
import FAQ from "../components/sections/faq/default";
import Footer from "../components/sections/footer/default";
import Hero from "../components/sections/hero/zehan-hero";
import Items from "../components/sections/items/default";
import Logos from "../components/sections/logos/default";
import Navbar from "../components/sections/navbar/default";
import Stats from "../components/sections/stats/default";
import SocialShare from "../components/ui/social-share";

export const metadata: Metadata = {
  title: "AI & Web Development Experts | Zehan X Technologies",
  description: "Expert AI & web development company. Next.js, machine learning & deep learning solutions. Transform your business with AI technology.",
  keywords: [
    "AI development company",
    "machine learning services", 
    "Next.js development",
    "artificial intelligence solutions",
    "deep learning company",
    "AI consulting services",
    "custom AI models",
    "web development agency",
    "React development",
    "AI automation",
    "predictive analytics",
    "business intelligence",
    "AI chatbots",
    "computer vision",
    "natural language processing"
  ],
  openGraph: {
    title: "AI & Web Development Experts | Zehan X Technologies",
    description: "Expert AI & web development company. Next.js, machine learning & deep learning solutions.",
    type: "website",
  },
  twitter: {
    title: "AI & Web Development Experts | Zehan X Technologies", 
    description: "Expert AI & web development company. Next.js, machine learning & deep learning solutions.",
    card: "summary_large_image",
  },
};

export default function Home() {
  return (
    <main className="min-h-screen w-full overflow-hidden bg-background text-foreground">
      <Navbar />
      <Hero />
      <Logos />
      <Items />
      <Stats />
      
      {/* SEO Content Section */}
      <section className="py-16 bg-muted/30">
        <div className="max-w-container mx-auto px-4">
          <div className="max-w-4xl mx-auto">
            <h2 className="text-3xl font-bold text-center mb-8">
              Why Choose Zehan X Technologies for AI & Web Development?
            </h2>
            
            <div className="grid md:grid-cols-2 gap-8 mb-12">
              <div>
                <h3 className="text-xl font-semibold mb-4">Artificial Intelligence Expertise</h3>
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
                <h3 className="text-xl font-semibold mb-4">Modern Web Development</h3>
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
              <h3 className="text-xl font-semibold mb-4">Experience Our AI Technology - Try Zehan AI!</h3>
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
