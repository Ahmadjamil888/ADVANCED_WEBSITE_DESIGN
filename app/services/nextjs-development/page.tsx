import { Metadata } from "next";
import { Badge } from "@/components/ui/badge";
import { Section } from "@/components/ui/section";
import { Code, CheckCircle, ArrowRight, Zap, Globe, Shield } from "lucide-react";
import Link from "next/link";
import Navbar from "@/components/sections/navbar/default";
import Footer from "@/components/sections/footer/default";

export const metadata: Metadata = {
  title: "Next.js Development Services - Modern React Web Applications | Zehan X",
  description: "Professional Next.js development services. Build fast, scalable, SEO-optimized web applications with React, TypeScript, and modern web technologies. Expert Next.js developers.",
  keywords: [
    "Next.js development",
    "React development services",
    "Next.js developers",
    "modern web applications",
    "React web development",
    "TypeScript development",
    "SSR development",
    "JAMstack development",
    "Next.js consulting",
    "React consulting",
    "web application development",
    "frontend development services"
  ],
  openGraph: {
    title: "Next.js Development Services - Modern React Web Applications",
    description: "Professional Next.js development services. Build fast, scalable, SEO-optimized web applications with React and modern web technologies.",
    type: "website",
    url: "https://zehanx.com/services/nextjs-development",
  },
  alternates: {
    canonical: "https://zehanx.com/services/nextjs-development",
  },
};

export default function NextJSDevelopment() {
  return (
    <main className="min-h-screen w-full overflow-hidden bg-background text-foreground">
      <Navbar />
      
      {/* Hero Section */}
      <Section className="pt-24 pb-16">
        <div className="max-w-container mx-auto">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div>
              <Badge variant="outline" className="mb-6">
                <Code className="mr-2 size-4" />
                Next.js Development
              </Badge>
              <h1 className="text-4xl font-bold mb-6 sm:text-6xl">
                Modern Web Applications with Next.js
              </h1>
              <p className="text-xl text-muted-foreground mb-8">
                Build fast, scalable, and SEO-optimized web applications using Next.js, 
                the React framework trusted by leading companies worldwide.
              </p>
              <div className="flex gap-4">
                <Link 
                  href="/contact" 
                  className="inline-flex items-center justify-center rounded-md bg-primary px-6 py-3 text-sm font-medium text-primary-foreground shadow transition-colors hover:bg-primary/90"
                >
                  Get Started
                  <ArrowRight className="ml-2 size-4" />
                </Link>
                <Link 
                  href="/services" 
                  className="inline-flex items-center justify-center rounded-md border border-input bg-background px-6 py-3 text-sm font-medium shadow-sm transition-colors hover:bg-accent hover:text-accent-foreground"
                >
                  All Services
                </Link>
              </div>
            </div>
            <div className="relative">
              <div className="size-64 mx-auto bg-gradient-to-br from-blue-600/20 to-purple-600/20 rounded-full flex items-center justify-center">
                <Code className="size-32 text-primary" />
              </div>
            </div>
          </div>
        </div>
      </Section>

      {/* Features Section */}
      <Section className="py-16 bg-muted/30">
        <div className="max-w-container mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Why Choose Next.js?</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Next.js provides the best developer experience with all the features you need for production.
            </p>
          </div>
          
          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                icon: <Zap className="size-8" />,
                title: "Lightning Fast",
                description: "Optimized performance with automatic code splitting, image optimization, and built-in caching."
              },
              {
                icon: <Globe className="size-8" />,
                title: "SEO Optimized",
                description: "Server-side rendering and static generation for better search engine visibility."
              },
              {
                icon: <Shield className="size-8" />,
                title: "Production Ready",
                description: "Built-in security features, TypeScript support, and enterprise-grade reliability."
              }
            ].map((feature, index) => (
              <div key={index} className="text-center p-6 bg-background rounded-lg border">
                <div className="size-16 bg-primary/10 rounded-full flex items-center justify-center mx-auto mb-4 text-primary">
                  {feature.icon}
                </div>
                <h3 className="font-semibold mb-2">{feature.title}</h3>
                <p className="text-muted-foreground text-sm">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </Section>

      {/* Services Section */}
      <Section className="py-16">
        <div className="max-w-container mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Our Next.js Development Services</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Comprehensive Next.js solutions for all your web development needs.
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {[
              {
                title: "Full-Stack Applications",
                description: "Complete web applications with frontend and backend built using Next.js and modern technologies."
              },
              {
                title: "E-commerce Solutions",
                description: "High-performance online stores with payment integration and inventory management."
              },
              {
                title: "API Development",
                description: "RESTful APIs and GraphQL endpoints built with Next.js API routes."
              },
              {
                title: "Performance Optimization",
                description: "Speed optimization, Core Web Vitals improvement, and performance monitoring."
              },
              {
                title: "Migration Services",
                description: "Migrate existing applications to Next.js for better performance and maintainability."
              },
              {
                title: "Maintenance & Support",
                description: "Ongoing maintenance, updates, and technical support for your Next.js applications."
              }
            ].map((service, index) => (
              <div key={index} className="p-6 bg-background rounded-lg border">
                <div className="flex items-start gap-3">
                  <CheckCircle className="size-6 text-green-500 flex-shrink-0 mt-0.5" />
                  <div>
                    <h3 className="font-semibold mb-2">{service.title}</h3>
                    <p className="text-muted-foreground text-sm">{service.description}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </Section>

      {/* Tech Stack Section */}
      <Section className="py-16 bg-muted/30">
        <div className="max-w-container mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Our Next.js Tech Stack</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              We use the latest technologies and best practices for Next.js development.
            </p>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-6">
            {[
              "Next.js 15",
              "React 19",
              "TypeScript",
              "Tailwind CSS",
              "Prisma",
              "PostgreSQL",
              "Vercel",
              "AWS",
              "Docker",
              "Jest",
              "Cypress",
              "Storybook"
            ].map((tech, index) => (
              <div key={index} className="text-center p-4 bg-background rounded-lg border">
                <span className="text-sm font-medium">{tech}</span>
              </div>
            ))}
          </div>
        </div>
      </Section>

      {/* CTA Section */}
      <Section className="py-16">
        <div className="max-w-container mx-auto text-center">
          <h2 className="text-3xl font-bold mb-6">
            Ready to Build Your Next.js Application?
          </h2>
          <p className="text-muted-foreground mb-8 max-w-2xl mx-auto">
            Let's create a fast, scalable, and modern web application that drives your business forward.
          </p>
          <Link 
            href="/contact" 
            className="inline-flex items-center justify-center rounded-md bg-primary px-8 py-3 text-sm font-medium text-primary-foreground shadow transition-colors hover:bg-primary/90"
          >
            Start Your Project
            <ArrowRight className="ml-2 size-4" />
          </Link>
        </div>
      </Section>

      <Footer />
    </main>
  );
}