import { Metadata } from "next";
import { Badge } from "@/components/ui/badge";
import { Section } from "@/components/ui/section";
import { Globe, CheckCircle, ArrowRight, Database, Server, Smartphone } from "lucide-react";
import Link from "next/link";
import Navbar from "@/components/sections/navbar/default";
import Footer from "@/components/sections/footer/default";

export const metadata: Metadata = {
  title: "Full-Stack Web Development Services - Complete Web Solutions | Zehan X",
  description: "Professional full-stack web development services. Frontend, backend, database design, and cloud deployment. React, Next.js, Node.js, and modern web technologies.",
  keywords: [
    "full-stack web development",
    "complete web solutions",
    "frontend backend development",
    "React development",
    "Node.js development",
    "database design",
    "cloud deployment",
    "web application development",
    "API development",
    "full-stack developers",
    "modern web development",
    "scalable web applications"
  ],
  alternates: {
    canonical: "https://zehanx.com/services/fullstack-web-development",
  },
};

export default function FullStackWebDevelopment() {
  return (
    <main className="min-h-screen w-full overflow-hidden bg-background text-foreground">
      <Navbar />
      
      {/* Hero Section */}
      <Section className="pt-24 pb-16">
        <div className="max-w-container mx-auto">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div>
              <Badge variant="outline" className="mb-6">
                <Globe className="mr-2 size-4" />
                Full-Stack Web Development
              </Badge>
              <h1 className="text-4xl font-bold mb-6 sm:text-6xl">
                Complete Web Solutions from Frontend to Backend
              </h1>
              <p className="text-xl text-muted-foreground mb-8">
                We build comprehensive web applications with modern frontend interfaces, 
                robust backend systems, and scalable database architectures.
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
                <Globe className="size-32 text-primary" />
              </div>
            </div>
          </div>
        </div>
      </Section>

      {/* Stack Overview */}
      <Section className="py-16 bg-muted/30">
        <div className="max-w-container mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Full-Stack Architecture</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              We handle every layer of your web application for seamless integration and optimal performance.
            </p>
          </div>
          
          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                icon: <Smartphone className="size-8" />,
                title: "Frontend Development",
                description: "Modern, responsive user interfaces built with React, Next.js, and cutting-edge CSS frameworks.",
                technologies: ["React", "Next.js", "TypeScript", "Tailwind CSS", "Framer Motion"]
              },
              {
                icon: <Server className="size-8" />,
                title: "Backend Development",
                description: "Robust server-side applications with RESTful APIs, authentication, and business logic.",
                technologies: ["Node.js", "Express", "Next.js API", "GraphQL", "Authentication"]
              },
              {
                icon: <Database className="size-8" />,
                title: "Database & Infrastructure",
                description: "Scalable database design and cloud infrastructure for reliable, high-performance applications.",
                technologies: ["PostgreSQL", "MongoDB", "Prisma", "AWS", "Vercel"]
              }
            ].map((stack, index) => (
              <div key={index} className="p-6 bg-background rounded-lg border">
                <div className="size-16 bg-primary/10 rounded-full flex items-center justify-center mb-4 text-primary">
                  {stack.icon}
                </div>
                <h3 className="font-semibold mb-2">{stack.title}</h3>
                <p className="text-muted-foreground text-sm mb-4">{stack.description}</p>
                <div className="flex flex-wrap gap-2">
                  {stack.technologies.map((tech, techIndex) => (
                    <span key={techIndex} className="px-2 py-1 bg-muted rounded text-xs">
                      {tech}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </Section>

      {/* Services Section */}
      <Section className="py-16">
        <div className="max-w-container mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Our Full-Stack Development Services</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              End-to-end web development solutions for businesses of all sizes.
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {[
              {
                title: "Custom Web Applications",
                description: "Tailored web applications built to meet your specific business requirements and workflows."
              },
              {
                title: "E-commerce Platforms",
                description: "Complete online stores with payment processing, inventory management, and admin dashboards."
              },
              {
                title: "SaaS Applications",
                description: "Software-as-a-Service platforms with user management, subscriptions, and multi-tenancy."
              },
              {
                title: "Progressive Web Apps",
                description: "PWAs that work offline, send push notifications, and provide native app-like experiences."
              },
              {
                title: "API Development",
                description: "RESTful APIs and GraphQL endpoints for mobile apps, third-party integrations, and microservices."
              },
              {
                title: "Database Design",
                description: "Optimized database schemas, migrations, and performance tuning for scalable data storage."
              },
              {
                title: "Cloud Deployment",
                description: "Scalable cloud infrastructure setup with CI/CD pipelines and automated deployments."
              },
              {
                title: "Performance Optimization",
                description: "Speed optimization, caching strategies, and monitoring for high-performance applications."
              },
              {
                title: "Maintenance & Support",
                description: "Ongoing maintenance, security updates, and technical support for your web applications."
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

      {/* Development Process */}
      <Section className="py-16 bg-muted/30">
        <div className="max-w-container mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Our Development Process</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              We follow agile development practices to deliver high-quality web applications on time.
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {[
              {
                step: "01",
                title: "Planning & Design",
                description: "Requirements analysis, system architecture design, and UI/UX planning."
              },
              {
                step: "02",
                title: "Development",
                description: "Agile development with regular sprints, code reviews, and testing."
              },
              {
                step: "03",
                title: "Testing & QA",
                description: "Comprehensive testing including unit tests, integration tests, and user acceptance testing."
              },
              {
                step: "04",
                title: "Deployment & Support",
                description: "Production deployment, monitoring setup, and ongoing maintenance support."
              }
            ].map((process, index) => (
              <div key={index} className="text-center">
                <div className="size-16 bg-primary/10 rounded-full flex items-center justify-center mx-auto mb-4">
                  <span className="text-primary font-bold text-lg">{process.step}</span>
                </div>
                <h3 className="font-semibold mb-2">{process.title}</h3>
                <p className="text-muted-foreground text-sm">{process.description}</p>
              </div>
            ))}
          </div>
        </div>
      </Section>

      {/* CTA Section */}
      <Section className="py-16">
        <div className="max-w-container mx-auto text-center">
          <h2 className="text-3xl font-bold mb-6">
            Ready to Build Your Web Application?
          </h2>
          <p className="text-muted-foreground mb-8 max-w-2xl mx-auto">
            Let's create a powerful, scalable web application that drives your business success.
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