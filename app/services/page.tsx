import { Badge } from "@/components/ui/badge";
import { Section } from "@/components/ui/section";
import { BrainCircuit, Code, Database, Globe, Bot, Zap, BarChart3, Shield, Palette, Video, PenTool } from "lucide-react";
import Link from "next/link";
import Navbar from "@/components/sections/navbar/default";
import Footer from "@/components/sections/footer/default";

const services = [
  {
    id: "ai-machine-learning",
    icon: <BrainCircuit className="size-12" />,
    title: "AI & Machine Learning",
    description: "Custom AI solutions, predictive analytics, and intelligent automation to transform your business processes.",
    features: ["Custom ML Models", "Predictive Analytics", "Computer Vision", "NLP Solutions"],
    href: "/services/ai-machine-learning"
  },
  {
    id: "nextjs-development",
    icon: <Code className="size-12" />,
    title: "Next.js Development",
    description: "Modern, fast, and scalable web applications built with Next.js and the latest React technologies.",
    features: ["Full-Stack Apps", "Server-Side Rendering", "API Development", "Performance Optimization"],
    href: "/services/nextjs-development"
  },
  {
    id: "deep-learning",
    icon: <Database className="size-12" />,
    title: "Deep Learning",
    description: "Advanced neural networks and deep learning models for complex pattern recognition and decision making.",
    features: ["Neural Networks", "Image Recognition", "Speech Processing", "Recommendation Systems"],
    href: "/services/deep-learning"
  },
  {
    id: "fullstack-web-development",
    icon: <Globe className="size-12" />,
    title: "Full-Stack Web Development",
    description: "Complete web solutions from frontend to backend, optimized for performance and user experience.",
    features: ["Frontend Development", "Backend APIs", "Database Design", "Cloud Deployment"],
    href: "/services/fullstack-web-development"
  },
  {
    id: "ai-chatbots",
    icon: <Bot className="size-12" />,
    title: "AI Chatbots",
    description: "Intelligent conversational AI systems that enhance customer engagement and automate support.",
    features: ["Custom Chatbots", "Voice Assistants", "Customer Support AI", "Integration Services"],
    href: "/services/ai-chatbots"
  },
  {
    id: "ai-consulting",
    icon: <Zap className="size-12" />,
    title: "AI Consulting",
    description: "Strategic AI consulting to help you identify opportunities and implement AI solutions effectively.",
    features: ["AI Strategy", "Technology Assessment", "Implementation Planning", "Training & Support"],
    href: "/services/ai-consulting"
  },
  {
    id: "data-analytics",
    icon: <BarChart3 className="size-12" />,
    title: "Data Analytics",
    description: "Transform your data into actionable business insights with advanced analytics and visualization.",
    features: ["Business Intelligence", "Data Visualization", "Predictive Modeling", "Real-time Analytics"],
    href: "/services/data-analytics"
  },
  {
    id: "enterprise-solutions",
    icon: <Shield className="size-12" />,
    title: "Enterprise Solutions",
    description: "Scalable, secure enterprise-grade solutions built for mission-critical business operations.",
    features: ["Enterprise Architecture", "Security Implementation", "Scalability Planning", "24/7 Support"],
    href: "/services/enterprise-solutions"
  },
  {
    id: "graphic-design",
    icon: <Palette className="size-12" />,
    title: "Graphic Design",
    description: "Creative visual solutions that make an impact with stunning designs for your brand.",
    features: ["Logo Design", "Brand Identity", "Print Design", "Digital Graphics"],
    href: "/services/graphic-design"
  },
  {
    id: "video-editing",
    icon: <Video className="size-12" />,
    title: "Video Editing",
    description: "Professional video editing that brings your stories to life with cinematic quality.",
    features: ["Professional Editing", "Motion Graphics", "Color Correction", "Social Media Content"],
    href: "/services/video-editing"
  },
  {
    id: "content-writing",
    icon: <PenTool className="size-12" />,
    title: "Content Writing",
    description: "Powerful words that drive results with compelling content across all platforms.",
    features: ["Blog Writing", "SEO Content", "Social Media Copy", "Marketing Materials"],
    href: "/services/content-writing"
  }
];

export default function Services() {
  return (
    <main className="min-h-screen w-full overflow-hidden bg-background text-foreground">
      <Navbar />
      
      {/* Hero Section */}
      <Section className="pt-24 pb-16">
        <div className="max-w-container mx-auto text-center">
          <Badge variant="outline" className="mb-6">
            <Zap className="mr-2 size-4" />
            Our Services
          </Badge>
          <h1 className="text-4xl font-bold mb-6 sm:text-6xl">
            Comprehensive Digital Solutions & Creative Services
          </h1>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            From intelligent AI systems to creative content, we deliver end-to-end solutions 
            that drive innovation, engagement, and business growth across all digital platforms.
          </p>
        </div>
      </Section>

      {/* Services Grid */}
      <Section className="py-16">
        <div className="max-w-container mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-8">
            {services.map((service, index) => (
              <Link
                key={index}
                href={service.href}
                className="group relative overflow-hidden rounded-lg border bg-background p-6 hover:shadow-lg transition-all duration-300 hover:border-primary/50"
              >
                <div className="flex flex-col gap-4">
                  <div className="flex-shrink-0 p-3 rounded-lg bg-primary/10 text-primary group-hover:bg-primary group-hover:text-primary-foreground transition-colors w-fit">
                    {service.icon}
                  </div>
                  
                  <div>
                    <h3 className="text-xl font-semibold mb-2 group-hover:text-primary transition-colors">
                      {service.title}
                    </h3>
                    <p className="text-muted-foreground text-sm mb-4">
                      {service.description}
                    </p>
                  </div>
                  
                  <ul className="space-y-2">
                    {service.features.map((feature, featureIndex) => (
                      <li key={featureIndex} className="flex items-center gap-2 text-sm">
                        <div className="size-1.5 rounded-full bg-primary flex-shrink-0" />
                        {feature}
                      </li>
                    ))}
                  </ul>
                </div>
              </Link>
            ))}
          </div>
        </div>
      </Section>

      {/* CTA Section */}
      <Section className="py-16 bg-muted/30">
        <div className="max-w-container mx-auto text-center">
          <h2 className="text-3xl font-bold mb-6">
            Ready to Transform Your Business?
          </h2>
          <p className="text-muted-foreground mb-8 max-w-2xl mx-auto">
            Let's discuss how our comprehensive digital and creative services can help you achieve your goals.
          </p>
          <div className="flex justify-center gap-4">
            <Link 
              href="/contact" 
              className="inline-flex items-center justify-center rounded-md bg-primary px-8 py-3 text-sm font-medium text-primary-foreground shadow transition-colors hover:bg-primary/90"
            >
              Get Started
            </Link>
            <Link 
              href="/about" 
              className="inline-flex items-center justify-center rounded-md border border-input bg-background px-8 py-3 text-sm font-medium shadow-sm transition-colors hover:bg-accent hover:text-accent-foreground"
            >
              Learn More
            </Link>
          </div>
        </div>
      </Section>

      <Footer />
    </main>
  );
}