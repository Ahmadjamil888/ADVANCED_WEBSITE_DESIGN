import { Badge } from "@/components/ui/badge";
import { Section } from "@/components/ui/section";
import { PenTool, FileText, BookOpen, MessageSquare, Zap, CheckCircle, Search, Target } from "lucide-react";
import Link from "next/link";
import Navbar from "@/components/sections/navbar/default";
import Footer from "@/components/sections/footer/default";

const features = [
  {
    icon: <FileText className="size-6" />,
    title: "Blog Writing",
    description: "Engaging blog posts that drive traffic and establish thought leadership."
  },
  {
    icon: <BookOpen className="size-6" />,
    title: "Website Copy",
    description: "Compelling website content that converts visitors into customers."
  },
  {
    icon: <MessageSquare className="size-6" />,
    title: "Social Media Content",
    description: "Creative social media posts that engage and grow your audience."
  },
  {
    icon: <Search className="size-6" />,
    title: "SEO Content",
    description: "Search-optimized content that ranks well and drives organic traffic."
  },
  {
    icon: <Target className="size-6" />,
    title: "Marketing Copy",
    description: "Persuasive marketing materials that drive conversions and sales."
  },
  {
    icon: <PenTool className="size-6" />,
    title: "Technical Writing",
    description: "Clear, concise technical documentation and instructional content."
  }
];

const packages = [
  {
    name: "Content Starter",
    price: "Starting at $149",
    features: [
      "Up to 5 blog posts/month",
      "Basic SEO optimization",
      "Content calendar",
      "2 revisions per piece",
      "Email delivery"
    ]
  },
  {
    name: "Content Pro", 
    price: "Starting at $399",
    features: [
      "Up to 15 pieces/month",
      "Advanced SEO strategy",
      "Social media content",
      "Keyword research included",
      "Unlimited revisions",
      "Priority support"
    ],
    popular: true
  },
  {
    name: "Content Enterprise",
    price: "Starting at $899",
    features: [
      "Unlimited content",
      "Multi-platform strategy",
      "Content performance analytics",
      "Dedicated content manager",
      "Brand voice development",
      "Custom content strategy",
      "24/7 support"
    ]
  }
];

export default function ContentWritingService() {
  return (
    <main className="min-h-screen w-full overflow-hidden bg-background text-foreground">
      <Navbar />
      
      {/* Hero Section */}
      <Section className="pt-24 pb-16">
        <div className="max-w-container mx-auto">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            <div>
              <Badge variant="outline" className="mb-6">
                <PenTool className="mr-2 size-4" />
                Content Writing Services
              </Badge>
              <h1 className="text-4xl font-bold mb-6 sm:text-6xl">
                Powerful Words That Drive Results
              </h1>
              <p className="text-xl text-muted-foreground mb-8">
                Transform your brand's message with compelling content that engages audiences, 
                builds trust, and drives conversions across all platforms.
              </p>
              <div className="flex gap-4">
                <Link 
                  href="/contact" 
                  className="inline-flex items-center justify-center rounded-md bg-primary px-8 py-3 text-sm font-medium text-primary-foreground shadow transition-colors hover:bg-primary/90"
                >
                  <Zap className="mr-2 size-4" />
                  Start Your Project
                </Link>
                <Link 
                  href="/portfolio" 
                  className="inline-flex items-center justify-center rounded-md border border-input bg-background px-8 py-3 text-sm font-medium shadow-sm transition-colors hover:bg-accent hover:text-accent-foreground"
                >
                  View Samples
                </Link>
              </div>
            </div>
            <div className="relative">
              <div className="aspect-square rounded-lg bg-gradient-to-br from-primary/20 to-secondary/20 p-8 flex items-center justify-center">
                <PenTool className="size-32 text-primary" />
              </div>
            </div>
          </div>
        </div>
      </Section>

      {/* Features Section */}
      <Section className="py-16 bg-muted/30">
        <div className="max-w-container mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Our Content Services</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              From blog posts to marketing copy, we create content that resonates 
              with your audience and achieves your business objectives.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <div 
                key={index} 
                className="p-6 rounded-lg bg-background border hover:shadow-lg transition-all duration-300"
              >
                <div className="flex items-center gap-4 mb-4">
                  <div className="p-2 rounded-lg bg-primary/10 text-primary">
                    {feature.icon}
                  </div>
                  <h3 className="text-lg font-semibold">{feature.title}</h3>
                </div>
                <p className="text-muted-foreground">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </Section>

      {/* Packages Section */}
      <Section className="py-16">
        <div className="max-w-container mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Content Writing Packages</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Choose the perfect package for your content needs. All packages include 
              professional research, writing, and editing services.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {packages.map((pkg, index) => (
              <div 
                key={index} 
                className={`relative p-8 rounded-lg border transition-all duration-300 hover:shadow-lg ${
                  pkg.popular ? 'border-primary bg-primary/5 scale-105' : 'bg-background'
                }`}
              >
                {pkg.popular && (
                  <Badge className="absolute -top-3 left-1/2 -translate-x-1/2 bg-primary">
                    Most Popular
                  </Badge>
                )}
                
                <div className="text-center mb-6">
                  <h3 className="text-xl font-bold mb-2">{pkg.name}</h3>
                  <p className="text-2xl font-bold text-primary">{pkg.price}</p>
                </div>
                
                <ul className="space-y-3 mb-8">
                  {pkg.features.map((feature, featureIndex) => (
                    <li key={featureIndex} className="flex items-center gap-3">
                      <CheckCircle className="size-5 text-green-500 flex-shrink-0" />
                      <span className="text-sm">{feature}</span>
                    </li>
                  ))}
                </ul>
                
                <Link 
                  href="/contact" 
                  className={`w-full inline-flex items-center justify-center rounded-md px-6 py-3 text-sm font-medium transition-colors ${
                    pkg.popular 
                      ? 'bg-primary text-primary-foreground hover:bg-primary/90' 
                      : 'border border-input bg-background hover:bg-accent hover:text-accent-foreground'
                  }`}
                >
                  Get Started
                </Link>
              </div>
            ))}
          </div>
        </div>
      </Section>

      {/* Content Types Section */}
      <Section className="py-16 bg-muted/30">
        <div className="max-w-container mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Content Types We Create</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              We specialize in creating various types of content across different industries and platforms.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              { title: "Blog Posts", description: "SEO-optimized articles and thought leadership pieces" },
              { title: "Website Copy", description: "Landing pages, product descriptions, about pages" },
              { title: "Email Marketing", description: "Newsletters, drip campaigns, promotional emails" },
              { title: "Social Media", description: "Posts, captions, stories, and engagement content" },
              { title: "White Papers", description: "In-depth research and industry reports" },
              { title: "Case Studies", description: "Success stories and customer testimonials" },
              { title: "Product Descriptions", description: "Compelling e-commerce and catalog copy" },
              { title: "Press Releases", description: "News announcements and media communications" }
            ].map((type, index) => (
              <div key={index} className="p-6 rounded-lg bg-background border text-center">
                <h3 className="text-lg font-semibold mb-2">{type.title}</h3>
                <p className="text-muted-foreground text-sm">{type.description}</p>
              </div>
            ))}
          </div>
        </div>
      </Section>

      {/* Process Section */}
      <Section className="py-16">
        <div className="max-w-container mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Our Writing Process</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              We follow a proven content creation process to ensure every piece meets your standards and objectives.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            {[
              { step: "01", title: "Research", description: "In-depth research on your industry, audience, and competitors" },
              { step: "02", title: "Strategy", description: "Content strategy development and keyword optimization" },
              { step: "03", title: "Create", description: "Writing compelling, engaging, and valuable content" },
              { step: "04", title: "Optimize", description: "Final editing, SEO optimization, and quality assurance" }
            ].map((process, index) => (
              <div key={index} className="text-center">
                <div className="size-16 rounded-full bg-primary/10 text-primary flex items-center justify-center text-xl font-bold mx-auto mb-4">
                  {process.step}
                </div>
                <h3 className="text-lg font-semibold mb-2">{process.title}</h3>
                <p className="text-muted-foreground text-sm">{process.description}</p>
              </div>
            ))}
          </div>
        </div>
      </Section>

      {/* Why Choose Us Section */}
      <Section className="py-16 bg-muted/30">
        <div className="max-w-container mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Why Choose Our Content Writing?</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              We combine creativity with strategy to deliver content that not only engages but also drives measurable results.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {[
              { title: "Expert Writers", description: "Professional writers with industry expertise and proven track records" },
              { title: "SEO Optimized", description: "All content is optimized for search engines to improve visibility" },
              { title: "Brand Consistency", description: "Maintaining your unique brand voice across all content pieces" },
              { title: "Data-Driven", description: "Content strategy backed by research and performance analytics" },
              { title: "Quick Turnaround", description: "Fast delivery without compromising on quality or accuracy" },
              { title: "Unlimited Revisions", description: "We work with you until the content meets your exact requirements" }
            ].map((benefit, index) => (
              <div key={index} className="p-6 rounded-lg bg-background border">
                <h3 className="text-lg font-semibold mb-2">{benefit.title}</h3>
                <p className="text-muted-foreground text-sm">{benefit.description}</p>
              </div>
            ))}
          </div>
        </div>
      </Section>

      {/* CTA Section */}
      <Section className="py-16">
        <div className="max-w-container mx-auto text-center">
          <h2 className="text-3xl font-bold mb-6">
            Ready to Transform Your Content Strategy?
          </h2>
          <p className="text-muted-foreground mb-8 max-w-2xl mx-auto">
            Let's create compelling content that tells your brand story, engages your audience, 
            and drives the results your business needs.
          </p>
          <div className="flex justify-center gap-4">
            <Link 
              href="/contact" 
              className="inline-flex items-center justify-center rounded-md bg-primary px-8 py-3 text-sm font-medium text-primary-foreground shadow transition-colors hover:bg-primary/90"
            >
              Start Your Project
            </Link>
            <Link 
              href="/services" 
              className="inline-flex items-center justify-center rounded-md border border-input bg-background px-8 py-3 text-sm font-medium shadow-sm transition-colors hover:bg-accent hover:text-accent-foreground"
            >
              View All Services
            </Link>
          </div>
        </div>
      </Section>

      <Footer />
    </main>
  );
}
