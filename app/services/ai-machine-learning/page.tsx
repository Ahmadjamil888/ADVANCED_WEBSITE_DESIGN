import { Metadata } from "next";
import { Badge } from "@/components/ui/badge";
import { Section } from "@/components/ui/section";
import { BrainCircuit, CheckCircle, ArrowRight } from "lucide-react";
import Link from "next/link";
import Navbar from "@/components/sections/navbar/default";
import Footer from "@/components/sections/footer/default";

export const metadata: Metadata = {
  title: "AI & Machine Learning Development Services - Zehan X Technologies",
  description: "Expert AI and machine learning development services. Custom ML models, predictive analytics, computer vision, NLP solutions, and intelligent automation for your business.",
  keywords: [
    "AI development services",
    "machine learning development",
    "custom ML models",
    "predictive analytics",
    "computer vision",
    "natural language processing",
    "AI automation",
    "intelligent systems",
    "artificial intelligence consulting",
    "ML model deployment",
    "data science services",
    "AI solutions company"
  ],
  openGraph: {
    title: "AI & Machine Learning Development Services - Zehan X Technologies",
    description: "Expert AI and machine learning development services. Custom ML models, predictive analytics, computer vision, NLP solutions, and intelligent automation.",
    type: "website",
    url: "https://zehanx.com/services/ai-machine-learning",
  },
  alternates: {
    canonical: "https://zehanx.com/services/ai-machine-learning",
  },
};

export default function AIMachineLearning() {
  return (
    <main className="min-h-screen w-full overflow-hidden bg-background text-foreground">
      <Navbar />
      
      {/* Hero Section */}
      <Section className="pt-24 pb-16">
        <div className="max-w-container mx-auto">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div>
              <Badge variant="outline" className="mb-6">
                <BrainCircuit className="mr-2 size-4" />
                AI & Machine Learning
              </Badge>
              <h1 className="text-4xl font-bold mb-6 sm:text-6xl">
                Transform Your Business with Intelligent AI Solutions
              </h1>
              <p className="text-xl text-muted-foreground mb-8">
                Harness the power of artificial intelligence and machine learning to automate processes, 
                gain insights, and drive innovation in your business operations.
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
                <BrainCircuit className="size-32 text-primary" />
              </div>
            </div>
          </div>
        </div>
      </Section>

      {/* Services Section */}
      <Section className="py-16 bg-muted/30">
        <div className="max-w-container mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Our AI & ML Services</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              We offer comprehensive AI and machine learning solutions tailored to your business needs.
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {[
              {
                title: "Custom ML Models",
                description: "Develop tailored machine learning models for your specific business requirements and data patterns."
              },
              {
                title: "Predictive Analytics",
                description: "Forecast trends, customer behavior, and business outcomes using advanced predictive algorithms."
              },
              {
                title: "Computer Vision",
                description: "Image recognition, object detection, and visual analysis solutions for various industries."
              },
              {
                title: "Natural Language Processing",
                description: "Text analysis, sentiment analysis, and language understanding capabilities."
              },
              {
                title: "Recommendation Systems",
                description: "Personalized recommendation engines to enhance user experience and increase engagement."
              },
              {
                title: "Automated Decision Making",
                description: "Intelligent systems that can make data-driven decisions automatically."
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

      {/* Process Section */}
      <Section className="py-16">
        <div className="max-w-container mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Our AI Development Process</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              We follow a structured approach to ensure successful AI implementation.
            </p>
          </div>
          
          <div className="grid md:grid-cols-4 gap-8">
            {[
              {
                step: "01",
                title: "Discovery & Analysis",
                description: "We analyze your business needs and data to identify AI opportunities."
              },
              {
                step: "02",
                title: "Data Preparation",
                description: "Clean, process, and prepare your data for machine learning algorithms."
              },
              {
                step: "03",
                title: "Model Development",
                description: "Build and train custom AI models tailored to your specific requirements."
              },
              {
                step: "04",
                title: "Deployment & Support",
                description: "Deploy the solution and provide ongoing support and optimization."
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
      <Section className="py-16 bg-muted/30">
        <div className="max-w-container mx-auto text-center">
          <h2 className="text-3xl font-bold mb-6">
            Ready to Implement AI in Your Business?
          </h2>
          <p className="text-muted-foreground mb-8 max-w-2xl mx-auto">
            Let's discuss how our AI and machine learning solutions can transform your operations.
          </p>
          <Link 
            href="/contact" 
            className="inline-flex items-center justify-center rounded-md bg-primary px-8 py-3 text-sm font-medium text-primary-foreground shadow transition-colors hover:bg-primary/90"
          >
            Start Your AI Journey
            <ArrowRight className="ml-2 size-4" />
          </Link>
        </div>
      </Section>

      <Footer />
    </main>
  );
}