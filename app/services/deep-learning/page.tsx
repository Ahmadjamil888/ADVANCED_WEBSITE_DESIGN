import { Badge } from "@/components/ui/badge";
import { Section } from "@/components/ui/section";
import { Database, CheckCircle, ArrowRight } from "lucide-react";
import Link from "next/link";
import Navbar from "@/components/sections/navbar/default";
import Footer from "@/components/sections/footer/default";

export default function DeepLearning() {
  return (
    <main className="min-h-screen w-full overflow-hidden bg-background text-foreground">
      <Navbar />
      
      {/* Hero Section */}
      <Section className="pt-24 pb-16">
        <div className="max-w-container mx-auto">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div>
              <Badge variant="outline" className="mb-6">
                <Database className="mr-2 size-4" />
                Deep Learning
              </Badge>
              <h1 className="text-4xl font-bold mb-6 sm:text-6xl">
                Advanced Neural Networks & Deep Learning Solutions
              </h1>
              <p className="text-xl text-muted-foreground mb-8">
                Harness the power of deep learning for complex pattern recognition, 
                computer vision, natural language processing, and intelligent decision making.
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
              <div className="size-64 mx-auto bg-gradient-to-br from-blue-600/20 to-cyan-600/20 rounded-full flex items-center justify-center">
                <Database className="size-32 text-primary" />
              </div>
            </div>
          </div>
        </div>
      </Section>

      {/* Services Section */}
      <Section className="py-16 bg-muted/30">
        <div className="max-w-container mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Deep Learning Applications</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              We develop sophisticated deep learning models for various industry applications.
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {[
              {
                title: "Computer Vision",
                description: "Image classification, object detection, facial recognition, and medical image analysis."
              },
              {
                title: "Natural Language Processing",
                description: "Text analysis, language translation, sentiment analysis, and chatbot development."
              },
              {
                title: "Speech Recognition",
                description: "Voice-to-text conversion, speech synthesis, and audio processing solutions."
              },
              {
                title: "Recommendation Systems",
                description: "Personalized content recommendations using deep neural networks."
              },
              {
                title: "Predictive Analytics",
                description: "Time series forecasting and predictive modeling using deep learning."
              },
              {
                title: "Autonomous Systems",
                description: "Self-driving algorithms, robotics control, and intelligent automation."
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

      {/* CTA Section */}
      <Section className="py-16">
        <div className="max-w-container mx-auto text-center">
          <h2 className="text-3xl font-bold mb-6">
            Ready to Implement Deep Learning?
          </h2>
          <p className="text-muted-foreground mb-8 max-w-2xl mx-auto">
            Let's discuss how deep learning can solve your complex business challenges.
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