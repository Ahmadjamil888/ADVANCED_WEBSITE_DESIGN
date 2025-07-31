import { Badge } from "@/components/ui/badge";
import { Section } from "@/components/ui/section";
import { BrainCircuit, Code, Users, Award, Target, Lightbulb } from "lucide-react";
import Link from "next/link";
import Navbar from "@/components/sections/navbar/default";
import Footer from "@/components/sections/footer/default";

export default function About() {
  return (
    <main className="min-h-screen w-full overflow-hidden bg-background text-foreground">
      <Navbar />
      {/* Hero Section */}
      <Section className="pt-24 pb-16">
        <div className="max-w-container mx-auto text-center">
          <Badge variant="outline" className="mb-6 border-blue-500/20 bg-blue-500/10 hover-glow">
            <Users className="mr-2 size-4 text-blue-400" />
            <span className="text-blue-400 font-medium">About Us</span>
          </Badge>
          <h1 className="text-4xl font-bold mb-6 sm:text-6xl text-gradient-blue">
            Meet Zehan X Technologies
          </h1>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            We are a cutting-edge AI and web development company dedicated to transforming 
            businesses through intelligent technology solutions and modern web applications.
          </p>
        </div>
      </Section>

      {/* Mission Section */}
      <Section className="py-16 bg-muted/30">
        <div className="max-w-container mx-auto">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div>
              <Badge variant="outline" className="mb-4 border-green-500/20 bg-green-500/10">
                <Target className="mr-2 size-4 text-green-400" />
                <span className="text-green-400 font-medium">Our Mission</span>
              </Badge>
              <h2 className="text-3xl font-bold mb-6 text-gradient-green">
                Empowering Businesses with AI Innovation
              </h2>
              <p className="text-muted-foreground mb-6">
                At Zehan X Technologies, we believe that artificial intelligence and modern web 
                technologies should be accessible to every business, regardless of size or industry. 
                Our mission is to democratize AI by creating intelligent solutions that solve real-world problems.
              </p>
              <p className="text-muted-foreground">
                We combine deep technical expertise with a passion for innovation to deliver 
                AI-powered applications that drive growth, efficiency, and competitive advantage.
              </p>
            </div>
            <div className="grid grid-cols-2 gap-6">
              <div className="enhanced-card p-6 rounded-lg hover-lift">
                <BrainCircuit className="size-8 text-blue-400 mb-4" />
                <h3 className="font-semibold mb-2">AI Expertise</h3>
                <p className="text-sm text-muted-foreground">
                  Deep learning, machine learning, and neural networks
                </p>
              </div>
              <div className="enhanced-card p-6 rounded-lg hover-lift">
                <Code className="size-8 text-green-400 mb-4" />
                <h3 className="font-semibold mb-2">Web Development</h3>
                <p className="text-sm text-muted-foreground">
                  Next.js, React, and modern web technologies
                </p>
              </div>
              <div className="enhanced-card p-6 rounded-lg hover-lift">
                <Award className="size-8 text-purple-400 mb-4" />
                <h3 className="font-semibold mb-2">Quality Focused</h3>
                <p className="text-sm text-muted-foreground">
                  Enterprise-grade solutions with 99% satisfaction
                </p>
              </div>
              <div className="enhanced-card p-6 rounded-lg hover-lift">
                <Lightbulb className="size-8 text-yellow-400 mb-4" />
                <h3 className="font-semibold mb-2">Innovation</h3>
                <p className="text-sm text-muted-foreground">
                  Cutting-edge technologies and creative solutions
                </p>
              </div>
            </div>
          </div>
        </div>
      </Section>

      {/* Values Section */}
      <Section className="py-16">
        <div className="max-w-container mx-auto text-center">
          <Badge variant="outline" className="mb-6 border-purple-500/20 bg-purple-500/10">
            <Award className="mr-2 size-4 text-purple-400" />
            <span className="text-purple-400 font-medium">Our Values</span>
          </Badge>
          <h2 className="text-3xl font-bold mb-12 text-gradient-blue">
            What Drives Us Forward
          </h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="enhanced-card p-8 rounded-lg hover-lift glow-blue">
              <div className="size-12 bg-blue-500/10 rounded-lg flex items-center justify-center mx-auto mb-4">
                <BrainCircuit className="size-6 text-blue-400" />
              </div>
              <h3 className="text-xl font-semibold mb-4">Innovation First</h3>
              <p className="text-muted-foreground">
                We stay at the forefront of AI and web technologies, constantly exploring 
                new possibilities to deliver breakthrough solutions.
              </p>
            </div>
            <div className="enhanced-card p-8 rounded-lg hover-lift glow-green">
              <div className="size-12 bg-green-500/10 rounded-lg flex items-center justify-center mx-auto mb-4">
                <Users className="size-6 text-green-400" />
              </div>
              <h3 className="text-xl font-semibold mb-4">Client Success</h3>
              <p className="text-muted-foreground">
                Your success is our success. We work closely with clients to understand 
                their unique challenges and deliver tailored solutions.
              </p>
            </div>
            <div className="enhanced-card p-8 rounded-lg hover-lift glow-purple">
              <div className="size-12 bg-purple-500/10 rounded-lg flex items-center justify-center mx-auto mb-4">
                <Award className="size-6 text-purple-400" />
              </div>
              <h3 className="text-xl font-semibold mb-4">Excellence</h3>
              <p className="text-muted-foreground">
                We maintain the highest standards in code quality, security, and 
                performance across all our projects.
              </p>
            </div>
          </div>
        </div>
      </Section>

      {/* CTA Section */}
      <Section className="py-16 bg-muted/30">
        <div className="max-w-container mx-auto text-center">
          <h2 className="text-3xl font-bold mb-6 text-gradient-green">
            Ready to Start Your AI Journey?
          </h2>
          <p className="text-muted-foreground mb-8 max-w-2xl mx-auto">
            Let's discuss how Zehan X Technologies can help transform your business 
            with cutting-edge AI solutions and modern web development.
          </p>
          <div className="flex justify-center gap-4">
            <Link 
              href="/#contact" 
              className="inline-flex items-center justify-center rounded-md btn-gradient-secondary hover-lift glow-green px-8 py-3 text-sm font-medium text-white shadow transition-colors"
            >
              Get Started
            </Link>
            <Link 
              href="/" 
              className="inline-flex items-center justify-center rounded-md border border-border/50 hover:border-blue-500/50 bg-background px-8 py-3 text-sm font-medium shadow-sm transition-colors hover-lift"
            >
              Back to Home
            </Link>
          </div>
        </div>
      </Section>
      
      <Footer />
    </main>
  );
}