import { Badge } from "@/components/ui/badge";
import { Section } from "@/components/ui/section";
import { BrainCircuit, Code, Users, Award, Target, Lightbulb } from "lucide-react";
import Link from "next/link";

export default function About() {
  return (
    <main className="min-h-screen w-full overflow-hidden bg-background text-foreground">
      {/* Hero Section */}
      <Section className="pt-24 pb-16">
        <div className="max-w-container mx-auto text-center">
          <Badge variant="outline" className="mb-6">
            <Users className="mr-2 size-4" />
            About Us
          </Badge>
          <h1 className="text-4xl font-bold mb-6 sm:text-6xl">
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
              <Badge variant="outline" className="mb-4">
                <Target className="mr-2 size-4" />
                Our Mission
              </Badge>
              <h2 className="text-3xl font-bold mb-6">
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
              <div className="p-6 bg-background rounded-lg border">
                <BrainCircuit className="size-8 text-primary mb-4" />
                <h3 className="font-semibold mb-2">AI Expertise</h3>
                <p className="text-sm text-muted-foreground">
                  Deep learning, machine learning, and neural networks
                </p>
              </div>
              <div className="p-6 bg-background rounded-lg border">
                <Code className="size-8 text-primary mb-4" />
                <h3 className="font-semibold mb-2">Web Development</h3>
                <p className="text-sm text-muted-foreground">
                  Next.js, React, and modern web technologies
                </p>
              </div>
              <div className="p-6 bg-background rounded-lg border">
                <Award className="size-8 text-primary mb-4" />
                <h3 className="font-semibold mb-2">Quality Focused</h3>
                <p className="text-sm text-muted-foreground">
                  Enterprise-grade solutions with 99% satisfaction
                </p>
              </div>
              <div className="p-6 bg-background rounded-lg border">
                <Lightbulb className="size-8 text-primary mb-4" />
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
          <Badge variant="outline" className="mb-6">
            <Award className="mr-2 size-4" />
            Our Values
          </Badge>
          <h2 className="text-3xl font-bold mb-12">
            What Drives Us Forward
          </h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="p-8 rounded-lg border bg-background">
              <div className="size-12 bg-primary/10 rounded-lg flex items-center justify-center mx-auto mb-4">
                <BrainCircuit className="size-6 text-primary" />
              </div>
              <h3 className="text-xl font-semibold mb-4">Innovation First</h3>
              <p className="text-muted-foreground">
                We stay at the forefront of AI and web technologies, constantly exploring 
                new possibilities to deliver breakthrough solutions.
              </p>
            </div>
            <div className="p-8 rounded-lg border bg-background">
              <div className="size-12 bg-primary/10 rounded-lg flex items-center justify-center mx-auto mb-4">
                <Users className="size-6 text-primary" />
              </div>
              <h3 className="text-xl font-semibold mb-4">Client Success</h3>
              <p className="text-muted-foreground">
                Your success is our success. We work closely with clients to understand 
                their unique challenges and deliver tailored solutions.
              </p>
            </div>
            <div className="p-8 rounded-lg border bg-background">
              <div className="size-12 bg-primary/10 rounded-lg flex items-center justify-center mx-auto mb-4">
                <Award className="size-6 text-primary" />
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
          <h2 className="text-3xl font-bold mb-6">
            Ready to Start Your AI Journey?
          </h2>
          <p className="text-muted-foreground mb-8 max-w-2xl mx-auto">
            Let's discuss how Zehan X Technologies can help transform your business 
            with cutting-edge AI solutions and modern web development.
          </p>
          <div className="flex justify-center gap-4">
            <Link 
              href="/#contact" 
              className="inline-flex items-center justify-center rounded-md bg-primary px-8 py-3 text-sm font-medium text-primary-foreground shadow transition-colors hover:bg-primary/90"
            >
              Get Started
            </Link>
            <Link 
              href="/" 
              className="inline-flex items-center justify-center rounded-md border border-input bg-background px-8 py-3 text-sm font-medium shadow-sm transition-colors hover:bg-accent hover:text-accent-foreground"
            >
              Back to Home
            </Link>
          </div>
        </div>
      </Section>
    </main>
  );
}