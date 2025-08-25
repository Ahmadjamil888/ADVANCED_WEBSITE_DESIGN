import { Badge } from "@/components/ui/badge";
import { Section } from "@/components/ui/section";
import { Shield, CheckCircle, ArrowRight } from "lucide-react";
import Link from "next/link";
import Navbar from "@/components/sections/navbar/default";
import Footer from "@/components/sections/footer/default";

export default function EnterpriseSolutions() {
  return (
    <main className="min-h-screen w-full overflow-hidden bg-background text-foreground">
      <Navbar />
      
      <Section className="pt-24 pb-16">
        <div className="max-w-container mx-auto">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div>
              <Badge variant="outline" className="mb-6">
                <Shield className="mr-2 size-4" />
                Enterprise Solutions
              </Badge>
              <h1 className="text-4xl font-bold mb-6 sm:text-6xl">
                Scalable Enterprise-Grade Solutions
              </h1>
              <p className="text-xl text-muted-foreground mb-8">
                Build robust, secure, and scalable enterprise applications that handle 
                mission-critical operations with enterprise-grade reliability and performance.
              </p>
              <div className="flex gap-4">
                <Link href="/contact" className="inline-flex items-center justify-center rounded-md bg-primary px-6 py-3 text-sm font-medium text-primary-foreground shadow transition-colors hover:bg-primary/90">
                  Get Started <ArrowRight className="ml-2 size-4" />
                </Link>
                <Link href="/services" className="inline-flex items-center justify-center rounded-md border border-input bg-background px-6 py-3 text-sm font-medium shadow-sm transition-colors hover:bg-accent hover:text-accent-foreground">
                  All Services
                </Link>
              </div>
            </div>
            <div className="relative">
              <div className="size-64 mx-auto bg-gradient-to-br from-blue-600/20 to-cyan-600/20 rounded-full flex items-center justify-center">
                <Shield className="size-32 text-primary" />
              </div>
            </div>
          </div>
        </div>
      </Section>

      <Section className="py-16 bg-muted/30">
        <div className="max-w-container mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Enterprise Solutions</h2>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {[
              { title: "Enterprise Architecture", description: "Scalable system architecture designed for enterprise-level operations." },
              { title: "Security Implementation", description: "Comprehensive security measures and compliance for enterprise applications." },
              { title: "24/7 Support", description: "Round-the-clock technical support and monitoring for critical systems." }
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

      <Footer />
    </main>
  );
}