import { Badge } from "@/components/ui/badge";
import { Section } from "@/components/ui/section";
import { FileText } from "lucide-react";
import Navbar from "@/components/sections/navbar/default";
import Footer from "@/components/sections/footer/default";

export const metadata = {
  title: "Terms of Service - Zehan X Technologies",
  description: "Terms of Service for Zehan X Technologies. Read our terms and conditions for using our AI and web development services.",
  keywords: ["terms of service", "terms and conditions", "Zehan X Technologies", "AI services terms"],
};

export default function Terms() {
  return (
    <main className="min-h-screen w-full overflow-hidden bg-background text-foreground">
      <Navbar />
      
      <Section className="pt-24 pb-16">
        <div className="max-w-container mx-auto">
          <div className="max-w-4xl mx-auto">
            <div className="text-center mb-12">
              <Badge variant="outline" className="mb-6">
                <FileText className="mr-2 size-4" />
                Terms of Service
              </Badge>
              <h1 className="text-4xl font-bold mb-6">Terms of Service</h1>
              <p className="text-muted-foreground">
                Last updated: January 2025
              </p>
            </div>

            <div className="prose prose-gray dark:prose-invert max-w-none">
              <section className="mb-8">
                <h2 className="text-2xl font-semibold mb-4">1. Acceptance of Terms</h2>
                <p className="text-muted-foreground mb-4">
                  By accessing and using the services provided by Zehan X Technologies, you accept and agree to be bound by the terms and provision of this agreement.
                </p>
              </section>

              <section className="mb-8">
                <h2 className="text-2xl font-semibold mb-4">2. Services</h2>
                <p className="text-muted-foreground mb-4">
                  Zehan X Technologies provides AI and web development services including but not limited to:
                </p>
                <ul className="list-disc pl-6 text-muted-foreground space-y-2">
                  <li>AI & Machine Learning solutions</li>
                  <li>Next.js and full-stack web development</li>
                  <li>Deep learning implementations</li>
                  <li>AI chatbot development</li>
                  <li>AI consulting services</li>
                  <li>Data analytics solutions</li>
                </ul>
              </section>

              <section className="mb-8">
                <h2 className="text-2xl font-semibold mb-4">3. User Responsibilities</h2>
                <p className="text-muted-foreground mb-4">
                  You are responsible for:
                </p>
                <ul className="list-disc pl-6 text-muted-foreground space-y-2">
                  <li>Providing accurate information when requesting services</li>
                  <li>Complying with all applicable laws and regulations</li>
                  <li>Respecting intellectual property rights</li>
                  <li>Using our services in a lawful manner</li>
                </ul>
              </section>

              <section className="mb-8">
                <h2 className="text-2xl font-semibold mb-4">4. Intellectual Property</h2>
                <p className="text-muted-foreground mb-4">
                  All content, features, and functionality of our services are owned by Zehan X Technologies and are protected by copyright, trademark, and other intellectual property laws.
                </p>
              </section>

              <section className="mb-8">
                <h2 className="text-2xl font-semibold mb-4">5. Limitation of Liability</h2>
                <p className="text-muted-foreground mb-4">
                  Zehan X Technologies shall not be liable for any indirect, incidental, special, consequential, or punitive damages resulting from your use of our services.
                </p>
              </section>

              <section className="mb-8">
                <h2 className="text-2xl font-semibold mb-4">6. Contact Information</h2>
                <p className="text-muted-foreground mb-4">
                  For questions about these Terms of Service, please contact us at:
                </p>
                <p className="text-muted-foreground">
                  Email: <a href="mailto:shazabjamildhami@gmail.com" className="text-primary hover:underline">shazabjamildhami@gmail.com</a>
                </p>
              </section>
            </div>
          </div>
        </div>
      </Section>

      <Footer />
    </main>
  );
}