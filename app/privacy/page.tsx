import { Badge } from "@/components/ui/badge";
import { Section } from "@/components/ui/section";
import { Shield } from "lucide-react";
import Navbar from "@/components/sections/navbar/default";
import Footer from "@/components/sections/footer/default";

export const metadata = {
  title: "Privacy Policy - Zehan X Technologies",
  description: "Privacy Policy for Zehan X Technologies. Learn how we collect, use, and protect your personal information.",
  keywords: ["privacy policy", "data protection", "Zehan X Technologies", "AI company privacy"],
};

export default function Privacy() {
  return (
    <main className="min-h-screen w-full overflow-hidden bg-background text-foreground">
      <Navbar />
      
      <Section className="pt-24 pb-16">
        <div className="max-w-container mx-auto">
          <div className="max-w-4xl mx-auto">
            <div className="text-center mb-12">
              <Badge variant="outline" className="mb-6">
                <Shield className="mr-2 size-4" />
                Privacy Policy
              </Badge>
              <h1 className="text-4xl font-bold mb-6">Privacy Policy</h1>
              <p className="text-muted-foreground">
                Last updated: January 2025
              </p>
            </div>

            <div className="prose prose-gray dark:prose-invert max-w-none">
              <section className="mb-8">
                <h2 className="text-2xl font-semibold mb-4">1. Information We Collect</h2>
                <p className="text-muted-foreground mb-4">
                  At Zehan X Technologies, we collect information you provide directly to us, such as when you:
                </p>
                <ul className="list-disc pl-6 text-muted-foreground space-y-2">
                  <li>Fill out our contact form</li>
                  <li>Create an account on our website</li>
                  <li>Subscribe to our newsletter</li>
                  <li>Communicate with us via email</li>
                </ul>
              </section>

              <section className="mb-8">
                <h2 className="text-2xl font-semibold mb-4">2. How We Use Your Information</h2>
                <p className="text-muted-foreground mb-4">
                  We use the information we collect to:
                </p>
                <ul className="list-disc pl-6 text-muted-foreground space-y-2">
                  <li>Respond to your inquiries and provide customer support</li>
                  <li>Send you technical notices and support messages</li>
                  <li>Communicate with you about our services</li>
                  <li>Improve our website and services</li>
                </ul>
              </section>

              <section className="mb-8">
                <h2 className="text-2xl font-semibold mb-4">3. Information Sharing</h2>
                <p className="text-muted-foreground mb-4">
                  We do not sell, trade, or otherwise transfer your personal information to third parties without your consent, except as described in this policy.
                </p>
              </section>

              <section className="mb-8">
                <h2 className="text-2xl font-semibold mb-4">4. Data Security</h2>
                <p className="text-muted-foreground mb-4">
                  We implement appropriate security measures to protect your personal information against unauthorized access, alteration, disclosure, or destruction.
                </p>
              </section>

              <section className="mb-8">
                <h2 className="text-2xl font-semibold mb-4">5. Contact Us</h2>
                <p className="text-muted-foreground mb-4">
                  If you have any questions about this Privacy Policy, please contact us at:
                </p>
                <p className="text-muted-foreground">
                  Email: <a href="/contact" className="text-primary hover:underline">Contact Us</a>
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