"use client";

import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Section } from "@/components/ui/section";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Mail, Send, CheckCircle, MapPin, Clock } from "lucide-react";
import Navbar from "@/components/sections/navbar/default";
import Footer from "@/components/sections/footer/default";

export default function Contact() {
  const [formData, setFormData] = useState({
    email: "",
    name: "",
    company: "",
    message: "",
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError("");

    try {
      const response = await fetch("/api/subscribe", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      const result = await response.json();

      if (result.success) {
        setIsSuccess(true);
        setFormData({ email: "", name: "", company: "", message: "" });
      } else {
        setError(result.message || "Something went wrong");
      }
    } catch {
      setError("Failed to send message. Please try again.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  if (isSuccess) {
    return (
      <main className="min-h-screen w-full overflow-hidden bg-background text-foreground">
        <Navbar />
        <Section className="pt-24 pb-16">
          <div className="max-w-container mx-auto text-center">
            <div className="flex flex-col items-center gap-6">
              <CheckCircle className="size-16 text-green-500" />
              <h1 className="text-4xl font-bold">Thank You!</h1>
              <p className="text-muted-foreground max-w-md text-lg">
                Your message has been sent successfully. We'll get back to you within 24 hours!
              </p>
              <Button onClick={() => setIsSuccess(false)} variant="outline" size="lg">
                Send Another Message
              </Button>
            </div>
          </div>
        </Section>
        <Footer />
      </main>
    );
  }

  return (
    <main className="min-h-screen w-full overflow-hidden bg-background text-foreground">
      <Navbar />
      
      {/* Hero Section */}
      <Section className="pt-24 pb-16">
        <div className="max-w-container mx-auto text-center">
          <Badge variant="outline" className="mb-6">
            <Mail className="mr-2 size-4" />
            Contact Us
          </Badge>
          <h1 className="text-4xl font-bold mb-6 sm:text-6xl">
            Let's Build Something Amazing Together
          </h1>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Ready to transform your business with AI and modern web development? 
            Get in touch with our team of experts and let's discuss your project.
          </p>
        </div>
      </Section>

      {/* Contact Form & Info */}
      <Section className="py-16">
        <div className="max-w-container mx-auto">
          <div className="grid lg:grid-cols-2 gap-12">
            {/* Contact Form */}
            <div>
              <h2 className="text-2xl font-bold mb-6">Send us a message</h2>
              <form onSubmit={handleSubmit} className="space-y-6">
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div>
                    <label htmlFor="name" className="block text-sm font-medium mb-2">
                      Full Name *
                    </label>
                    <Input
                      id="name"
                      type="text"
                      name="name"
                      placeholder="Your full name"
                      value={formData.name}
                      onChange={handleChange}
                      required
                    />
                  </div>
                  <div>
                    <label htmlFor="email" className="block text-sm font-medium mb-2">
                      Email Address *
                    </label>
                    <Input
                      id="email"
                      type="email"
                      name="email"
                      placeholder="your@email.com"
                      value={formData.email}
                      onChange={handleChange}
                      required
                    />
                  </div>
                </div>
                
                <div>
                  <label htmlFor="company" className="block text-sm font-medium mb-2">
                    Company (Optional)
                  </label>
                  <Input
                    id="company"
                    type="text"
                    name="company"
                    placeholder="Your company name"
                    value={formData.company}
                    onChange={handleChange}
                  />
                </div>
                
                <div>
                  <label htmlFor="message" className="block text-sm font-medium mb-2">
                    Project Details *
                  </label>
                  <Textarea
                    id="message"
                    name="message"
                    placeholder="Tell us about your project, requirements, timeline, and how we can help..."
                    value={formData.message}
                    onChange={handleChange}
                    rows={6}
                    required
                  />
                </div>

                {error && (
                  <div className="text-red-500 text-sm bg-red-50 dark:bg-red-900/20 p-3 rounded-md">
                    {error}
                  </div>
                )}

                <Button
                  type="submit"
                  disabled={isSubmitting || !formData.email || !formData.name || !formData.message}
                  className="w-full"
                  size="lg"
                >
                  {isSubmitting ? (
                    "Sending..."
                  ) : (
                    <>
                      Send Message
                      <Send className="ml-2 size-4" />
                    </>
                  )}
                </Button>
              </form>
            </div>

            {/* Contact Information */}
            <div className="lg:pl-8">
              <h2 className="text-2xl font-bold mb-6">Get in touch</h2>
              <div className="space-y-6">
                <div className="flex items-start gap-4">
                  <div className="size-12 bg-primary/10 rounded-lg flex items-center justify-center flex-shrink-0">
                    <Mail className="size-6 text-primary" />
                  </div>
                  <div>
                    <h3 className="font-semibold mb-1">Email</h3>
                    <p className="text-muted-foreground">shazabjamildhami@gmail.com</p>
                    <p className="text-sm text-muted-foreground">We'll respond within 24 hours</p>
                  </div>
                </div>

                <div className="flex items-start gap-4">
                  <div className="size-12 bg-primary/10 rounded-lg flex items-center justify-center flex-shrink-0">
                    <Clock className="size-6 text-primary" />
                  </div>
                  <div>
                    <h3 className="font-semibold mb-1">Response Time</h3>
                    <p className="text-muted-foreground">Within 24 hours</p>
                    <p className="text-sm text-muted-foreground">Monday to Friday, 9 AM - 6 PM</p>
                  </div>
                </div>

                <div className="flex items-start gap-4">
                  <div className="size-12 bg-primary/10 rounded-lg flex items-center justify-center flex-shrink-0">
                    <MapPin className="size-6 text-primary" />
                  </div>
                  <div>
                    <h3 className="font-semibold mb-1">Location</h3>
                    <p className="text-muted-foreground">Remote & Global</p>
                    <p className="text-sm text-muted-foreground">Serving clients worldwide</p>
                  </div>
                </div>
              </div>

              {/* Services Quick Links */}
              <div className="mt-8 p-6 bg-muted/30 rounded-lg">
                <h3 className="font-semibold mb-4">Our Services</h3>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <a href="/services/ai-machine-learning" className="text-muted-foreground hover:text-primary">AI & Machine Learning</a>
                  <a href="/services/nextjs-development" className="text-muted-foreground hover:text-primary">Next.js Development</a>
                  <a href="/services/fullstack-web-development" className="text-muted-foreground hover:text-primary">Full-Stack Development</a>
                  <a href="/services/deep-learning" className="text-muted-foreground hover:text-primary">Deep Learning</a>
                  <a href="/services/ai-chatbots" className="text-muted-foreground hover:text-primary">AI Chatbots</a>
                  <a href="/services/ai-consulting" className="text-muted-foreground hover:text-primary">AI Consulting</a>
                </div>
              </div>
            </div>
          </div>
        </div>
      </Section>

      <Footer />
    </main>
  );
}