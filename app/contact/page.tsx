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

  const CONTACT_EMAIL =
    process.env.NEXT_PUBLIC_CONTACT_EMAIL || "contact@example.com";
  const RESPONSE_TIME =
    process.env.NEXT_PUBLIC_CONTACT_RESPONSE_TIME || "Within 24 hours";
  const WORKING_HOURS =
    process.env.NEXT_PUBLIC_CONTACT_HOURS || "Mon–Fri, 9 AM – 6 PM";
  const LOCATION =
    process.env.NEXT_PUBLIC_CONTACT_LOCATION || "Remote & Global";

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError("");

    try {
      const response = await fetch("/api/contact", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });

      const result = await response.json();

      if (response.ok && result.success) {
        setIsSuccess(true);
        setFormData({ email: "", name: "", company: "", message: "" });
      } else {
        setError(result.error || "Something went wrong. Please try again.");
      }
    } catch (err) {
      setError("Network error. Please try again.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  if (isSuccess) {
    return (
      <main className="min-h-screen w-full bg-background text-foreground">
        <Navbar />
        <Section className="pt-24 pb-16">
          <div className="max-w-container mx-auto text-center">
            <div className="flex flex-col items-center gap-6">
              <CheckCircle className="size-16 text-green-500" />
              <h1 className="text-4xl font-bold">Thank You!</h1>
              <p className="text-muted-foreground max-w-md text-lg">
                Your message has been sent successfully. We'll get back to you
                soon!
              </p>
              <Button
                onClick={() => setIsSuccess(false)}
                variant="outline"
                size="lg"
              >
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
    <main className="min-h-screen w-full bg-background text-foreground">
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
            Get in touch with our team and let's discuss your project.
          </p>
        </div>
      </Section>

      {/* Contact Form & Info */}
      <Section className="py-16">
        <div className="max-w-container mx-auto grid lg:grid-cols-2 gap-12">
          {/* Contact Form */}
          <div>
            <h2 className="text-2xl font-bold mb-6">Send us a message</h2>
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <InputField
                  id="name"
                  name="name"
                  label="Full Name *"
                  placeholder="Your full name"
                  value={formData.name}
                  onChange={handleChange}
                  required
                />
                <InputField
                  id="email"
                  name="email"
                  type="email"
                  label="Email Address *"
                  placeholder="your@email.com"
                  value={formData.email}
                  onChange={handleChange}
                  required
                />
              </div>

              <InputField
                id="company"
                name="company"
                label="Company (Optional)"
                placeholder="Your company name"
                value={formData.company}
                onChange={handleChange}
              />

              <div>
                <label
                  htmlFor="message"
                  className="block text-sm font-medium mb-2"
                >
                  Project Details *
                </label>
                <Textarea
                  id="message"
                  name="message"
                  placeholder="Tell us about your project, requirements, timeline..."
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
                disabled={
                  isSubmitting ||
                  !formData.email ||
                  !formData.name ||
                  !formData.message
                }
                className="w-full"
                size="lg"
              >
                {isSubmitting ? (
                  "Sending..."
                ) : (
                  <>
                    Send Message <Send className="ml-2 size-4" />
                  </>
                )}
              </Button>
            </form>
          </div>

          {/* Contact Info */}
          <div className="lg:pl-8">
            <h2 className="text-2xl font-bold mb-6">Get in touch</h2>
            <ContactInfo
              icon={<Mail className="size-6 text-primary" />}
              title="Email"
              line1={CONTACT_EMAIL}
              line2="We'll respond quickly"
            />
            <ContactInfo
              icon={<Clock className="size-6 text-primary" />}
              title="Response Time"
              line1={RESPONSE_TIME}
              line2={WORKING_HOURS}
            />
            <ContactInfo
              icon={<MapPin className="size-6 text-primary" />}
              title="Location"
              line1={LOCATION}
              line2="Serving clients worldwide"
            />

            <div className="mt-8 p-6 bg-muted/30 rounded-lg">
              <h3 className="font-semibold mb-4">Our Services</h3>
              <div className="grid grid-cols-2 gap-2 text-sm">
                {[
                  ["AI & Machine Learning", "/services/ai-machine-learning"],
                  ["Next.js Development", "/services/nextjs-development"],
                  ["Full-Stack Development", "/services/fullstack-web-development"],
                  ["Deep Learning", "/services/deep-learning"],
                  ["AI Chatbots", "/services/ai-chatbots"],
                  ["AI Consulting", "/services/ai-consulting"],
                ].map(([label, href]) => (
                  <a
                    key={href}
                    href={href}
                    className="text-muted-foreground hover:text-primary"
                  >
                    {label}
                  </a>
                ))}
              </div>
            </div>
          </div>
        </div>
      </Section>

      <Footer />
    </main>
  );
}

type InputFieldProps = {
  id: string;
  name: string;
  type?: string;
  label: string;
  placeholder?: string;
  value: string;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  required?: boolean;
};

function InputField({
  id,
  name,
  type = "text",
  label,
  placeholder,
  value,
  onChange,
  required = false,
}: InputFieldProps) {
  return (
    <div>
      <label htmlFor={id} className="block text-sm font-medium mb-2">
        {label}
      </label>
      <Input
        id={id}
        name={name}
        type={type}
        placeholder={placeholder}
        value={value}
        onChange={onChange}
        required={required}
      />
    </div>
  );
}

type ContactInfoProps = {
  icon: React.ReactNode;
  title: string;
  line1: string;
  line2?: string;
};

function ContactInfo({ icon, title, line1, line2 }: ContactInfoProps) {
  return (
    <div className="flex items-start gap-4 mb-6">
      <div className="size-12 bg-primary/10 rounded-lg flex items-center justify-center flex-shrink-0">
        {icon}
      </div>
      <div>
        <h3 className="font-semibold mb-1">{title}</h3>
        <p className="text-muted-foreground">{line1}</p>
        {line2 && <p className="text-sm text-muted-foreground">{line2}</p>}
      </div>
    </div>
  );
}
