"use client";

import { Badge } from "@/components/ui/badge";
import { Section } from "@/components/ui/section";
import { Mail, MapPin, Clock, Phone } from "lucide-react";
import Navbar from "@/components/sections/navbar/default";
import Footer from "@/components/sections/footer/default";

export default function Contact() {
  const CONTACT_EMAIL =
    process.env.NEXT_PUBLIC_CONTACT_EMAIL || "zehanxtech@gmail.com";
  const RESPONSE_TIME =
    process.env.NEXT_PUBLIC_CONTACT_RESPONSE_TIME || "Within 24 hours";
  const WORKING_HOURS =
    process.env.NEXT_PUBLIC_CONTACT_HOURS || "Mon–Fri, 9 AM – 6 PM";
  const LOCATION =
    process.env.NEXT_PUBLIC_CONTACT_LOCATION || "Remote & Global";
  const CONTACT_PHONE = "+92 344 2693910";

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
            Ready to transform your business with AI, creative design, and modern web development?
            Get in touch with our team and let's discuss your project.
          </p>
        </div>
      </Section>

      {/* Contact Info */}
      <Section className="py-16">
        <div className="max-w-container mx-auto grid lg:grid-cols-2 gap-12">
          <div className="lg:col-span-2">
            <h2 className="text-2xl font-bold mb-6">Get in touch</h2>

            <ContactInfo
              icon={<Mail className="size-6 text-primary" />}
              title="Email"
              line1={
                <a
                  href={`mailto:${CONTACT_EMAIL}`}
                  className="hover:underline text-primary"
                >
                  {CONTACT_EMAIL}
                </a>
              }
              line2="We'll respond quickly"
            />
            <ContactInfo
              icon={<Phone className="size-6 text-primary" />}
              title="Phone"
              line1={
                <a
                  href={`tel:${CONTACT_PHONE.replace(/\s+/g, "")}`}
                  className="hover:underline text-primary"
                >
                  {CONTACT_PHONE}
                </a>
              }
              line2="Available during working hours"
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
                  ["Graphic Designing", "/services/graphic-designing"],
                  ["Video Editing", "/services/video-editing"],
                  ["Content Writing", "/services/content-writing"],
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

type ContactInfoProps = {
  icon: React.ReactNode;
  title: string;
  line1: React.ReactNode;
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
