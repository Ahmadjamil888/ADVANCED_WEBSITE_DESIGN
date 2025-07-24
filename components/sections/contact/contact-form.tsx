"use client";

import { useState } from "react";
import { Button } from "../../ui/button";
import { Input } from "../../ui/input";
import { Textarea } from "../../ui/textarea";
import { Section } from "../../ui/section";
import { Badge } from "../../ui/badge";
import { Mail, Send, CheckCircle } from "lucide-react";

export default function ContactForm() {
  const [formData, setFormData] = useState({
    email: "",
    name: "",
    message: "",
    company: "",
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError("");

    try {
      const response = await fetch("/api/contact-simple", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      const result = await response.json();

      if (result.success) {
        setIsSuccess(true);
        setFormData({ email: "", name: "", message: "", company: "" });
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
      <Section id="contact" className="py-24 bg-muted/30">
        <div className="max-w-container mx-auto text-center">
          <div className="flex flex-col items-center gap-6">
            <CheckCircle className="size-16 text-green-500" />
            <h2 className="text-3xl font-bold">Thank You!</h2>
            <p className="text-muted-foreground max-w-md">
              Your message has been sent successfully. We'll get back to you within 24 hours!
            </p>
            <Button onClick={() => setIsSuccess(false)} variant="outline">
              Send Another Message
            </Button>
          </div>
        </div>
      </Section>
    );
  }

  return (
    <Section id="contact" className="py-24 bg-muted/30">
      <div className="max-w-container mx-auto">
        <div className="text-center mb-12">
          <Badge variant="outline" className="mb-4 border-green-500/20 bg-green-500/10 hover-glow">
            <Mail className="mr-2 size-4 text-green-400" />
            <span className="text-green-400">Get In Touch</span>
          </Badge>
          <h2 className="text-3xl font-bold mb-4 sm:text-5xl text-gradient-green">
            Let's Build Something Amazing Together
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Ready to transform your business with AI? Send us a message and let's discuss 
            how we can help you achieve your goals with cutting-edge technology.
          </p>
        </div>

        <div className="max-w-lg mx-auto">
          <div className="enhanced-card p-6 rounded-lg">
            <form onSubmit={handleSubmit} className="space-y-6">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <Input
                  type="text"
                  name="name"
                  placeholder="Your name"
                  value={formData.name}
                  onChange={handleChange}
                  required
                  className="w-full"
                />
              </div>
              <div>
                <Input
                  type="email"
                  name="email"
                  placeholder="Your email address"
                  value={formData.email}
                  onChange={handleChange}
                  required
                  className="w-full"
                />
              </div>
            </div>
            <div>
              <Input
                type="text"
                name="company"
                placeholder="Your company (optional)"
                value={formData.company}
                onChange={handleChange}
                className="w-full"
              />
            </div>
            
            <div>
              <Textarea
                name="message"
                placeholder="Tell us about your project and how we can help..."
                value={formData.message}
                onChange={handleChange}
                rows={5}
                className="w-full"
                required
              />
            </div>

            {error && (
              <div className="text-red-500 text-sm text-center bg-red-50 dark:bg-red-900/20 p-3 rounded-md">
                {error}
              </div>
            )}

            <Button
              type="submit"
              disabled={isSubmitting || !formData.email || !formData.name}
              className="w-full btn-gradient-secondary hover-lift glow-green"
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

          <div className="mt-8 text-center">
            <p className="text-muted-foreground text-sm">
              Or email us directly at{" "}
              <a 
                href="mailto:shazabjamildhami@gmail.com" 
                className="text-primary hover:underline"
              >
                shazabjamildhami@gmail.com
              </a>
            </p>
          </div>
        </div>
      </div>
    </Section>
  );
}