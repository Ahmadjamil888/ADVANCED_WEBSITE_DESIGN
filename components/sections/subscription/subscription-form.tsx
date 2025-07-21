"use client";

import { useState } from "react";
import { Button } from "../../ui/button";
import { Input } from "../../ui/input";
import { Textarea } from "../../ui/textarea";
import { Section } from "../../ui/section";
import { Badge } from "../../ui/badge";
import { Mail, Send, CheckCircle } from "lucide-react";

export default function SubscriptionForm() {
  const [formData, setFormData] = useState({
    email: "",
    name: "",
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
        setFormData({ email: "", name: "", message: "" });
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
      <Section id="contact" className="py-24">
        <div className="max-w-container mx-auto text-center">
          <div className="flex flex-col items-center gap-6">
            <CheckCircle className="size-16 text-green-500" />
            <h2 className="text-3xl font-bold">Thank You!</h2>
            <p className="text-muted-foreground max-w-md">
              Your message has been sent successfully. We'll get back to you soon!
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
    <Section id="contact" className="py-24">
      <div className="max-w-container mx-auto">
        <div className="text-center mb-12">
          <Badge variant="outline" className="mb-4">
            <Mail className="mr-2 size-4" />
            Get In Touch
          </Badge>
          <h2 className="text-3xl font-bold mb-4">
            Ready to Transform Your Business?
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Contact us to discuss your AI and web development needs. 
            We'll help you build intelligent solutions that drive growth.
          </p>
        </div>

        <div className="max-w-md mx-auto">
          <form onSubmit={handleSubmit} className="space-y-6">
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
            
            <div>
              <Input
                type="text"
                name="name"
                placeholder="Your name (optional)"
                value={formData.name}
                onChange={handleChange}
                className="w-full"
              />
            </div>
            
            <div>
              <Textarea
                name="message"
                placeholder="Tell us about your project..."
                value={formData.message}
                onChange={handleChange}
                rows={4}
                className="w-full"
              />
            </div>

            {error && (
              <div className="text-red-500 text-sm text-center">
                {error}
              </div>
            )}

            <Button
              type="submit"
              disabled={isSubmitting || !formData.email}
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
      </div>
    </Section>
  );
}