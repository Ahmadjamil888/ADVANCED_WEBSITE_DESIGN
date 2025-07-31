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
      <Section id="contact" className="py-24 bg-background">
        <div className="max-w-container mx-auto text-center">
          <div className="flex flex-col items-center gap-6">
            <CheckCircle className="size-16 text-primary glow-primary" />
            <h2 className="text-3xl font-bold heading-professional text-gradient-primary">Thank You!</h2>
            <p className="subheading-professional max-w-md">
              Your message has been sent successfully. We'll get back to you soon!
            </p>
            <Button 
              onClick={() => setIsSuccess(false)} 
              variant="outline"
              className="professional-card hover-glow-professional border-primary/30 text-primary hover:border-primary/50 hover-lift-professional"
            >
              Send Another Message
            </Button>
          </div>
        </div>
      </Section>
    );
  }

  return (
    <Section id="contact" className="py-24 bg-background">
      <div className="max-w-container mx-auto">
        <div className="text-center mb-16 section-divider-professional">
          <Badge variant="outline" className="mb-6 badge-professional hover-glow-professional">
            <Mail className="mr-2 size-4" />
            Professional Consultation
          </Badge>
          <h2 className="text-4xl font-bold mb-6 heading-professional text-gradient-primary">
            Ready to Transform Your Enterprise?
          </h2>
          <p className="subheading-professional max-w-3xl mx-auto text-lg leading-relaxed">
            Schedule a consultation to discuss your AI and web development requirements. 
            We'll architect intelligent solutions that deliver measurable business value and competitive advantage.
          </p>
        </div>

        <div className="max-w-2xl mx-auto">
          <form onSubmit={handleSubmit} className="space-y-8 professional-card p-8">
            <div className="form-field-professional">
              <Input
                type="email"
                name="email"
                placeholder="Your professional email address"
                value={formData.email}
                onChange={handleChange}
                required
                className="w-full h-12 text-base border-border/60 focus:border-primary/60 rounded-lg"
              />
            </div>
            
            <div className="form-field-professional">
              <Input
                type="text"
                name="name"
                placeholder="Your full name"
                value={formData.name}
                onChange={handleChange}
                className="w-full h-12 text-base border-border/60 focus:border-primary/60 rounded-lg"
              />
            </div>
            
            <div className="form-field-professional">
              <Textarea
                name="message"
                placeholder="Describe your project requirements, goals, and timeline..."
                value={formData.message}
                onChange={handleChange}
                rows={5}
                className="w-full text-base border-border/60 focus:border-primary/60 rounded-lg resize-none"
              />
            </div>

            {error && (
              <div className="text-destructive text-sm text-center p-3 bg-destructive/10 rounded-lg border border-destructive/20">
                {error}
              </div>
            )}

            <Button
              type="submit"
              disabled={isSubmitting || !formData.email}
              className="w-full h-12 btn-gradient-primary text-white font-semibold text-base hover-lift-professional"
              size="lg"
            >
              {isSubmitting ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                  Processing Request...
                </>
              ) : (
                <>
                  Schedule Consultation
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