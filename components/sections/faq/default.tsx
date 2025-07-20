import Link from "next/link";
import { ReactNode } from "react";

import { siteConfig } from "@/config/site";

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "../../ui/accordion";
import { Section } from "../../ui/section";

interface FAQItemProps {
  question: string;
  answer: ReactNode;
  value?: string;
}

interface FAQProps {
  title?: string;
  items?: FAQItemProps[] | false;
  className?: string;
}

export default function FAQ({
  title = "Frequently Asked Questions",
  items = [
    {
      question: "What AI services does Zehan X Technologies offer?",
      answer: (
        <>
          <p className="text-muted-foreground mb-4 max-w-[640px] text-balance">
            We offer comprehensive AI solutions including custom machine learning models, 
            deep learning systems, computer vision, natural language processing, AI chatbots, 
            and predictive analytics.
          </p>
          <p className="text-muted-foreground mb-4 max-w-[640px] text-balance">
            Our team specializes in building intelligent applications that solve real business problems.
          </p>
        </>
      ),
    },
    {
      question: "How can AI benefit my business?",
      answer: (
        <>
          <p className="text-muted-foreground mb-4 max-w-[600px]">
            AI can automate repetitive tasks, provide data-driven insights, improve customer 
            experiences through personalization, and enhance decision-making processes.
          </p>
          <p className="text-muted-foreground mb-4 max-w-[600px]">
            It can also reduce operational costs and create new revenue opportunities. 
            We help identify the best AI applications for your specific industry and business needs.
          </p>
        </>
      ),
    },
    {
      question: "Do you specialize in Next.js development?",
      answer: (
        <>
          <p className="text-muted-foreground mb-4 max-w-[580px]">
            Yes! We specialize in Next.js development and modern React applications. 
            We build fast, scalable, and SEO-optimized web applications.
          </p>
          <p className="text-muted-foreground mb-4 max-w-[580px]">
            Our expertise includes App Router, Server Components, and advanced optimization 
            techniques to deliver exceptional performance.
          </p>
        </>
      ),
    },
    {
      question: "How long does it take to develop an AI solution?",
      answer: (
        <>
          <p className="text-muted-foreground mb-4 max-w-[580px]">
            Project timelines vary based on complexity and requirements. Simple AI integrations 
            might take 2-4 weeks, while custom machine learning models can take 2-6 months.
          </p>
          <p className="text-muted-foreground mb-4 max-w-[580px]">
            We provide detailed project timelines during our initial consultation and keep 
            you updated throughout the development process.
          </p>
        </>
      ),
    },
    {
      question: "Do you provide ongoing support and maintenance?",
      answer: (
        <p className="text-muted-foreground mb-4 max-w-[580px]">
          Absolutely! We offer comprehensive support packages including model monitoring, 
          performance optimization, updates, bug fixes, and feature enhancements. 
          Our team ensures your AI solutions continue to perform optimally.
        </p>
      ),
    },
    {
      question: "How do I get started with Zehan X Technologies?",
      answer: (
        <>
          <p className="text-muted-foreground mb-4 max-w-[580px]">
            Getting started is easy! Simply contact us through our contact form or email us directly. 
            We'll schedule a consultation to discuss your project requirements and provide a detailed proposal.
          </p>
          <p className="text-muted-foreground mb-4 max-w-[580px]">
            Contact us at{" "}
            <a
              href={siteConfig.links.email}
              className="underline underline-offset-2"
            >
              {siteConfig.links.email.replace('mailto:', '')}
            </a>
            {" "}to begin your AI transformation journey.
          </p>
        </>
      ),
    },
  ],
  className,
}: FAQProps) {
  return (
    <Section className={className}>
      <div className="max-w-container mx-auto flex flex-col items-center gap-8">
        <h2 className="text-center text-3xl font-semibold sm:text-5xl">
          {title}
        </h2>
        {items !== false && items.length > 0 && (
          <Accordion type="single" collapsible className="w-full max-w-[800px]">
            {items.map((item, index) => (
              <AccordionItem
                key={index}
                value={item.value || `item-${index + 1}`}
              >
                <AccordionTrigger>{item.question}</AccordionTrigger>
                <AccordionContent>{item.answer}</AccordionContent>
              </AccordionItem>
            ))}
          </Accordion>
        )}
      </div>
    </Section>
  );
}
