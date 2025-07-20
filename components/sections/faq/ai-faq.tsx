import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "../../ui/accordion";
import { Badge } from "../../ui/badge";
import { Section } from "../../ui/section";
import { HelpCircle } from "lucide-react";

const faqs = [
  {
    question: "What AI services does Zehan X Technologies offer?",
    answer: "We offer comprehensive AI solutions including custom machine learning models, deep learning systems, computer vision, natural language processing, AI chatbots, and predictive analytics. Our team specializes in building intelligent applications that solve real business problems."
  },
  {
    question: "How can AI benefit my business?",
    answer: "AI can automate repetitive tasks, provide data-driven insights, improve customer experiences through personalization, enhance decision-making processes, reduce operational costs, and create new revenue opportunities. We help identify the best AI applications for your specific industry and business needs."
  },
  {
    question: "Do you work with Next.js for web development?",
    answer: "Yes! We specialize in Next.js development and modern React applications. We build fast, scalable, and SEO-optimized web applications using the latest Next.js features including App Router, Server Components, and advanced optimization techniques."
  },
  {
    question: "What's the difference between machine learning and deep learning?",
    answer: "Machine learning is a broader field that includes various algorithms for pattern recognition and prediction. Deep learning is a subset of ML that uses neural networks with multiple layers to process complex data like images, text, and speech. We work with both approaches depending on your project requirements."
  },
  {
    question: "How long does it take to develop an AI solution?",
    answer: "Project timelines vary based on complexity and requirements. Simple AI integrations might take 2-4 weeks, while custom machine learning models can take 2-6 months. We provide detailed project timelines during our initial consultation and keep you updated throughout the development process."
  },
  {
    question: "Do you provide ongoing support and maintenance?",
    answer: "Absolutely! We offer comprehensive support packages including model monitoring, performance optimization, updates, bug fixes, and feature enhancements. Our team ensures your AI solutions continue to perform optimally as your business grows."
  },
  {
    question: "Can you integrate AI into existing systems?",
    answer: "Yes, we specialize in seamless AI integration with existing business systems. Whether you're using CRM, ERP, e-commerce platforms, or custom applications, we can integrate AI capabilities without disrupting your current workflows."
  },
  {
    question: "What industries do you serve?",
    answer: "We work across various industries including healthcare, finance, e-commerce, manufacturing, education, and technology startups. Our AI solutions are customized to meet the specific challenges and requirements of each industry."
  }
];

export default function AIFAQ() {
  return (
    <Section className="py-24">
      <div className="max-w-container mx-auto">
        <div className="text-center mb-16">
          <Badge variant="outline" className="mb-4">
            <HelpCircle className="mr-2 size-4" />
            FAQ
          </Badge>
          <h2 className="text-3xl font-bold mb-4">
            Frequently Asked Questions
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Get answers to common questions about our AI and web development services.
          </p>
        </div>

        <div className="max-w-3xl mx-auto">
          <Accordion type="single" collapsible className="w-full">
            {faqs.map((faq, index) => (
              <AccordionItem key={index} value={`item-${index}`}>
                <AccordionTrigger className="text-left">
                  {faq.question}
                </AccordionTrigger>
                <AccordionContent className="text-muted-foreground">
                  {faq.answer}
                </AccordionContent>
              </AccordionItem>
            ))}
          </Accordion>
        </div>
      </div>
    </Section>
  );
}