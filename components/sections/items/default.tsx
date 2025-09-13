import Link from "next/link";

import { Item, ItemDescription, ItemTitle } from "../../ui/item";
import { Section } from "../../ui/section";

interface ItemProps {
  title: string;
  description: string;
  href?: string;
}

interface ItemsProps {
  title?: string;
  items?: ItemProps[] | false;
  className?: string;
}

export default function Items({
  title = "AI & Creative Solutions",
  items = [
    {
      title: "AI Development Services",
      description: "Custom machine learning models, NLP, computer vision, and intelligent automation engineered for measurable outcomes.",
      href: "/services/ai-machine-learning",
    },
    {
      title: "Next.js Development Agency",
      description: "SEO-first, high-performance web apps with server-side rendering, edge optimizations, and TypeScript.",
      href: "/services/nextjs-development",
    },
    {
      title: "Graphic Design Studio",
      description: "Creative visual solutions with AI-enhanced design workflows for brand identity, print, and digital graphics.",
      href: "/services/graphic-design",
    },
    {
      title: "Video Editing & Production",
      description: "Professional video editing with AI-powered enhancement tools for cinematic quality and social media content.",
      href: "/services/video-editing",
    },
    {
      title: "Content Writing Services",
      description: "AI-assisted content creation for blogs, marketing copy, and SEO-optimized materials that drive engagement.",
      href: "/services/content-writing",
    },
    {
      title: "Deep Learning Solutions",
      description: "Advanced neural architectures for prediction, classification, and complex pattern recognition.",
      href: "/services/deep-learning",
    },
    {
      title: "Full‑Stack Web Development",
      description: "From frontend UX to backend APIs and databases—secure, scalable, production-ready delivery.",
      href: "/services/fullstack-web-development",
    },
    {
      title: "AI Chatbots & Automation",
      description: "Conversational AI and automated workflows integrated into your existing systems and channels.",
      href: "/services/ai-chatbots",
    },
    {
      title: "Enterprise Solutions",
      description: "Compliance, security, and reliability for mission‑critical platforms at scale.",
      href: "/services/enterprise-solutions",
    },
    {
      title: "Data Analytics & BI",
      description: "Dashboards, forecasting, and decision intelligence with clean pipelines and governance.",
      href: "/services/data-analytics",
    },
    {
      title: "AI Consulting",
      description: "Strategy, architecture, and roadmap to align AI investments with revenue and efficiency goals.",
      href: "/services/ai-consulting",
    },
  ],
  className,
}: ItemsProps) {
  return (
    <Section className={className}>
      <div className="max-w-container mx-auto flex flex-col items-center gap-6 sm:gap-12 fade-in-professional">
        <h2 className="max-w-[720px] text-center text-3xl leading-tight font-semibold sm:text-5xl sm:leading-tight">
          {title}
        </h2>
        {items !== false && items.length > 0 && (
          <div className="grid auto-rows-fr grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {items.map((item, index) => (
              <Item key={index} className={item.href ? "professional-card rounded-xl hover-lift-professional cursor-pointer" : "professional-card rounded-xl"}>
                {item.href ? (
                  <Link href={item.href} className="block h-full">
                    <ItemTitle className="text-primary hover:text-primary/80 transition-colors">
                      {item.title}
                    </ItemTitle>
                    <ItemDescription className="hover:text-foreground transition-colors">
                      {item.description}
                    </ItemDescription>
                  </Link>
                ) : (
                  <>
                    <ItemTitle>{item.title}</ItemTitle>
                    <ItemDescription>{item.description}</ItemDescription>
                  </>
                )}
              </Item>
            ))}
          </div>
        )}
      </div>
    </Section>
  );
}
