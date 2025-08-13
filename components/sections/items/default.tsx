import {
  BrainCircuitIcon,
  CodeIcon,
  DatabaseIcon,
  GlobeIcon,
  BotIcon,
  ZapIcon,
  BarChart3Icon,
  ShieldCheckIcon,
} from "lucide-react";
import Link from "next/link";
import { ReactNode } from "react";

import { Item, ItemDescription,ItemIcon, ItemTitle } from "../../ui/item";
import { Section } from "../../ui/section";

interface ItemProps {
  title: string;
  description: string;
  icon: ReactNode;
  href?: string;
}

interface ItemsProps {
  title?: string;
  items?: ItemProps[] | false;
  className?: string;
}

export default function Items({
  title = "Experience Our AI Technology - Try Zehan AI Now!",
  items = [
    {
      title: "Try Zehan AI",
      description: "Experience our AI technology firsthand with our interactive chat demo",
      icon: <BrainCircuitIcon className="size-5 stroke-1" />,
      href: "/zehan",
    },
    {
      title: "Next.js Development",
      description: "Modern, fast, and scalable web applications with React - Try Zehan AI!",
      icon: <CodeIcon className="size-5 stroke-1" />,
      href: "/zehan",
    },
    {
      title: "Deep Learning",
      description: "Advanced neural networks for complex pattern recognition - Demo available!",
      icon: <DatabaseIcon className="size-5 stroke-1" />,
      href: "/zehan",
    },
    {
      title: "Web Applications",
      description: "Full-stack solutions optimized for performance - See Zehan AI in action!",
      icon: <GlobeIcon className="size-5 stroke-1" />,
      href: "/zehan",
    },
    {
      title: "AI Chatbots",
      description: "Intelligent conversational AI - Try our Zehan AI chatbot now!",
      icon: <BotIcon className="size-5 stroke-1" />,
      href: "/zehan",
    },
    {
      title: "Performance Optimization",
      description: "Lightning-fast applications - Experience the speed with Zehan AI!",
      icon: <ZapIcon className="size-5 stroke-1" />,
      href: "/zehan",
    },
    {
      title: "Data Analytics",
      description: "Transform data into insights - See AI analytics with Zehan AI!",
      icon: <BarChart3Icon className="size-5 stroke-1" />,
      href: "/zehan",
    },
    {
      title: "Enterprise Security",
      description: "Robust security measures - Try our secure Zehan AI platform!",
      icon: <ShieldCheckIcon className="size-5 stroke-1" />,
      href: "/zehan",
    },
  ],
  className,
}: ItemsProps) {
  return (
    <Section className={className}>
      <div className="max-w-container mx-auto flex flex-col items-center gap-6 sm:gap-20">
        <h2 className="max-w-[560px] text-center text-3xl leading-tight font-semibold sm:text-5xl sm:leading-tight">
          {title}
        </h2>
        {items !== false && items.length > 0 && (
          <div className="grid auto-rows-fr grid-cols-2 gap-0 sm:grid-cols-3 sm:gap-4 lg:grid-cols-4">
            {items.map((item, index) => (
              <Item key={index} className={item.href ? "hover-lift-professional cursor-pointer" : ""}>
                {item.href ? (
                  <Link href={item.href} className="block h-full">
                    <ItemTitle className="flex items-center gap-2 text-primary hover:text-primary/80 transition-colors">
                      <ItemIcon>{item.icon}</ItemIcon>
                      {item.title}
                    </ItemTitle>
                    <ItemDescription className="hover:text-foreground transition-colors">
                      {item.description}
                    </ItemDescription>
                  </Link>
                ) : (
                  <>
                    <ItemTitle className="flex items-center gap-2">
                      <ItemIcon>{item.icon}</ItemIcon>
                      {item.title}
                    </ItemTitle>
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
