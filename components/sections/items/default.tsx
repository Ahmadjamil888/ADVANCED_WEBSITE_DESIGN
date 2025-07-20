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
import { ReactNode } from "react";

import { Item, ItemDescription,ItemIcon, ItemTitle } from "../../ui/item";
import { Section } from "../../ui/section";

interface ItemProps {
  title: string;
  description: string;
  icon: ReactNode;
}

interface ItemsProps {
  title?: string;
  items?: ItemProps[] | false;
  className?: string;
}

export default function Items({
  title = "Comprehensive AI & Web Development Solutions",
  items = [
    {
      title: "AI & Machine Learning",
      description: "Custom ML models, predictive analytics, and intelligent automation",
      icon: <BrainCircuitIcon className="size-5 stroke-1" />,
    },
    {
      title: "Next.js Development",
      description: "Modern, fast, and scalable web applications with React",
      icon: <CodeIcon className="size-5 stroke-1" />,
    },
    {
      title: "Deep Learning",
      description: "Advanced neural networks for complex pattern recognition",
      icon: <DatabaseIcon className="size-5 stroke-1" />,
    },
    {
      title: "Web Applications",
      description: "Full-stack solutions optimized for performance and UX",
      icon: <GlobeIcon className="size-5 stroke-1" />,
    },
    {
      title: "AI Chatbots",
      description: "Intelligent conversational AI for customer engagement",
      icon: <BotIcon className="size-5 stroke-1" />,
    },
    {
      title: "Performance Optimization",
      description: "Lightning-fast applications with cutting-edge techniques",
      icon: <ZapIcon className="size-5 stroke-1" />,
    },
    {
      title: "Data Analytics",
      description: "Transform your data into actionable business insights",
      icon: <BarChart3Icon className="size-5 stroke-1" />,
    },
    {
      title: "Enterprise Security",
      description: "Robust security measures for mission-critical applications",
      icon: <ShieldCheckIcon className="size-5 stroke-1" />,
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
              <Item key={index}>
                <ItemTitle className="flex items-center gap-2">
                  <ItemIcon>{item.icon}</ItemIcon>
                  {item.title}
                </ItemTitle>
                <ItemDescription>{item.description}</ItemDescription>
              </Item>
            ))}
          </div>
        )}
      </div>
    </Section>
  );
}
