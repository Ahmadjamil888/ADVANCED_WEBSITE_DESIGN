import { ArrowRightIcon, BrainCircuit, Code, Zap } from "lucide-react";
import { ReactNode } from "react";


import { cn } from "@/lib/utils";

import { Badge } from "../../ui/badge";
import { Button, type ButtonProps } from "../../ui/button";
import Glow from "../../ui/glow";
import { Section } from "../../ui/section";

interface HeroButtonProps {
  href: string;
  text: string;
  variant?: ButtonProps["variant"];
  icon?: ReactNode;
  iconRight?: ReactNode;
}

interface HeroProps {
  className?: string;
}

export default function ZehanHero({ className }: HeroProps) {
  const buttons: HeroButtonProps[] = [
    {
      href: "#contact",
      text: "Get Started",
      variant: "default",
      iconRight: <ArrowRightIcon className="ml-2 size-4" />,
    },
    {
      href: "#services",
      text: "Our Services",
      variant: "outline",
    },
  ];

  return (
    <Section
      className={cn(
        "fade-bottom overflow-hidden pb-0 sm:pb-0 md:pb-0",
        className,
      )}
    >
      <div className="max-w-container mx-auto flex flex-col gap-12 pt-16 sm:gap-24">
        <div className="flex flex-col items-center gap-6 text-center sm:gap-12">
          <Badge variant="outline" className="animate-appear">
            <BrainCircuit className="mr-2 size-4" />
            <span className="text-muted-foreground">
              AI-Powered Solutions
            </span>
          </Badge>
          
          <h1 className="animate-appear from-foreground to-foreground dark:to-muted-foreground relative z-10 inline-block bg-linear-to-r bg-clip-text text-4xl leading-tight font-semibold text-balance text-transparent drop-shadow-2xl sm:text-6xl sm:leading-tight md:text-8xl md:leading-tight">
            Transform Your Business with{" "}
            <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              AI & Web Development
            </span>
          </h1>
          
          <p className="text-md animate-appear text-muted-foreground relative z-10 max-w-[740px] font-medium text-balance opacity-0 delay-100 sm:text-xl">
            Zehan X Technologies specializes in cutting-edge AI solutions, Next.js development, 
            deep learning, and machine learning. We build intelligent applications that drive innovation 
            and accelerate your digital transformation.
          </p>
          
          <div className="animate-appear relative z-10 flex justify-center gap-4 opacity-0 delay-300">
            {buttons.map((button, index) => (
              <Button
                key={index}
                variant={button.variant || "default"}
                size="lg"
                asChild
              >
                <a href={button.href}>
                  {button.icon}
                  {button.text}
                  {button.iconRight}
                </a>
              </Button>
            ))}
          </div>

          {/* AI Tech Stack Icons */}
          <div className="animate-appear relative z-10 flex flex-wrap justify-center gap-8 pt-8 opacity-0 delay-500">
            <div className="flex items-center gap-2 text-muted-foreground">
              <BrainCircuit className="size-6" />
              <span className="text-sm font-medium">AI/ML</span>
            </div>
            <div className="flex items-center gap-2 text-muted-foreground">
              <Code className="size-6" />
              <span className="text-sm font-medium">Next.js</span>
            </div>
            <div className="flex items-center gap-2 text-muted-foreground">
              <Zap className="size-6" />
              <span className="text-sm font-medium">Deep Learning</span>
            </div>
          </div>

          <div className="relative w-full pt-12">
            <Glow
              variant="top"
              className="animate-appear-zoom opacity-0 delay-1000"
            />
          </div>
        </div>
      </div>
    </Section>
  );
}