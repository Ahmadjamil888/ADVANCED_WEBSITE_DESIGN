import { ArrowRightIcon, BrainCircuit, Code, Zap } from "lucide-react";
import { ReactNode } from "react";

import { cn } from "@/lib/utils";

import { Badge } from "../../ui/badge";
import { Button, type ButtonProps } from "../../ui/button";
import Glow from "../../ui/glow";
import { Mockup, MockupFrame } from "../../ui/mockup";
import YouTubeVideo from "../../ui/youtube-video";
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
          <Badge variant="outline" className="animate-appear border-blue-500/20 bg-blue-500/10 hover-glow">
            <BrainCircuit className="mr-2 size-4 text-blue-400" />
            <span className="text-blue-400 font-medium">
              AI-Powered Solutions
            </span>
          </Badge>
          
          <h1 className="animate-appear from-foreground to-foreground dark:to-muted-foreground relative z-10 inline-block bg-linear-to-r bg-clip-text text-4xl leading-tight font-semibold text-balance text-transparent drop-shadow-2xl sm:text-6xl sm:leading-tight md:text-8xl md:leading-tight">
            Transform Your Business with AI & Web Development
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
                className={button.variant === "default" ? "btn-gradient-primary hover-lift glow-blue" : "hover-lift border-border/50 hover:border-blue-500/50"}
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
            <div className="flex items-center gap-2 text-muted-foreground hover:text-blue-400 transition-colors duration-300 cursor-default">
              <BrainCircuit className="size-6" />
              <span className="text-sm font-medium">AI/ML</span>
            </div>
            <div className="flex items-center gap-2 text-muted-foreground hover:text-green-400 transition-colors duration-300 cursor-default">
              <Code className="size-6" />
              <span className="text-sm font-medium">Next.js</span>
            </div>
            <div className="flex items-center gap-2 text-muted-foreground hover:text-purple-400 transition-colors duration-300 cursor-default">
              <Zap className="size-6" />
              <span className="text-sm font-medium">Deep Learning</span>
            </div>
          </div>

          {/* Video Section */}
          <div className="relative w-full pt-12">
            <MockupFrame
              className="animate-appear opacity-0 delay-700"
              size="small"
            >
              <Mockup
                type="responsive"
                className="bg-background/90 w-full rounded-xl border-0"
              >
                <YouTubeVideo
                  videoId="fa8k8IQ1_X0"
                  className="w-full"
                  autoplay={true}
                  controls={false}
                  mute={true}
                  loop={true}
                  title="Zehan X Technologies - AI & Web Development Solutions"
                />
              </Mockup>
            </MockupFrame>
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