import { ReactNode } from "react";

import { cn } from "@/lib/utils";

import { Badge } from "../../ui/badge";
import { Button, type ButtonProps } from "../../ui/button";
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
      href: "/services",
      text: "Learn More",
      variant: "default",
    },
    {
      href: "/contact",
      text: "Get Started",
      variant: "outline",
    },
  ];

  return (
    <Section
      className={cn(
        "fade-bottom overflow-hidden pb-0 sm:pb-0 md:pb-0 gradient-bg-professional relative min-h-screen",
        className,
      )}
    >
      {/* Creative Background Elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-20 left-10 w-32 h-32 bg-blue-500/10 rounded-full blur-xl animate-pulse glow-orb-1"></div>
        <div className="absolute top-40 right-20 w-24 h-24 bg-cyan-500/10 rounded-full blur-lg animate-pulse glow-orb-2"></div>
        <div className="absolute bottom-40 left-1/4 w-40 h-40 bg-purple-500/10 rounded-full blur-2xl animate-pulse glow-orb-3"></div>
        <div className="absolute bottom-20 right-1/3 w-28 h-28 bg-green-500/10 rounded-full blur-xl animate-pulse glow-orb-4"></div>
        
        {/* Animated gradient lines */}
        <div className="absolute top-1/4 left-0 w-full h-px bg-gradient-to-r from-transparent via-blue-400/30 to-transparent animate-gradient-x"></div>
        <div className="absolute top-3/4 left-0 w-full h-px bg-gradient-to-r from-transparent via-cyan-400/30 to-transparent animate-gradient-x" style={{animationDelay: '2s'}}></div>
      </div>

      <div className="max-w-container mx-auto flex flex-col gap-12 pt-20 sm:gap-24 relative z-10">
        <div className="flex flex-col items-center gap-8 text-center sm:gap-12">
          <Badge variant="outline" className="badge-professional fade-in-professional">
            <span className="text-primary font-semibold">Creative Web Development Agency</span>
          </Badge>

          <h1 className="fade-in-professional heading-professional relative z-10 inline-block text-4xl leading-tight text-balance sm:text-6xl sm:leading-tight md:text-8xl md:leading-tight animate-fade-in-up">
            <span className="text-foreground">Where Creativity Meets</span>{" "}
            <span className="text-gradient-primary animate-pulse">Technology</span>
          </h1>

          <p className="fade-in-professional subheading-professional relative z-10 max-w-[800px] text-lg font-medium text-balance opacity-0 delay-100 sm:text-xl leading-relaxed animate-fade-in-up" style={{animationDelay: '0.3s'}}>
            We're a full-service creative agency specializing in cutting-edge web development, 
            digital marketing, video editing, graphic design, and content creation. We transform your vision into 
            stunning digital experiences that drive results and captivate your audience.
          </p>

          <div className="fade-in-professional relative z-10 flex flex-col sm:flex-row justify-center gap-4 opacity-0 delay-300">
            {buttons.map((button, index) => (
              <Button
                key={index}
                variant={button.variant || "default"}
                size="lg"
                className={cn(
                  "font-semibold px-8 py-3",
                  button.variant === "default"
                    ? "btn-gradient-primary text-white"
                    : "professional-card border-primary/30 text-primary"
                )}
                asChild
              >
                <a href={button.href}>
                  {button.text}
                </a>
              </Button>
            ))}
          </div>


          {/* Professional Video Section */}
          <div className="relative w-full pt-16">
            <MockupFrame
              className="fade-in-professional opacity-0 delay-700"
              size="small"
            >
              <Mockup
                type="responsive"
                className="professional-card w-full rounded-2xl border-border/40"
              >
                <YouTubeVideo
                  videoId="fa8k8IQ1_X0"
                  className="w-full rounded-xl"
                  autoplay={true}
                  controls={false}
                  mute={true}
                  loop={true}
                  title="Zehan X Technologies - Professional AI & Web Development Solutions"
                />
              </Mockup>
            </MockupFrame>
          </div>
        </div>
      </div>
    </Section>
  );
}