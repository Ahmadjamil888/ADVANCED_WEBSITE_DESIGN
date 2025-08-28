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
      href: "/zehan",
      text: "Try Zehan AI",
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

      <div className="max-w-container mx-auto flex flex-col gap-12 pt-20 sm:gap-24 relative z-10">
        <div className="flex flex-col items-center gap-8 text-center sm:gap-12">
          <Badge variant="outline" className="badge-professional fade-in-professional">
            <span className="text-primary font-semibold">Enterprise AI & Web Development</span>
          </Badge>

          <h1 className="fade-in-professional heading-professional relative z-10 inline-block text-4xl leading-tight text-balance sm:text-6xl sm:leading-tight md:text-8xl md:leading-tight">
            <span className="text-foreground">Transform Your Business with</span>{" "}
            <span className="text-gradient-primary">Professional AI & Web Development</span>
          </h1>

          <p className="fade-in-professional subheading-professional relative z-10 max-w-[800px] text-lg font-medium text-balance opacity-0 delay-100 sm:text-xl leading-relaxed">
            Zehan X Technologies delivers enterprise-grade AI solutions, sophisticated Next.js development,
            and advanced machine learning systems. We architect intelligent applications that drive measurable
            business outcomes and accelerate your digital transformation journey.
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