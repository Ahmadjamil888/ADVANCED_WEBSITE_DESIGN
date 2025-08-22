import * as React from "react";

import { cn } from "@/lib/utils";

import { Badge } from "../../ui/badge";
import { Section } from "../../ui/section";

interface Company {
  name: string;
  subtitle?: string;
}

const companies: Company[] = [
  { name: "IRTCoP" },
  { name: "Aurion Tech Global" },
  { name: "Usman Hospital" },
  { name: "Daak Khana", subtitle: "Owned by ZehanX Technologies" },
  { name: "Budget Plus Services" },
];

function CompanyPill({ name, subtitle }: Company) {
  return (
    <div className="professional-card hover-lift-professional rounded-xl px-6 py-3 min-w-[220px] flex items-center justify-center text-center select-none">
      <div className="flex flex-col items-center">
        <span className="font-semibold tracking-tight">{name}</span>
        {subtitle && (
          <span className="text-xs text-muted-foreground mt-0.5">{subtitle}</span>
        )}
      </div>
    </div>
  );
}

export default function CompanyMarquee({ className }: { className?: string }) {
  return (
    <Section className={cn("py-10 sm:py-14 relative", className)}>
      <div className="max-w-container mx-auto">
        <div className="flex flex-col items-center gap-4 text-center mb-6 fade-in-professional">
          <Badge variant="outline" className="badge-professional">
            Trusted by forward-thinking teams
          </Badge>
          <h2 className="text-2xl sm:text-3xl font-semibold">
            Partnering with leading companies
          </h2>
        </div>

        <div className="relative group fade-in-professional">
          {/* Edge shadow/gradient overlays */}
          <div className="pointer-events-none absolute inset-y-0 left-0 w-24 bg-gradient-to-r from-background via-background/80 to-transparent dark:from-black/40 dark:via-black/30"></div>
          <div className="pointer-events-none absolute inset-y-0 right-0 w-24 bg-gradient-to-l from-background via-background/80 to-transparent dark:from-black/40 dark:via-black/30"></div>

          <div className="overflow-hidden rounded-xl border border-border/50 bg-card/70">
            {/* Double row for seamless infinite scroll */}
            <div className="flex w-[200%] animate-[marquee_28s_linear_infinite] hover:[animation-play-state:paused]">
              <div className="flex items-center gap-6 sm:gap-8 w-1/2 px-2">
                {companies.map((c, i) => (
                  <CompanyPill key={`a-${i}`} name={c.name} subtitle={c.subtitle} />
                ))}
              </div>
              <div className="flex items-center gap-6 sm:gap-8 w-1/2 px-2">
                {companies.map((c, i) => (
                  <CompanyPill key={`b-${i}`} name={c.name} subtitle={c.subtitle} />
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </Section>
  );
}
