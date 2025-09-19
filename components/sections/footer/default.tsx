import { ReactNode } from "react";

import { siteConfig } from "@/config/site";
import { cn } from "@/lib/utils";
import SimpleLogo from "../../logos/simple-logo";


import {
  Footer,
  FooterBottom,
  FooterColumn,
  FooterContent,
} from "../../ui/footer";
import { ModeToggle } from "../../ui/mode-toggle";

interface FooterLink {
  text: string;
  href: string;
}

interface FooterColumnProps {
  title: string;
  links: FooterLink[];
}

interface FooterProps {
  logo?: ReactNode;
  name?: string;
  columns?: FooterColumnProps[];
  copyright?: string;
  policies?: FooterLink[];
  showModeToggle?: boolean;
  className?: string;
}

export default function FooterSection({
  logo = <SimpleLogo size="md" />,
  name = "",
  columns = [
    {
      title: "Services",
      links: [
        { text: "AI & Machine Learning", href: "/services/ai-machine-learning" },
        { text: "Web Development", href: "/services/fullstack-web-development" },
        { text: "Digital Marketing", href: "/services/digital-marketing" },
        { text: "Video Editing", href: "/services/video-editing" },
        { text: "Graphic Design", href: "/services/graphic-design" },
        { text: "Content Writing", href: "/services/content-writing" },
        { text: "AI Chatbots", href: "/services/ai-chatbots" },
        { text: "AI Consulting", href: "/services/ai-consulting" },
      ],
    },
    {
      title: "Company",
      links: [
        { text: "About Us", href: "/about" },
        { text: "All Services", href: "/services" },
        { text: "Contact", href: "/contact" },
        { text: "Data Analytics", href: "/services/data-analytics" },
        { text: "Enterprise Solutions", href: "/services/enterprise-solutions" },
      ],
    },
    {
      title: "Connect",
      links: [
        { text: "Contact Us", href: "/contact" },
        { text: "Twitter", href: siteConfig.links.twitter },
        { text: "GitHub", href: siteConfig.links.github },
        { text: "LinkedIn", href: "https://linkedin.com/company/zehanx" },
      ],
    },
  ],
  copyright = "© 2025 Zehan X Technologies. All rights reserved.",
  policies = [
    { text: "Privacy Policy", href: "/privacy" },
    { text: "Terms of Service", href: "/terms" },
  ],
  showModeToggle = true,
  className,
}: FooterProps) {
  return (
    <footer className={cn("bg-background w-full px-4", className)}>
      <div className="max-w-container mx-auto">
        <Footer>
          <FooterContent>
            <FooterColumn className="col-span-2 sm:col-span-3 md:col-span-1">
              <div className="flex items-center gap-3">
                {logo}
                <h3 className="text-xl font-bold heading-professional text-primary">{name}</h3>
              </div>
            </FooterColumn>
            {columns.map((column, index) => (
              <FooterColumn key={index}>
                <h3 className="text-md pt-1 font-semibold text-primary">{column.title}</h3>
                {column.links.map((link, linkIndex) => (
                  <a
                    key={linkIndex}
                    href={link.href}
                    className="nav-item-professional text-sm transition-all duration-300"
                  >
                    {link.text}
                  </a>
                ))}
              </FooterColumn>
            ))}
          </FooterContent>
          <FooterBottom>
            <div>{copyright}</div>
            <div className="flex items-center gap-4">
              {policies.map((policy, index) => (
                <a key={index} href={policy.href}>
                  {policy.text}
                </a>
              ))}
              {showModeToggle && <ModeToggle />}
            </div>
          </FooterBottom>
        </Footer>
      </div>
    </footer>
  );
}
