"use client";

import Link from "next/link";
import * as React from "react";
import { ReactNode } from "react";

import { cn } from "@/lib/utils";
import ZehanLogo from "../logos/zehan-logo";
import {
  NavigationMenu,
  NavigationMenuContent,
  NavigationMenuItem,
  NavigationMenuLink,
  NavigationMenuList,
  NavigationMenuTrigger,
  navigationMenuTriggerStyle,
} from "./navigation-menu";

interface ComponentItem {
  title: string;
  href: string;
  description: string;
}

interface MenuItem {
  title: string;
  href?: string;
  isLink?: boolean;
  content?: ReactNode;
}

interface NavigationProps {
  menuItems?: MenuItem[];
  components?: ComponentItem[];
  logo?: ReactNode;
  logoTitle?: string;
  logoDescription?: string;
  logoHref?: string;
  introItems?: {
    title: string;
    href: string;
    description: string;
  }[];
}

export default function Navigation({
  menuItems = [
    {
      title: "Services",
      content: "services",
    },
    {
      title: "Blog",
      isLink: true,
      href: "/blog",
    },
    {
      title: "Portfolio",
      isLink: true,
      href: "/portfolio",
    },
    {
      title: "Our Journey",
      isLink: true,
      href: "/journey",
    },
    {
      title: "About",
      isLink: true,
      href: "/about",
    },
    {
      title: "Contact",
      isLink: true,
      href: "/contact",
    },
  ],
  components = [
    {
      title: "AI & Machine Learning",
      href: "/services/ai-machine-learning",
      description:
        "Custom ML models, predictive analytics, and intelligent automation solutions.",
    },
    {
      title: "Next.js Development",
      href: "/services/nextjs-development",
      description:
        "Modern, fast, and scalable web applications built with React and Next.js.",
    },
    {
      title: "Full-Stack Development",
      href: "/services/fullstack-web-development",
      description:
        "Complete web solutions from frontend to backend with modern technologies.",
    },
    {
      title: "Deep Learning",
      href: "/services/deep-learning",
      description:
        "Advanced neural networks for complex pattern recognition and decision making.",
    },
    {
      title: "AI Chatbots",
      href: "/services/ai-chatbots",
      description: "Intelligent conversational AI systems for customer engagement.",
    },
    {
      title: "AI Consulting",
      href: "/services/ai-consulting",
      description:
        "Strategic AI consulting to help identify opportunities and implement solutions.",
    },
  ],
  logo = <ZehanLogo size="md" />,
  logoTitle = "Zehan X Technologies",
  logoDescription = "Leading AI and web development company specializing in Next.js, deep learning, and machine learning solutions.",
  logoHref = "/",
  introItems = [
    {
      title: "Our Mission",
      href: "/about",
      description:
        "Transforming businesses with cutting-edge AI and web technologies.",
    },
    {
      title: "Get Started",
      href: "#contact",
      description: "Contact us to discuss your AI and web development needs.",
    },
    {
      title: "Learn More",
      href: "/about",
      description: "Discover how we can help accelerate your digital transformation.",
    },
  ],
}: NavigationProps) {
  return (
    <NavigationMenu className="hidden md:flex">
      <NavigationMenuList>
        {menuItems.map((item, index) => (
          <NavigationMenuItem key={index}>
            {item.isLink ? (
              <Link href={item.href || ""} legacyBehavior passHref>
                <NavigationMenuLink className={navigationMenuTriggerStyle()}>
                  {item.title}
                </NavigationMenuLink>
              </Link>
            ) : (
              <>
                <NavigationMenuTrigger>{item.title}</NavigationMenuTrigger>
                <NavigationMenuContent>
                  {item.content === "default" ? (
                    <ul className="grid gap-3 p-4 md:w-[400px] lg:w-[500px] lg:grid-cols-[.75fr_1fr]">
                      <li className="row-span-3">
                        <NavigationMenuLink asChild>
                          <a
                            className="from-muted/30 to-muted/10 flex h-full w-full flex-col justify-end rounded-md bg-linear-to-b p-6 no-underline outline-hidden select-none focus:shadow-md"
                            href={logoHref}
                          >
                            {logo}
                            <div className="mt-4 mb-2 text-lg font-medium">
                              {logoTitle}
                            </div>
                            <p className="text-muted-foreground text-sm leading-tight">
                              {logoDescription}
                            </p>
                          </a>
                        </NavigationMenuLink>
                      </li>
                      {introItems.map((intro, i) => (
                        <ListItem key={i} href={intro.href} title={intro.title}>
                          {intro.description}
                        </ListItem>
                      ))}
                    </ul>
                  ) : item.content === "services" ? (
                    <ul className="grid w-[400px] gap-3 p-4 md:w-[500px] md:grid-cols-2 lg:w-[600px]">
                      {components.map((component) => (
                        <ListItem
                          key={component.title}
                          title={component.title}
                          href={component.href}
                        >
                          {component.description}
                        </ListItem>
                      ))}
                    </ul>
                  ) : (
                    item.content
                  )}
                </NavigationMenuContent>
              </>
            )}
          </NavigationMenuItem>
        ))}
      </NavigationMenuList>
    </NavigationMenu>
  );
}

function ListItem({
  className,
  title,
  children,
  ...props
}: React.ComponentProps<"a"> & { title: string }) {
  return (
    <li>
      <NavigationMenuLink asChild>
        <a
          data-slot="list-item"
          className={cn(
            "hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground block space-y-1 rounded-md p-3 leading-none no-underline outline-hidden transition-colors select-none",
            className,
          )}
          {...props}
        >
          <div className="text-sm leading-none font-medium">{title}</div>
          <p className="text-muted-foreground line-clamp-2 text-sm leading-snug">
            {children}
          </p>
        </a>
      </NavigationMenuLink>
    </li>
  );
}
