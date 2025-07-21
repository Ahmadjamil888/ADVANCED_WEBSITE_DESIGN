import { Menu } from "lucide-react";
import { ReactNode } from "react";


import { cn } from "@/lib/utils";

import { Button, type ButtonProps } from "../../ui/button";
import {
  Navbar as NavbarComponent,
  NavbarLeft,
  NavbarRight,
} from "../../ui/navbar";
import Navigation from "../../ui/navigation";
import { Sheet, SheetContent, SheetTrigger } from "../../ui/sheet";

interface NavbarLink {
  text: string;
  href: string;
}

interface NavbarActionProps {
  text: string;
  href: string;
  variant?: ButtonProps["variant"];
  icon?: ReactNode;
  iconRight?: ReactNode;
  isButton?: boolean;
}

interface NavbarProps {
  logo?: ReactNode;
  name?: string;
  homeUrl?: string;
  mobileLinks?: NavbarLink[];
  actions?: NavbarActionProps[];
  showNavigation?: boolean;
  customNavigation?: ReactNode;
  className?: string;
}

export default function Navbar({
  logo = null,
  name = "ZEHANX",
  homeUrl = "/",
  mobileLinks = [
    { text: "Home", href: "/" },
    { text: "About", href: "/about" },
    { text: "Services", href: "/services" },
    { text: "Contact", href: "/contact" },
  ],
  showNavigation = true,
  customNavigation,
  className,
}: NavbarProps) {
  return (
    <header className={cn("sticky top-0 z-50 -mb-4 px-4 pb-4", className)}>
      <div className="fade-bottom bg-background/15 absolute left-0 h-24 w-full backdrop-blur-lg"></div>
      <div className="max-w-container relative mx-auto">
        <NavbarComponent>
          <NavbarLeft>
            <a
              href={homeUrl}
              className="flex items-center gap-2 text-xl font-bold tracking-wider text-white"
              style={{ 
                fontFamily: 'monospace, "Courier New", Courier',
                textShadow: '0 0 10px rgba(255,255,255,0.5)',
                letterSpacing: '0.2em'
              }}
            >
              {logo}
              {name}
            </a>
            {showNavigation && (customNavigation || <Navigation />)}
          </NavbarLeft>
          <NavbarRight>
            <Button variant="ghost" size="sm" asChild>
              <a href="/auth/signin">Sign In</a>
            </Button>
            <Button variant="default" size="sm" asChild>
              <a href="/contact">Get started</a>
            </Button>
            <Sheet>
              <SheetTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="shrink-0 md:hidden"
                >
                  <Menu className="size-5" />
                  <span className="sr-only">Toggle navigation menu</span>
                </Button>
              </SheetTrigger>
              <SheetContent side="right">
                <nav className="grid gap-6 text-lg font-medium">
                  <a
                    href={homeUrl}
                    className="flex items-center gap-2 text-xl font-bold"
                  >
                    <span>{name}</span>
                  </a>
                  {mobileLinks.map((link, index) => (
                    <a
                      key={index}
                      href={link.href}
                      className="text-muted-foreground hover:text-foreground"
                    >
                      {link.text}
                    </a>
                  ))}
                </nav>
              </SheetContent>
            </Sheet>
          </NavbarRight>
        </NavbarComponent>
      </div>
    </header>
  );
}
