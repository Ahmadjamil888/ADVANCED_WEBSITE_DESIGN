import { Menu, Youtube, Linkedin, Instagram } from "lucide-react";
import { ReactNode } from "react";
import {
  SignInButton,
  SignUpButton,
  SignedIn,
  SignedOut,
  UserButton,
} from '@clerk/nextjs';
import { siteConfig } from "@/config/site";

import { cn } from "@/lib/utils";
import SimpleLogo from "../../logos/simple-logo";

import { Button } from "../../ui/button";
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



interface NavbarProps {
  name?: string;
  homeUrl?: string;
  mobileLinks?: NavbarLink[];
  showNavigation?: boolean;
  customNavigation?: ReactNode;
  className?: string;
}

export default function Navbar({
  name = "ZEHANX",
  homeUrl = "/",
  mobileLinks = [
    { text: "Home", href: "/" },
    { text: "About", href: "/about" },
    { text: "Services", href: "/services" },
    { text: "Blog", href: "/blog" },
    { text: "Portfolio", href: "/portfolio" },
    { text: "Our Journey", href: "/journey" },
    { text: "Contact", href: "/contact" },
  ],
  showNavigation = true,
  customNavigation,
  className,
}: NavbarProps) {
  return (
    <header className={cn("sticky top-0 z-50 -mb-4 px-4 pb-3", className)}>
      <div className="absolute inset-0 bg-background/70 backdrop-blur supports-[backdrop-filter]:bg-background/60 border-b border-border/40"></div>
      <div className="max-w-container relative mx-auto">
        <NavbarComponent>
          <NavbarLeft>
            <a
              href={homeUrl}
              className="flex items-center gap-3"
            >
              <SimpleLogo size="md" />
              <span className="text-xl font-bold tracking-tight text-foreground">
                {name}
              </span>
            </a>
            {showNavigation && (customNavigation || <Navigation />)}
          </NavbarLeft>
          <NavbarRight>
            {/* Social Media Links */}
            <div className="hidden md:flex items-center gap-2">
              <Button variant="ghost" size="sm" asChild className="nav-item">
                <a 
                  href={siteConfig.links.youtube} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  aria-label="YouTube"
                >
                  <Youtube className="size-4" />
                </a>
              </Button>
              <Button variant="ghost" size="sm" asChild className="nav-item">
                <a 
                  href={siteConfig.links.linkedin} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  aria-label="LinkedIn"
                >
                  <Linkedin className="size-4" />
                </a>
              </Button>
              <Button variant="ghost" size="sm" asChild className="nav-item">
                <a 
                  href={siteConfig.links.instagram} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  aria-label="Instagram"
                >
                  <Instagram className="size-4" />
                </a>
              </Button>
            </div>
            
            <SignedOut>
              <SignInButton>
                <Button variant="ghost" size="sm" className="nav-item">
                  Sign In
                </Button>
              </SignInButton>
              <SignUpButton>
                <Button 
                  variant="default" 
                  size="sm" 
                  className="btn-gradient-primary hover-lift"
                >
                  Get Started
                </Button>
              </SignUpButton>
            </SignedOut>
            <SignedIn>
              <Button variant="ghost" size="sm" asChild className="nav-item">
                <a href="/contact">Contact</a>
              </Button>
              <UserButton 
                appearance={{
                  elements: {
                    avatarBox: "w-8 h-8 hover:shadow-lg hover:shadow-blue-500/20 transition-all duration-300"
                  }
                }}
              />
            </SignedIn>
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
                    className="flex items-center gap-3 text-xl font-bold"
                  >
                    <SimpleLogo size="sm" />
                    <span>{name}</span>
                  </a>
                  {mobileLinks.map((link, index) => (
                    <a
                      key={index}
                      href={link.href}
                      className="text-muted-foreground hover:text-foreground transition-colors duration-200"
                    >
                      {link.text}
                    </a>
                  ))}
                  <div className="border-t border-border/50 pt-4 mt-4">
                    {/* Social Media Links for Mobile */}
                    <div className="flex items-center gap-4 mb-4 flex-wrap">
                      <a 
                        href={siteConfig.links.youtube} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors"
                      >
                        <Youtube className="size-4" />
                        <span className="text-sm">YouTube</span>
                      </a>
                      <a 
                        href={siteConfig.links.linkedin} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors"
                      >
                        <Linkedin className="size-4" />
                        <span className="text-sm">LinkedIn</span>
                      </a>
                      <a 
                        href={siteConfig.links.instagram} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors"
                      >
                        <Instagram className="size-4" />
                        <span className="text-sm">Instagram</span>
                      </a>
                    </div>
                    
                    <SignedOut>
                      <div className="flex flex-col gap-3">
                        <SignInButton>
                          <Button variant="ghost" className="justify-start">
                            Sign In
                          </Button>
                        </SignInButton>
                        <SignUpButton>
                          <Button className="btn-gradient-primary justify-start">
                            Get Started
                          </Button>
                        </SignUpButton>
                      </div>
                    </SignedOut>
                    <SignedIn>
                      <div className="flex items-center gap-3">
                        <UserButton />
                        <span className="text-sm text-muted-foreground">Account</span>
                      </div>
                    </SignedIn>
                  </div>
                </nav>
              </SheetContent>
            </Sheet>
          </NavbarRight>
        </NavbarComponent>
      </div>
    </header>
  );
}
