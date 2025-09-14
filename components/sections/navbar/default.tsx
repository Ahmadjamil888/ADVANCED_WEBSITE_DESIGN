import { Menu } from "lucide-react";
import { ReactNode } from "react";
import {
  SignInButton,
  SignUpButton,
  SignedIn,
  SignedOut,
  UserButton,
} from '@clerk/nextjs';

import { cn } from "@/lib/utils";
import ZehanLogo from "../../logos/zehan-logo";

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
    { text: "Custom Orders", href: "/pricing" },
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
              <ZehanLogo size="md" />
              <span className="text-xl font-bold tracking-tight text-foreground">
                {name}
              </span>
            </a>
            {showNavigation && (customNavigation || <Navigation />)}
          </NavbarLeft>
          <NavbarRight>
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
                    <ZehanLogo size="sm" />
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
