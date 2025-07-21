"use client";

import { Button } from '@/components/ui/button';
import { Section } from '@/components/ui/section';
import { Search, Home, ArrowLeft } from 'lucide-react';
import Link from 'next/link';
import Navbar from '@/components/sections/navbar/default';
import Footer from '@/components/sections/footer/default';

export default function NotFound() {
  return (
    <main className="min-h-screen w-full overflow-hidden bg-background text-foreground">
      <Navbar />
      
      <Section className="py-24">
        <div className="max-w-container mx-auto text-center">
          <div className="size-24 bg-muted/30 rounded-full flex items-center justify-center mx-auto mb-8">
            <Search className="size-12 text-muted-foreground" />
          </div>
          
          <h1 className="text-6xl font-bold mb-4">404</h1>
          <h2 className="text-2xl font-semibold mb-4">Page Not Found</h2>
          
          <p className="text-muted-foreground mb-8 max-w-md mx-auto">
            The page you are looking for might have been removed, had its name changed, 
            or is temporarily unavailable.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center mb-12">
            <Button asChild>
              <Link href="/" className="flex items-center gap-2">
                <Home className="size-4" />
                Go Home
              </Link>
            </Button>
            
            <Button variant="outline" onClick={() => window.history.back()}>
              <ArrowLeft className="size-4 mr-2" />
              Go Back
            </Button>
          </div>
          
          <div className="grid md:grid-cols-3 gap-6 max-w-2xl mx-auto">
            <div className="p-4 border rounded-lg">
              <h3 className="font-semibold mb-2">Our Services</h3>
              <p className="text-sm text-muted-foreground mb-3">
                Explore our AI and web development solutions
              </p>
              <Link href="/services" className="text-primary hover:underline text-sm">
                View Services →
              </Link>
            </div>
            
            <div className="p-4 border rounded-lg">
              <h3 className="font-semibold mb-2">About Us</h3>
              <p className="text-sm text-muted-foreground mb-3">
                Learn more about Zehan X Technologies
              </p>
              <Link href="/about" className="text-primary hover:underline text-sm">
                About Us →
              </Link>
            </div>
            
            <div className="p-4 border rounded-lg">
              <h3 className="font-semibold mb-2">Contact</h3>
              <p className="text-sm text-muted-foreground mb-3">
                Get in touch with our team
              </p>
              <Link href="/contact" className="text-primary hover:underline text-sm">
                Contact Us →
              </Link>
            </div>
          </div>
        </div>
      </Section>
      
      <Footer />
    </main>
  );
}