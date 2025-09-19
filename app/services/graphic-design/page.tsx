import { Badge } from "@/components/ui/badge";
import { Section } from "@/components/ui/section";
import { Palette, Layers, Brush, Image as ImageIcon, Sparkles, Target, Zap, CheckCircle } from "lucide-react";
import Link from "next/link";
import Navbar from "@/components/sections/navbar/default";
import Footer from "@/components/sections/footer/default";

const features = [
  {
    icon: <Palette className="size-6" />,
    title: "Brand Identity Design",
    description: "Complete brand packages including logos, color schemes, and visual guidelines."
  },
  {
    icon: <Layers className="size-6" />,
    title: "Print Design",
    description: "Business cards, brochures, flyers, posters, and marketing materials."
  },
  {
    icon: <Brush className="size-6" />,
    title: "Digital Graphics",
    description: "Social media graphics, web banners, and digital marketing assets."
  },
  {
    icon: <ImageIcon className="size-6" />,
    title: "UI/UX Design",
    description: "User interface design for web and mobile applications."
  },
  {
    icon: <Sparkles className="size-6" />,
    title: "Creative Illustrations",
    description: "Custom illustrations, icons, and artistic visual elements."
  },
  {
    icon: <Target className="size-6" />,
    title: "Marketing Materials",
    description: "Eye-catching designs for advertisements, campaigns, and promotions."
  }
];

const packages = [
  {
    name: "Basic Package",
    price: "Starting at $299",
    features: [
      "Logo Design (3 concepts)",
      "Business Card Design",
      "Letterhead Design", 
      "2 Revisions Included",
      "High-Resolution Files"
    ]
  },
  {
    name: "Professional Package", 
    price: "Starting at $599",
    features: [
      "Complete Brand Identity",
      "Logo + Variations",
      "Business Stationery Set",
      "Social Media Templates",
      "Brand Guidelines",
      "Unlimited Revisions"
    ],
    popular: true
  },
  {
    name: "Enterprise Package",
    price: "Starting at $1299",
    features: [
      "Full Brand Strategy",
      "Complete Visual Identity",
      "Marketing Material Suite",
      "Web Graphics Package",
      "Print Design Assets",
      "Dedicated Designer",
      "Priority Support"
    ]
  }
];

export default function GraphicDesignService() {
  return (
    <main className="min-h-screen w-full overflow-hidden bg-background text-foreground">
      <Navbar />
      
      {/* Hero Section */}
      <Section className="pt-24 pb-16">
        <div className="max-w-container mx-auto">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            <div>
              <Badge variant="outline" className="mb-6">
                <Palette className="mr-2 size-4" />
                Graphic Design Services
              </Badge>
              <h1 className="text-4xl font-bold mb-6 sm:text-6xl">
                Creative Visual Solutions That Make an Impact
              </h1>
              <p className="text-xl text-muted-foreground mb-8">
                Transform your brand with stunning visual designs that captivate audiences 
                and drive engagement. From logos to marketing materials, we bring your vision to life.
              </p>
              <div className="flex gap-4">
                <Link 
                  href="/contact" 
                  className="inline-flex items-center justify-center rounded-md bg-primary px-8 py-3 text-sm font-medium text-primary-foreground shadow transition-colors hover:bg-primary/90"
                >
                  <Zap className="mr-2 size-4" />
                  Start Your Project
                </Link>
                <Link 
                  href="/portfolio" 
                  className="inline-flex items-center justify-center rounded-md border border-input bg-background px-8 py-3 text-sm font-medium shadow-sm transition-colors hover:bg-accent hover:text-accent-foreground"
                >
                  View Portfolio
                </Link>
              </div>
            </div>
            <div className="relative">
              <div className="aspect-square rounded-lg bg-gradient-to-br from-primary/20 to-secondary/20 p-8 flex items-center justify-center">
                <Palette className="size-32 text-primary" />
              </div>
            </div>
          </div>
        </div>
      </Section>

      {/* Features Section */}
      <Section className="py-16 bg-muted/30">
        <div className="max-w-container mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Our Design Services</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              From concept to creation, we offer comprehensive graphic design services 
              tailored to your brand's unique needs and vision.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <div 
                key={index} 
                className="p-6 rounded-lg bg-background border hover:shadow-lg transition-all duration-300"
              >
                <div className="flex items-center gap-4 mb-4">
                  <div className="p-2 rounded-lg bg-primary/10 text-primary">
                    {feature.icon}
                  </div>
                  <h3 className="text-lg font-semibold">{feature.title}</h3>
                </div>
                <p className="text-muted-foreground">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </Section>

      {/* Packages Section */}
      <Section className="py-16">
        <div className="max-w-container mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Design Packages</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Choose the perfect package for your design needs. All packages include 
              professional design consultation and high-quality deliverables.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {packages.map((pkg, index) => (
              <div 
                key={index} 
                className={`relative p-8 rounded-lg border transition-all duration-300 hover:shadow-lg ${
                  pkg.popular ? 'border-primary bg-primary/5 scale-105' : 'bg-background'
                }`}
              >
                {pkg.popular && (
                  <Badge className="absolute -top-3 left-1/2 -translate-x-1/2 bg-primary">
                    Most Popular
                  </Badge>
                )}
                
                <div className="text-center mb-6">
                  <h3 className="text-xl font-bold mb-2">{pkg.name}</h3>
                  <p className="text-2xl font-bold text-primary">{pkg.price}</p>
                </div>
                
                <ul className="space-y-3 mb-8">
                  {pkg.features.map((feature, featureIndex) => (
                    <li key={featureIndex} className="flex items-center gap-3">
                      <CheckCircle className="size-5 text-green-500 flex-shrink-0" />
                      <span className="text-sm">{feature}</span>
                    </li>
                  ))}
                </ul>
                
                <Link 
                  href="/contact" 
                  className={`w-full inline-flex items-center justify-center rounded-md px-6 py-3 text-sm font-medium transition-colors ${
                    pkg.popular 
                      ? 'bg-primary text-primary-foreground hover:bg-primary/90' 
                      : 'border border-input bg-background hover:bg-accent hover:text-accent-foreground'
                  }`}
                >
                  Get Started
                </Link>
              </div>
            ))}
          </div>
        </div>
      </Section>

      {/* Process Section */}
      <Section className="py-16 bg-muted/30">
        <div className="max-w-container mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Our Design Process</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              We follow a proven design process to ensure exceptional results that exceed your expectations.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            {[
              { step: "01", title: "Discovery", description: "Understanding your brand, goals, and target audience" },
              { step: "02", title: "Concept", description: "Creating initial design concepts and mood boards" },
              { step: "03", title: "Design", description: "Developing refined designs with your feedback" },
              { step: "04", title: "Delivery", description: "Final designs in all required formats and sizes" }
            ].map((process, index) => (
              <div key={index} className="text-center">
                <div className="size-16 rounded-full bg-primary/10 text-primary flex items-center justify-center text-xl font-bold mx-auto mb-4">
                  {process.step}
                </div>
                <h3 className="text-lg font-semibold mb-2">{process.title}</h3>
                <p className="text-muted-foreground text-sm">{process.description}</p>
              </div>
            ))}
          </div>
        </div>
      </Section>

      {/* CTA Section */}
      <Section className="py-16">
        <div className="max-w-container mx-auto text-center">
          <h2 className="text-3xl font-bold mb-6">
            Ready to Elevate Your Brand?
          </h2>
          <p className="text-muted-foreground mb-8 max-w-2xl mx-auto">
            Let's create stunning visual designs that make your brand stand out 
            and connect with your audience on a deeper level.
          </p>
          <div className="flex justify-center gap-4">
            <Link 
              href="/contact" 
              className="inline-flex items-center justify-center rounded-md bg-primary px-8 py-3 text-sm font-medium text-primary-foreground shadow transition-colors hover:bg-primary/90"
            >
              Start Your Project
            </Link>
            <Link 
              href="/services" 
              className="inline-flex items-center justify-center rounded-md border border-input bg-background px-8 py-3 text-sm font-medium shadow-sm transition-colors hover:bg-accent hover:text-accent-foreground"
            >
              View All Services
            </Link>
          </div>
        </div>
      </Section>

      <Footer />
    </main>
  );
}
