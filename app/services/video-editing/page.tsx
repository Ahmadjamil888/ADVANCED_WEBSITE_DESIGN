import { Badge } from "@/components/ui/badge";
import { Section } from "@/components/ui/section";
import { Video, Film, Camera, Play, Edit, Zap, CheckCircle, Clock } from "lucide-react";
import Link from "next/link";
import Navbar from "@/components/sections/navbar/default";
import Footer from "@/components/sections/footer/default";

const features = [
  {
    icon: <Film className="size-6" />,
    title: "Professional Video Editing",
    description: "High-quality editing for commercials, documentaries, and promotional videos."
  },
  {
    icon: <Camera className="size-6" />,
    title: "Social Media Content",
    description: "Short-form videos optimized for Instagram, TikTok, YouTube Shorts, and more."
  },
  {
    icon: <Play className="size-6" />,
    title: "Motion Graphics",
    description: "Animated graphics, titles, and visual effects to enhance your videos."
  },
  {
    icon: <Edit className="size-6" />,
    title: "Color Correction",
    description: "Professional color grading and correction for cinematic quality."
  },
  {
    icon: <Video className="size-6" />,
    title: "Video Production",
    description: "End-to-end video production from concept to final delivery."
  },
  {
    icon: <Clock className="size-6" />,
    title: "Quick Turnaround",
    description: "Fast delivery without compromising on quality and attention to detail."
  }
];

const packages = [
  {
    name: "Basic Edit",
    price: "Starting at $199",
    features: [
      "Basic cuts and transitions",
      "Audio sync and cleanup",
      "Simple titles and graphics",
      "1080p HD export",
      "2 revisions included"
    ]
  },
  {
    name: "Professional Edit", 
    price: "Starting at $399",
    features: [
      "Advanced editing techniques",
      "Color correction & grading",
      "Motion graphics & animations",
      "4K export capabilities",
      "Professional audio mixing",
      "Unlimited revisions"
    ],
    popular: true
  },
  {
    name: "Premium Production",
    price: "Starting at $799",
    features: [
      "Complete video production",
      "Advanced visual effects",
      "Custom animations & graphics",
      "Multiple format exports",
      "Sound design & mixing",
      "Dedicated video editor",
      "Priority support"
    ]
  }
];

export default function VideoEditingService() {
  return (
    <main className="min-h-screen w-full overflow-hidden bg-background text-foreground">
      <Navbar />
      
      {/* Hero Section */}
      <Section className="pt-24 pb-16">
        <div className="max-w-container mx-auto">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            <div>
              <Badge variant="outline" className="mb-6">
                <Video className="mr-2 size-4" />
                Video Editing Services
              </Badge>
              <h1 className="text-4xl font-bold mb-6 sm:text-6xl">
                Bring Your Stories to Life with Professional Video Editing
              </h1>
              <p className="text-xl text-muted-foreground mb-8">
                Transform raw footage into compelling narratives that engage, inspire, 
                and captivate your audience. From social media clips to full productions.
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
              <div className="aspect-video rounded-lg bg-gradient-to-br from-primary/20 to-secondary/20 p-8 flex items-center justify-center">
                <Video className="size-32 text-primary" />
              </div>
            </div>
          </div>
        </div>
      </Section>

      {/* Features Section */}
      <Section className="py-16 bg-muted/30">
        <div className="max-w-container mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Our Video Services</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              From basic editing to complex post-production, we handle every aspect 
              of video creation to deliver content that stands out.
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
            <h2 className="text-3xl font-bold mb-4">Video Editing Packages</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Choose the perfect package for your video editing needs. All packages include 
              professional consultation and high-quality deliverables.
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

      {/* Video Types Section */}
      <Section className="py-16 bg-muted/30">
        <div className="max-w-container mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Video Types We Edit</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              We specialize in editing various types of video content across different industries and platforms.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              { title: "Commercial Videos", description: "Product launches, brand stories, advertisements" },
              { title: "Social Media", description: "Instagram Reels, TikTok, YouTube content" },
              { title: "Corporate Videos", description: "Training videos, presentations, testimonials" },
              { title: "Event Coverage", description: "Weddings, conferences, live events" },
              { title: "Educational Content", description: "Tutorials, online courses, explainer videos" },
              { title: "Music Videos", description: "Performance videos, lyric videos, music content" },
              { title: "Documentary", description: "Long-form storytelling, interviews, narratives" },
              { title: "Real Estate", description: "Property tours, virtual walkthroughs" }
            ].map((type, index) => (
              <div key={index} className="p-6 rounded-lg bg-background border text-center">
                <h3 className="text-lg font-semibold mb-2">{type.title}</h3>
                <p className="text-muted-foreground text-sm">{type.description}</p>
              </div>
            ))}
          </div>
        </div>
      </Section>

      {/* Process Section */}
      <Section className="py-16">
        <div className="max-w-container mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Our Editing Process</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              We follow a structured workflow to ensure your video meets the highest standards of quality and storytelling.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            {[
              { step: "01", title: "Review", description: "Analyzing your footage and understanding your vision" },
              { step: "02", title: "Edit", description: "Crafting the story with professional editing techniques" },
              { step: "03", title: "Enhance", description: "Adding graphics, effects, and audio improvements" },
              { step: "04", title: "Deliver", description: "Final review and delivery in your preferred format" }
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
      <Section className="py-16 bg-muted/30">
        <div className="max-w-container mx-auto text-center">
          <h2 className="text-3xl font-bold mb-6">
            Ready to Create Amazing Videos?
          </h2>
          <p className="text-muted-foreground mb-8 max-w-2xl mx-auto">
            Let's transform your raw footage into compelling stories that resonate 
            with your audience and achieve your goals.
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
