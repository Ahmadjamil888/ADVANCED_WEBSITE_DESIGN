import { Badge } from "@/components/ui/badge";
import { Section } from "@/components/ui/section";
import { Users, Code, BrainCircuit, TrendingUp, Award, Rocket, Target } from "lucide-react";
import Link from "next/link";
import Navbar from "@/components/sections/navbar/default";
import Footer from "@/components/sections/footer/default";

export const metadata = {
  title: 'Our Journey - From Web Agency to AI Leaders | Zehan X Technologies',
  description: 'Discover how Zehan X Technologies evolved from a small web development agency to a leading AI and machine learning company, transforming businesses worldwide.',
};

const milestones = [
  {
    year: "2020",
    title: "The Beginning",
    description: "Started as a small web development agency with a passion for creating beautiful, functional websites for local businesses.",
    icon: <Code className="size-6" />,
    achievements: [
      "Founded with 2 developers",
      "First 10 client websites delivered",
      "Focus on responsive web design",
      "Local business partnerships"
    ]
  },
  {
    year: "2021",
    title: "Growing Our Expertise",
    description: "Expanded our team and capabilities, specializing in modern web technologies like React and Next.js.",
    icon: <Users className="size-6" />,
    achievements: [
      "Team grew to 5 developers",
      "50+ websites delivered",
      "Specialized in React & Next.js",
      "First enterprise clients"
    ]
  },
  {
    year: "2022",
    title: "The AI Revolution Begins",
    description: "Recognized the potential of AI and began integrating machine learning capabilities into our web solutions.",
    icon: <BrainCircuit className="size-6" />,
    achievements: [
      "First AI-powered web applications",
      "Machine learning integrations",
      "Data analytics dashboards",
      "AI chatbot implementations"
    ]
  },
  {
    year: "2023",
    title: "Becoming AI Specialists",
    description: "Pivoted to become a full-service AI and machine learning company while maintaining our web development excellence.",
    icon: <Rocket className="size-6" />,
    achievements: [
      "Custom ML model development",
      "Deep learning solutions",
      "Computer vision projects",
      "Predictive analytics systems"
    ]
  },
  {
    year: "2024",
    title: "Industry Recognition",
    description: "Established ourselves as leaders in AI-powered business solutions, serving clients across multiple industries.",
    icon: <Award className="size-6" />,
    achievements: [
      "100+ AI projects completed",
      "Enterprise-grade solutions",
      "Industry partnerships",
      "Thought leadership content"
    ]
  },
  {
    year: "2025",
    title: "The Future is Now",
    description: "Continuing to innovate and push the boundaries of what's possible with AI and modern web technologies.",
    icon: <Target className="size-6" />,
    achievements: [
      "Advanced AI research",
      "Global client base",
      "Cutting-edge solutions",
      "Industry transformation"
    ]
  }
];

const stats = [
  {
    number: "500+",
    label: "Projects Completed",
    description: "From simple websites to complex AI systems"
  },
  {
    number: "200+",
    label: "Happy Clients",
    description: "Businesses transformed with our solutions"
  },
  {
    number: "50+",
    label: "AI Models Deployed",
    description: "Custom machine learning solutions in production"
  },
  {
    number: "5",
    label: "Years of Innovation",
    description: "Continuous growth and technological advancement"
  }
];

export default function Journey() {
  return (
    <main className="min-h-screen w-full overflow-hidden bg-background text-foreground">
      <Navbar />
      
      {/* Hero Section */}
      <Section className="pt-24 pb-16">
        <div className="max-w-container mx-auto text-center">
          <Badge variant="outline" className="mb-6">
            <TrendingUp className="mr-2 size-4" />
            Our Journey
          </Badge>
          <h1 className="text-4xl font-bold mb-6 sm:text-6xl">
            From Web Agency to AI Leaders
          </h1>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Discover how Zehan X Technologies evolved from a small web development agency 
            to a leading AI and machine learning company, transforming businesses worldwide 
            with cutting-edge technology solutions.
          </p>
        </div>
      </Section>

      {/* Stats Section */}
      <Section className="py-16 bg-muted/30">
        <div className="max-w-container mx-auto">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <div key={index} className="text-center">
                <div className="text-3xl font-bold text-primary mb-2">{stat.number}</div>
                <div className="font-semibold mb-1">{stat.label}</div>
                <div className="text-sm text-muted-foreground">{stat.description}</div>
              </div>
            ))}
          </div>
        </div>
      </Section>

      {/* Timeline Section */}
      <Section className="py-16">
        <div className="max-w-container mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold mb-4">Our Evolution Timeline</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Follow our journey from a small web development team to becoming industry leaders 
              in AI and machine learning solutions.
            </p>
          </div>

          <div className="relative">
            {/* Timeline line */}
            <div className="absolute left-8 md:left-1/2 top-0 bottom-0 w-0.5 bg-border transform md:-translate-x-0.5"></div>
            
            <div className="space-y-12">
              {milestones.map((milestone, index) => (
                <div key={index} className={`relative flex items-center ${
                  index % 2 === 0 ? 'md:flex-row' : 'md:flex-row-reverse'
                }`}>
                  {/* Timeline dot */}
                  <div className="absolute left-8 md:left-1/2 size-4 bg-primary rounded-full transform -translate-x-2 md:-translate-x-2 z-10"></div>
                  
                  {/* Content */}
                  <div className={`flex-1 ml-16 md:ml-0 ${
                    index % 2 === 0 ? 'md:pr-8' : 'md:pl-8'
                  }`}>
                    <div className="p-6 rounded-lg border bg-background shadow-sm">
                      <div className="flex items-center gap-3 mb-4">
                        <div className="size-12 bg-primary/10 rounded-lg flex items-center justify-center text-primary">
                          {milestone.icon}
                        </div>
                        <div>
                          <div className="text-sm font-medium text-primary">{milestone.year}</div>
                          <h3 className="text-xl font-bold">{milestone.title}</h3>
                        </div>
                      </div>
                      
                      <p className="text-muted-foreground mb-4 leading-relaxed">
                        {milestone.description}
                      </p>
                      
                      <div className="grid grid-cols-2 gap-2">
                        {milestone.achievements.map((achievement, i) => (
                          <div key={i} className="flex items-center gap-2 text-sm">
                            <div className="size-1.5 bg-primary rounded-full"></div>
                            <span>{achievement}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                  
                  {/* Spacer for alternating layout */}
                  <div className="hidden md:block flex-1"></div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </Section>

      {/* Vision Section */}
      <Section className="py-16 bg-muted/30">
        <div className="max-w-container mx-auto">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div>
              <Badge variant="outline" className="mb-4">
                <Target className="mr-2 size-4" />
                Our Vision
              </Badge>
              <h2 className="text-3xl font-bold mb-6">
                Shaping the Future of Business with AI
              </h2>
              <p className="text-muted-foreground mb-6 leading-relaxed">
                What started as a passion for creating beautiful websites has evolved into a mission 
                to democratize artificial intelligence. We believe that every business, regardless of 
                size, should have access to the transformative power of AI and modern web technologies.
              </p>
              <p className="text-muted-foreground mb-6 leading-relaxed">
                Our journey from a small web agency to an AI powerhouse has taught us that innovation 
                comes from understanding real business problems and applying cutting-edge technology 
                to solve them effectively.
              </p>
              <div className="flex gap-4">
                <Link 
                  href="/services" 
                  className="inline-flex items-center justify-center rounded-md bg-primary px-6 py-3 text-sm font-medium text-primary-foreground shadow transition-colors hover:bg-primary/90"
                >
                  Explore Our Services
                </Link>
                <Link 
                  href="/portfolio" 
                  className="inline-flex items-center justify-center rounded-md border border-input bg-background px-6 py-3 text-sm font-medium shadow-sm transition-colors hover:bg-accent hover:text-accent-foreground"
                >
                  View Our Work
                </Link>
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-6">
              <div className="p-6 bg-background rounded-lg border">
                <Code className="size-8 text-primary mb-4" />
                <h3 className="font-semibold mb-2">Web Excellence</h3>
                <p className="text-sm text-muted-foreground">
                  Still delivering world-class web applications with modern technologies
                </p>
              </div>
              <div className="p-6 bg-background rounded-lg border">
                <BrainCircuit className="size-8 text-primary mb-4" />
                <h3 className="font-semibold mb-2">AI Innovation</h3>
                <p className="text-sm text-muted-foreground">
                  Leading the way in machine learning and artificial intelligence
                </p>
              </div>
              <div className="p-6 bg-background rounded-lg border">
                <Users className="size-8 text-primary mb-4" />
                <h3 className="font-semibold mb-2">Client Success</h3>
                <p className="text-sm text-muted-foreground">
                  Dedicated to transforming businesses and driving growth
                </p>
              </div>
              <div className="p-6 bg-background rounded-lg border">
                <Rocket className="size-8 text-primary mb-4" />
                <h3 className="font-semibold mb-2">Future Ready</h3>
                <p className="text-sm text-muted-foreground">
                  Continuously evolving with emerging technologies
                </p>
              </div>
            </div>
          </div>
        </div>
      </Section>

      {/* CTA Section */}
      <Section className="py-16">
        <div className="max-w-container mx-auto text-center">
          <h2 className="text-3xl font-bold mb-6">
            Be Part of Our Continuing Journey
          </h2>
          <p className="text-muted-foreground mb-8 max-w-2xl mx-auto">
            Join us as we continue to push the boundaries of what's possible with AI and web technology. 
            Let's build the future together.
          </p>
          <div className="flex justify-center gap-4">
            <Link 
              href="/contact" 
              className="inline-flex items-center justify-center rounded-md bg-primary px-8 py-3 text-sm font-medium text-primary-foreground shadow transition-colors hover:bg-primary/90"
            >
              Start Your Project
            </Link>
            <Link 
              href="/about" 
              className="inline-flex items-center justify-center rounded-md border border-input bg-background px-8 py-3 text-sm font-medium shadow-sm transition-colors hover:bg-accent hover:text-accent-foreground"
            >
              Learn More About Us
            </Link>
          </div>
        </div>
      </Section>
      
      <Footer />
    </main>
  );
}