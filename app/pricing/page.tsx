import { Badge } from "@/components/ui/badge";
import { Section } from "@/components/ui/section";
import { Check, Zap, Crown, Rocket, ArrowRight } from "lucide-react";
import Link from "next/link";
import Navbar from "@/components/sections/navbar/default";
import Footer from "@/components/sections/footer/default";

export const metadata = {
  title: 'Pricing - AI & Web Development Services | Zehan X Technologies',
  description: 'Transparent pricing for our AI solutions, machine learning services, and web development projects. Choose the perfect plan for your business needs.',
};

const pricingPlans = [
  {
    name: "Starter",
    description: "Perfect for small businesses and startups",
    price: "$2,500",
    period: "per project",
    icon: <Zap className="size-6" />,
    popular: false,
    features: [
      "Basic AI integration",
      "Simple web application",
      "Up to 5 pages/components",
      "Responsive design",
      "Basic SEO optimization",
      "2 rounds of revisions",
      "30 days support",
      "Source code included"
    ],
    limitations: [
      "No advanced ML models",
      "Limited customization",
      "Basic analytics only"
    ]
  },
  {
    name: "Professional",
    description: "Ideal for growing businesses with advanced needs",
    price: "$7,500",
    period: "per project",
    icon: <Crown className="size-6" />,
    popular: true,
    features: [
      "Custom AI/ML solutions",
      "Full-stack web application",
      "Unlimited pages/components",
      "Advanced responsive design",
      "Complete SEO optimization",
      "Database integration",
      "API development",
      "5 rounds of revisions",
      "90 days support",
      "Performance optimization",
      "Security implementation",
      "Documentation included"
    ],
    limitations: [
      "Complex enterprise features require consultation"
    ]
  },
  {
    name: "Enterprise",
    description: "For large organizations with complex requirements",
    price: "Custom",
    period: "quote",
    icon: <Rocket className="size-6" />,
    popular: false,
    features: [
      "Advanced AI/ML systems",
      "Enterprise-grade applications",
      "Microservices architecture",
      "Advanced security features",
      "Custom integrations",
      "Scalable infrastructure",
      "DevOps & CI/CD setup",
      "Unlimited revisions",
      "1 year support & maintenance",
      "Performance monitoring",
      "Staff training included",
      "24/7 priority support",
      "Dedicated project manager",
      "Custom SLA agreement"
    ],
    limitations: []
  }
];

const additionalServices = [
  {
    service: "AI Model Training",
    description: "Custom machine learning model development and training",
    price: "Starting at $3,000"
  },
  {
    service: "Data Analytics Dashboard",
    description: "Interactive dashboards with real-time analytics",
    price: "Starting at $2,000"
  },
  {
    service: "AI Chatbot Development",
    description: "Intelligent chatbots with NLP capabilities",
    price: "Starting at $1,500"
  },
  {
    service: "API Development",
    description: "RESTful APIs and GraphQL endpoints",
    price: "Starting at $1,000"
  },
  {
    service: "Mobile App Development",
    description: "React Native or Flutter mobile applications",
    price: "Starting at $5,000"
  },
  {
    service: "Maintenance & Support",
    description: "Ongoing maintenance and technical support",
    price: "$500/month"
  }
];

export default function Pricing() {
  return (
    <main className="min-h-screen w-full overflow-hidden bg-background text-foreground">
      <Navbar />
      
      {/* Hero Section */}
      <Section className="pt-24 pb-16">
        <div className="max-w-container mx-auto text-center">
          <Badge variant="outline" className="mb-6">
            <Crown className="mr-2 size-4" />
            Pricing Plans
          </Badge>
          <h1 className="text-4xl font-bold mb-6 sm:text-6xl">
            Transparent Pricing for AI Solutions
          </h1>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Choose the perfect plan for your business needs. All plans include source code, 
            documentation, and our commitment to delivering exceptional results.
          </p>
        </div>
      </Section>

      {/* Pricing Cards */}
      <Section className="py-16">
        <div className="max-w-container mx-auto">
          <div className="grid md:grid-cols-3 gap-8">
            {pricingPlans.map((plan, index) => (
              <div 
                key={index} 
                className={`relative p-8 rounded-lg border bg-background ${
                  plan.popular 
                    ? 'border-primary shadow-lg scale-105' 
                    : 'hover:shadow-lg'
                } transition-all duration-300`}
              >
                {plan.popular && (
                  <div className="absolute -top-3 left-1/2 transform -translate-x-1/2">
                    <span className="bg-primary text-primary-foreground px-4 py-1 rounded-full text-sm font-medium">
                      Most Popular
                    </span>
                  </div>
                )}
                
                <div className="text-center mb-8">
                  <div className="size-16 bg-primary/10 rounded-lg flex items-center justify-center mx-auto mb-4 text-primary">
                    {plan.icon}
                  </div>
                  <h3 className="text-2xl font-bold mb-2">{plan.name}</h3>
                  <p className="text-muted-foreground mb-4">{plan.description}</p>
                  <div className="mb-4">
                    <span className="text-4xl font-bold">{plan.price}</span>
                    <span className="text-muted-foreground ml-2">{plan.period}</span>
                  </div>
                </div>

                <div className="space-y-4 mb-8">
                  <h4 className="font-semibold text-sm">What's included:</h4>
                  <ul className="space-y-3">
                    {plan.features.map((feature, i) => (
                      <li key={i} className="flex items-start gap-3">
                        <Check className="size-4 text-green-500 mt-0.5 flex-shrink-0" />
                        <span className="text-sm">{feature}</span>
                      </li>
                    ))}
                  </ul>
                  
                  {plan.limitations.length > 0 && (
                    <div className="pt-4 border-t">
                      <h4 className="font-semibold text-sm text-muted-foreground mb-2">Limitations:</h4>
                      <ul className="space-y-2">
                        {plan.limitations.map((limitation, i) => (
                          <li key={i} className="text-xs text-muted-foreground">
                            • {limitation}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>

                <Link
                  href="/contact"
                  className={`w-full inline-flex items-center justify-center rounded-md px-6 py-3 text-sm font-medium transition-colors ${
                    plan.popular
                      ? 'bg-primary text-primary-foreground hover:bg-primary/90'
                      : 'border border-input bg-background hover:bg-accent hover:text-accent-foreground'
                  }`}
                >
                  {plan.name === 'Enterprise' ? 'Get Custom Quote' : 'Get Started'}
                  <ArrowRight className="ml-2 size-4" />
                </Link>
              </div>
            ))}
          </div>
        </div>
      </Section>

      {/* Additional Services */}
      <Section className="py-16 bg-muted/30">
        <div className="max-w-container mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Additional Services</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Need something specific? We offer additional services that can be added to any plan 
              or purchased separately.
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {additionalServices.map((service, index) => (
              <div key={index} className="p-6 rounded-lg border bg-background">
                <h3 className="font-semibold mb-2">{service.service}</h3>
                <p className="text-muted-foreground text-sm mb-4">{service.description}</p>
                <div className="flex items-center justify-between">
                  <span className="font-medium text-primary">{service.price}</span>
                  <button className="text-sm text-muted-foreground hover:text-primary transition-colors">
                    Learn More →
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </Section>

      {/* FAQ Section */}
      <Section className="py-16">
        <div className="max-w-container mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Frequently Asked Questions</h2>
          </div>
          
          <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
            <div>
              <h3 className="font-semibold mb-2">What's included in the project price?</h3>
              <p className="text-muted-foreground text-sm mb-6">
                All plans include complete source code, documentation, testing, deployment assistance, 
                and the specified support period.
              </p>
              
              <h3 className="font-semibold mb-2">Do you offer payment plans?</h3>
              <p className="text-muted-foreground text-sm mb-6">
                Yes, we offer flexible payment plans with 50% upfront and 50% upon completion 
                for Professional and Enterprise plans.
              </p>
              
              <h3 className="font-semibold mb-2">What if I need changes after launch?</h3>
              <p className="text-muted-foreground text-sm">
                Minor changes are included in your support period. Major changes can be handled 
                through our maintenance service or as a separate project.
              </p>
            </div>
            
            <div>
              <h3 className="font-semibold mb-2">How long does a typical project take?</h3>
              <p className="text-muted-foreground text-sm mb-6">
                Starter projects: 2-4 weeks, Professional: 4-8 weeks, Enterprise: 8-16 weeks. 
                Timeline depends on complexity and requirements.
              </p>
              
              <h3 className="font-semibold mb-2">Do you provide hosting and maintenance?</h3>
              <p className="text-muted-foreground text-sm mb-6">
                We can help with deployment and recommend hosting solutions. Ongoing maintenance 
                is available as an additional service.
              </p>
              
              <h3 className="font-semibold mb-2">Can I upgrade my plan later?</h3>
              <p className="text-muted-foreground text-sm">
                Absolutely! You can upgrade your plan at any time during the project. 
                We'll adjust the scope and pricing accordingly.
              </p>
            </div>
          </div>
        </div>
      </Section>

      {/* CTA Section */}
      <Section className="py-16 bg-muted/30">
        <div className="max-w-container mx-auto text-center">
          <h2 className="text-3xl font-bold mb-6">
            Ready to Transform Your Business?
          </h2>
          <p className="text-muted-foreground mb-8 max-w-2xl mx-auto">
            Let's discuss your project requirements and find the perfect solution for your business. 
            Get a free consultation and custom quote today.
          </p>
          <div className="flex justify-center gap-4">
            <Link 
              href="/contact" 
              className="inline-flex items-center justify-center rounded-md bg-primary px-8 py-3 text-sm font-medium text-primary-foreground shadow transition-colors hover:bg-primary/90"
            >
              Get Free Consultation
            </Link>
            <Link 
              href="/portfolio" 
              className="inline-flex items-center justify-center rounded-md border border-input bg-background px-8 py-3 text-sm font-medium shadow-sm transition-colors hover:bg-accent hover:text-accent-foreground"
            >
              View Our Work
            </Link>
          </div>
        </div>
      </Section>
      
      <Footer />
    </main>
  );
}