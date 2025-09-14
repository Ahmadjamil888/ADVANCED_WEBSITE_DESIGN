import { Badge } from "@/components/ui/badge";
import { Section } from "@/components/ui/section";
import { Mail, Clock, Users, CheckCircle, ArrowRight, Lightbulb, Code, Palette, Video, PenTool, BarChart3 } from "lucide-react";
import Link from "next/link";
import Navbar from "@/components/sections/navbar/default";
import Footer from "@/components/sections/footer/default";

export const metadata = {
  title: 'Custom Orders - AI & Web Development Services | Zehan X Technologies',
  description: 'Get personalized quotes for your AI solutions, web development, digital marketing, video editing, graphic design, and content writing projects.',
};

const services = [
  {
    name: "AI & Machine Learning",
    description: "Custom AI models, chatbots, and intelligent automation systems",
    icon: <Lightbulb className="size-6" />,
    color: "blue"
  },
  {
    name: "Web Development",
    description: "Modern websites and web applications with cutting-edge technologies",
    icon: <Code className="size-6" />,
    color: "green"
  },
  {
    name: "Digital Marketing",
    description: "Strategic campaigns that drive traffic, leads, and conversions",
    icon: <BarChart3 className="size-6" />,
    color: "purple"
  },
  {
    name: "Video Editing",
    description: "Professional video production and post-production services",
    icon: <Video className="size-6" />,
    color: "red"
  },
  {
    name: "Graphic Design",
    description: "Creative visual design for branding and marketing materials",
    icon: <Palette className="size-6" />,
    color: "yellow"
  },
  {
    name: "Content Writing",
    description: "Compelling content creation for websites, blogs, and marketing",
    icon: <PenTool className="size-6" />,
    color: "cyan"
  }
];

const processSteps = [
  {
    step: "1",
    title: "Initial Consultation",
    description: "We discuss your project requirements, goals, and vision in detail",
    icon: <Users className="size-6" />
  },
  {
    step: "2", 
    title: "Custom Proposal",
    description: "We create a detailed proposal tailored to your specific needs and budget",
    icon: <Mail className="size-6" />
  },
  {
    step: "3",
    title: "Project Execution",
    description: "Our team works closely with you to bring your vision to life",
    icon: <Code className="size-6" />
  },
  {
    step: "4",
    title: "Delivery & Support",
    description: "We deliver your project and provide ongoing support as needed",
    icon: <CheckCircle className="size-6" />
  }
];

export default function CustomOrders() {
  return (
    <main className="min-h-screen w-full overflow-hidden bg-background text-foreground">
      <Navbar />
      
      {/* Hero Section */}
      <Section className="pt-24 pb-16">
        <div className="max-w-container mx-auto text-center">
          <Badge variant="outline" className="mb-6">
            <Mail className="mr-2 size-4" />
            Custom Orders
          </Badge>
          <h1 className="text-4xl font-bold mb-6 sm:text-6xl">
            Personalized Solutions for Your Business
          </h1>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Every project is unique. Contact us directly to discuss your specific requirements 
            and get a personalized quote tailored to your needs and budget.
          </p>
        </div>
      </Section>

      {/* Services Grid */}
      <Section className="py-16">
        <div className="max-w-container mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Our Services</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              We offer comprehensive digital solutions that combine cutting-edge technology with creative excellence
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {services.map((service, index) => (
              <div 
                key={index} 
                className={`group p-8 rounded-lg border bg-background hover:shadow-lg transition-all duration-300 hover:transform hover:scale-105 hover:border-${service.color}-500/50`}
              >
                <div className={`w-16 h-16 bg-${service.color}-500/20 rounded-xl flex items-center justify-center mb-6 group-hover:bg-${service.color}-500/30 transition-colors text-${service.color}-400`}>
                  {service.icon}
                </div>
                <h3 className="text-xl font-semibold mb-4">{service.name}</h3>
                <p className="text-muted-foreground text-sm mb-6">{service.description}</p>
                <a 
                  href="mailto:shazabjamildhami@gmail.com" 
                  className={`text-${service.color}-400 hover:text-${service.color}-300 font-medium text-sm inline-flex items-center gap-2`}
                >
                  Get Quote
                  <ArrowRight className="size-4" />
                </a>
              </div>
            ))}
          </div>
        </div>
      </Section>

      {/* Process Section */}
      <Section className="py-16 bg-muted/30">
        <div className="max-w-container mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Our Process</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              We follow a proven process to ensure your project is delivered on time and exceeds expectations
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {processSteps.map((step, index) => (
              <div key={index} className="text-center">
                <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mx-auto mb-4 text-primary">
                  {step.icon}
                </div>
                <div className="w-8 h-8 bg-primary text-primary-foreground rounded-full flex items-center justify-center mx-auto mb-4 text-sm font-bold">
                  {step.step}
                </div>
                <h3 className="text-lg font-semibold mb-2">{step.title}</h3>
                <p className="text-muted-foreground text-sm">{step.description}</p>
              </div>
            ))}
          </div>
        </div>
      </Section>

      {/* Why Choose Us */}
      <Section className="py-16">
        <div className="max-w-container mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Why Choose Custom Orders?</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Every business is unique, and so should be your digital solutions
            </p>
          </div>
          
          <div className="grid md:grid-cols-3 gap-8">
            <div className="text-center p-6">
              <div className="w-16 h-16 bg-blue-500/10 rounded-full flex items-center justify-center mx-auto mb-4">
                <Users className="size-8 text-blue-400" />
              </div>
              <h3 className="text-xl font-semibold mb-4">Personalized Approach</h3>
              <p className="text-muted-foreground">
                Every project is tailored to your specific needs, goals, and budget requirements.
              </p>
            </div>
            
            <div className="text-center p-6">
              <div className="w-16 h-16 bg-green-500/10 rounded-full flex items-center justify-center mx-auto mb-4">
                <Clock className="size-8 text-green-400" />
              </div>
              <h3 className="text-xl font-semibold mb-4">Flexible Timeline</h3>
              <p className="text-muted-foreground">
                We work with your schedule and can adjust timelines based on your priorities.
              </p>
            </div>
            
            <div className="text-center p-6">
              <div className="w-16 h-16 bg-purple-500/10 rounded-full flex items-center justify-center mx-auto mb-4">
                <CheckCircle className="size-8 text-purple-400" />
              </div>
              <h3 className="text-xl font-semibold mb-4">Quality Guarantee</h3>
              <p className="text-muted-foreground">
                We ensure the highest quality standards and provide ongoing support for your project.
              </p>
            </div>
          </div>
        </div>
      </Section>

      {/* CTA Section */}
      <Section className="py-16 bg-muted/30">
        <div className="max-w-container mx-auto text-center">
          <h2 className="text-3xl font-bold mb-6">
            Ready to Start Your Custom Project?
          </h2>
          <p className="text-muted-foreground mb-8 max-w-2xl mx-auto">
            Get in touch with our team to discuss your project requirements, timeline, and budget. 
            We'll provide a detailed proposal tailored to your specific needs.
          </p>
          <div className="flex justify-center gap-4">
            <a 
              href="mailto:shazabjamildhami@gmail.com" 
              className="inline-flex items-center justify-center rounded-md bg-primary px-8 py-3 text-sm font-medium text-primary-foreground shadow transition-colors hover:bg-primary/90"
            >
              <Mail className="w-5 h-5 mr-2" />
              Get Custom Quote
            </a>
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