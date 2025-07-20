import { BrainCircuit, Code, Database, Zap, Globe, Bot } from "lucide-react";
import { Badge } from "../../ui/badge";
import { Section } from "../../ui/section";

const services = [
  {
    icon: <BrainCircuit className="size-8" />,
    title: "AI & Machine Learning",
    description: "Custom AI solutions, predictive analytics, and intelligent automation to transform your business processes.",
    features: ["Custom ML Models", "Predictive Analytics", "Computer Vision", "NLP Solutions"]
  },
  {
    icon: <Code className="size-8" />,
    title: "Next.js Development",
    description: "Modern, fast, and scalable web applications built with Next.js and the latest React technologies.",
    features: ["Full-Stack Apps", "Server-Side Rendering", "API Development", "Performance Optimization"]
  },
  {
    icon: <Database className="size-8" />,
    title: "Deep Learning",
    description: "Advanced neural networks and deep learning models for complex pattern recognition and decision making.",
    features: ["Neural Networks", "Image Recognition", "Speech Processing", "Recommendation Systems"]
  },
  {
    icon: <Globe className="size-8" />,
    title: "Web Development",
    description: "Complete web solutions from frontend to backend, optimized for performance and user experience.",
    features: ["Responsive Design", "Progressive Web Apps", "E-commerce Solutions", "CMS Development"]
  },
  {
    icon: <Bot className="size-8" />,
    title: "AI Chatbots",
    description: "Intelligent conversational AI systems that enhance customer engagement and automate support.",
    features: ["Custom Chatbots", "Voice Assistants", "Customer Support AI", "Integration Services"]
  },
  {
    icon: <Zap className="size-8" />,
    title: "AI Consulting",
    description: "Strategic AI consulting to help you identify opportunities and implement AI solutions effectively.",
    features: ["AI Strategy", "Technology Assessment", "Implementation Planning", "Training & Support"]
  }
];

export default function AIServices() {
  return (
    <Section id="services" className="py-24">
      <div className="max-w-container mx-auto">
        <div className="text-center mb-16">
          <Badge variant="outline" className="mb-4">
            <Zap className="mr-2 size-4" />
            Our Services
          </Badge>
          <h2 className="text-3xl font-bold mb-4">
            Comprehensive AI & Web Development Solutions
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            From intelligent AI systems to modern web applications, we deliver 
            cutting-edge solutions that drive innovation and business growth.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {services.map((service, index) => (
            <div
              key={index}
              className="group relative overflow-hidden rounded-lg border bg-background p-6 hover:shadow-lg transition-all duration-300"
            >
              <div className="flex items-center gap-4 mb-4">
                <div className="flex-shrink-0 p-2 rounded-lg bg-primary/10 text-primary group-hover:bg-primary group-hover:text-primary-foreground transition-colors">
                  {service.icon}
                </div>
                <h3 className="text-xl font-semibold">{service.title}</h3>
              </div>
              
              <p className="text-muted-foreground mb-4">
                {service.description}
              </p>
              
              <ul className="space-y-2">
                {service.features.map((feature, featureIndex) => (
                  <li key={featureIndex} className="flex items-center gap-2 text-sm">
                    <div className="size-1.5 rounded-full bg-primary flex-shrink-0" />
                    {feature}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>
    </Section>
  );
}