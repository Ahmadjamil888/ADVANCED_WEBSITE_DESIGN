import { Badge } from "@/components/ui/badge";
import { Section } from "@/components/ui/section";
import { ExternalLink, Github, Zap, BrainCircuit, Code, Database } from "lucide-react";
import Link from "next/link";
import Navbar from "@/components/sections/navbar/default";
import Footer from "@/components/sections/footer/default";

export const metadata = {
  title: 'Portfolio - Our AI & Web Development Projects | Zehan X Technologies',
  description: 'Explore our portfolio of AI solutions, machine learning projects, and modern web applications. See how we transform businesses with cutting-edge technology.',
};

const projects = [
  {
    title: "AI-Powered E-commerce Platform",
    description: "Built a complete e-commerce solution with AI-driven product recommendations, inventory management, and customer behavior analytics.",
    technologies: ["Next.js", "TensorFlow", "Python", "PostgreSQL"],
    category: "AI & Web Development",
    icon: <BrainCircuit className="size-6" />,
    features: ["Personalized recommendations", "Real-time analytics", "Automated inventory", "Customer insights"]
  },
  {
    title: "Deep Learning Image Recognition System",
    description: "Developed a computer vision system for automated quality control in manufacturing, achieving 99.2% accuracy.",
    technologies: ["PyTorch", "OpenCV", "FastAPI", "Docker"],
    category: "Deep Learning",
    icon: <Zap className="size-6" />,
    features: ["Real-time processing", "99.2% accuracy", "Scalable architecture", "Quality control automation"]
  },
  {
    title: "Enterprise AI Chatbot Platform",
    description: "Created an intelligent chatbot system for customer support with natural language processing and multi-language support.",
    technologies: ["React", "Node.js", "OpenAI API", "MongoDB"],
    category: "AI Chatbots",
    icon: <Code className="size-6" />,
    features: ["Multi-language support", "Context awareness", "Integration ready", "Analytics dashboard"]
  },
  {
    title: "Predictive Analytics Dashboard",
    description: "Built a comprehensive analytics platform for business intelligence with machine learning-powered forecasting.",
    technologies: ["Next.js", "Python", "Scikit-learn", "PostgreSQL"],
    category: "Data Analytics",
    icon: <Database className="size-6" />,
    features: ["Predictive modeling", "Real-time dashboards", "Custom reports", "Data visualization"]
  },
  {
    title: "SaaS Application with AI Features",
    description: "Developed a full-stack SaaS platform with AI-powered automation, user management, and subscription billing.",
    technologies: ["Next.js", "TypeScript", "Stripe", "Prisma"],
    category: "Full-Stack Development",
    icon: <Code className="size-6" />,
    features: ["Subscription billing", "AI automation", "User management", "API integrations"]
  },
  {
    title: "Machine Learning Model Deployment",
    description: "Deployed and scaled ML models for real-time predictions with high availability and performance optimization.",
    technologies: ["Docker", "Kubernetes", "TensorFlow Serving", "AWS"],
    category: "MLOps",
    icon: <BrainCircuit className="size-6" />,
    features: ["Auto-scaling", "High availability", "Model versioning", "Performance monitoring"]
  }
];

export default function Portfolio() {
  return (
    <main className="min-h-screen w-full overflow-hidden bg-background text-foreground">
      <Navbar />
      
      {/* Hero Section */}
      <Section className="pt-24 pb-16">
        <div className="max-w-container mx-auto text-center">
          <Badge variant="outline" className="mb-6">
            <Code className="mr-2 size-4" />
            Our Portfolio
          </Badge>
          <h1 className="text-4xl font-bold mb-6 sm:text-6xl">
            Transforming Ideas into Reality
          </h1>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Explore our portfolio of AI solutions, machine learning projects, and modern web applications. 
            See how we've helped businesses leverage cutting-edge technology to achieve their goals.
          </p>
        </div>
      </Section>

      {/* Projects Grid */}
      <Section className="py-16">
        <div className="max-w-container mx-auto">
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {projects.map((project, index) => (
              <div key={index} className="group p-6 rounded-lg border bg-background hover:shadow-lg transition-all duration-300">
                <div className="flex items-center gap-3 mb-4">
                  <div className="size-12 bg-primary/10 rounded-lg flex items-center justify-center text-primary">
                    {project.icon}
                  </div>
                  <div>
                    <h3 className="font-semibold text-lg group-hover:text-primary transition-colors">
                      {project.title}
                    </h3>
                    <p className="text-sm text-muted-foreground">{project.category}</p>
                  </div>
                </div>
                
                <p className="text-muted-foreground mb-4 text-sm leading-relaxed">
                  {project.description}
                </p>
                
                <div className="mb-4">
                  <h4 className="font-medium mb-2 text-sm">Key Features:</h4>
                  <ul className="grid grid-cols-2 gap-1 text-xs text-muted-foreground">
                    {project.features.map((feature, i) => (
                      <li key={i} className="flex items-center gap-1">
                        <div className="size-1 bg-primary rounded-full"></div>
                        {feature}
                      </li>
                    ))}
                  </ul>
                </div>
                
                <div className="mb-4">
                  <h4 className="font-medium mb-2 text-sm">Technologies:</h4>
                  <div className="flex flex-wrap gap-1">
                    {project.technologies.map((tech, i) => (
                      <span key={i} className="px-2 py-1 bg-muted rounded text-xs">
                        {tech}
                      </span>
                    ))}
                  </div>
                </div>
                
                <div className="flex gap-2 pt-4 border-t">
                  <button className="flex items-center gap-1 text-xs text-muted-foreground hover:text-primary transition-colors">
                    <ExternalLink className="size-3" />
                    View Details
                  </button>
                  <button className="flex items-center gap-1 text-xs text-muted-foreground hover:text-primary transition-colors">
                    <Github className="size-3" />
                    Case Study
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </Section>

      {/* CTA Section */}
      <Section className="py-16 bg-muted/30">
        <div className="max-w-container mx-auto text-center">
          <h2 className="text-3xl font-bold mb-6">
            Ready to Start Your Next Project?
          </h2>
          <p className="text-muted-foreground mb-8 max-w-2xl mx-auto">
            Let's discuss how we can help you build innovative AI solutions and modern web applications 
            that drive your business forward.
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
              View Services
            </Link>
          </div>
        </div>
      </Section>
      
      <Footer />
    </main>
  );
}