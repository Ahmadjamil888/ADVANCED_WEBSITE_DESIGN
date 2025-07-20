import { Badge } from "../../ui/badge";
import { Section } from "../../ui/section";
import { Zap } from "lucide-react";

const technologies = [
  { name: "Next.js", logo: "⚡" },
  { name: "React", logo: "⚛️" },
  { name: "TensorFlow", logo: "🧠" },
  { name: "PyTorch", logo: "🔥" },
  { name: "OpenAI", logo: "🤖" },
  { name: "Python", logo: "🐍" },
  { name: "TypeScript", logo: "📘" },
  { name: "Node.js", logo: "🟢" },
  { name: "AWS", logo: "☁️" },
  { name: "Docker", logo: "🐳" },
  { name: "Kubernetes", logo: "⚙️" },
  { name: "MongoDB", logo: "🍃" }
];

export default function TechLogos() {
  return (
    <Section className="py-16 bg-muted/20">
      <div className="max-w-container mx-auto">
        <div className="text-center mb-12">
          <Badge variant="outline" className="mb-4">
            <Zap className="mr-2 size-4" />
            Technologies We Master
          </Badge>
          <h2 className="text-2xl font-bold mb-4">
            Built with Industry-Leading Technologies
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            We leverage the most advanced and reliable technologies to deliver 
            cutting-edge AI and web solutions.
          </p>
        </div>

        <div className="grid grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-8 items-center">
          {technologies.map((tech, index) => (
            <div
              key={index}
              className="flex flex-col items-center gap-3 p-4 rounded-lg hover:bg-background transition-colors group"
            >
              <div className="text-3xl group-hover:scale-110 transition-transform">
                {tech.logo}
              </div>
              <span className="text-sm font-medium text-muted-foreground group-hover:text-foreground transition-colors">
                {tech.name}
              </span>
            </div>
          ))}
        </div>

        <div className="text-center mt-12">
          <p className="text-muted-foreground text-sm">
            And many more cutting-edge technologies to bring your vision to life
          </p>
        </div>
      </div>
    </Section>
  );
}