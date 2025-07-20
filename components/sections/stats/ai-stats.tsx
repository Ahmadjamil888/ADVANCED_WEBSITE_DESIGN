import { Badge } from "../../ui/badge";
import { Section } from "../../ui/section";
import { siteConfig } from "@/config/site";
import { TrendingUp, Users, Brain, Award } from "lucide-react";

const stats = [
  {
    icon: <Brain className="size-6" />,
    value: siteConfig.stats.projects,
    label: "AI Projects Delivered",
    suffix: "+"
  },
  {
    icon: <Users className="size-6" />,
    value: siteConfig.stats.clients,
    label: "Happy Clients",
    suffix: "+"
  },
  {
    icon: <TrendingUp className="size-6" />,
    value: siteConfig.stats.aiModels,
    label: "ML Models Deployed",
    suffix: "+"
  },
  {
    icon: <Award className="size-6" />,
    value: siteConfig.stats.satisfaction,
    label: "Client Satisfaction",
    suffix: ""
  }
];

const additionalStats = [
  {
    label: "Years of Experience",
    value: siteConfig.stats.experience
  },
  {
    label: "Technologies Mastered",
    value: `${siteConfig.stats.technologies}+`
  },
  {
    label: "Industries Served",
    value: `${siteConfig.stats.industries}+`
  },
  {
    label: "Countries Reached",
    value: `${siteConfig.stats.countries}+`
  }
];

export default function AIStats() {
  return (
    <Section className="py-24 bg-muted/30">
      <div className="max-w-container mx-auto">
        <div className="text-center mb-16">
          <Badge variant="outline" className="mb-4">
            <TrendingUp className="mr-2 size-4" />
            Our Impact
          </Badge>
          <h2 className="text-3xl font-bold mb-4">
            Trusted by Businesses Worldwide
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Our track record speaks for itself. We've helped businesses across the globe 
            harness the power of AI and modern web technologies.
          </p>
        </div>

        {/* Main Stats Grid */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-8 mb-16">
          {stats.map((stat, index) => (
            <div
              key={index}
              className="text-center p-6 rounded-lg bg-background border hover:shadow-lg transition-shadow"
            >
              <div className="flex justify-center mb-4">
                <div className="p-3 rounded-full bg-primary/10 text-primary">
                  {stat.icon}
                </div>
              </div>
              <div className="text-3xl font-bold mb-2">
                {stat.value}{stat.suffix}
              </div>
              <div className="text-muted-foreground text-sm">
                {stat.label}
              </div>
            </div>
          ))}
        </div>

        {/* Additional Stats */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-6">
          {additionalStats.map((stat, index) => (
            <div key={index} className="text-center">
              <div className="text-2xl font-semibold mb-1">
                {stat.value}
              </div>
              <div className="text-muted-foreground text-sm">
                {stat.label}
              </div>
            </div>
          ))}
        </div>
      </div>
    </Section>
  );
}