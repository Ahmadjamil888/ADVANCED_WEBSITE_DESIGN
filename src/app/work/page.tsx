import { Column, Heading } from "@/once-ui/components";
import { baseURL } from "@/app/resources";
import { about, person, work } from "@/app/resources/content";
import { CustomCard } from "@/components/CustomCard";

export const metadata = {
  title: 'Portfolio - AI Systems & Development Projects | zehanxtech',
  description: 'Explore zehanxtech\'s portfolio of AI systems including Byte AI, UAP Universal Agentic Platform, Pharma CoPilot, Zehan AI, and Pakistan\'s first AI assistant.',
  keywords: [
    'AI portfolio',
    'machine learning projects',
    'Byte AI',
    'UAP platform',
    'Pharma CoPilot',
    'Zehan AI',
    'Pakistan AI',
    'AI app builder',
    'artificial intelligence projects',
    'zehanxtech projects'
  ],
  openGraph: {
    title: 'Portfolio - AI Systems & Development Projects | zehanxtech',
    description: 'Discover our innovative AI systems and applications that are transforming industries.',
    url: `${baseURL}${work.path}`,
    images: [
      {
        url: '/og-portfolio.jpg',
        width: 1200,
        height: 630,
        alt: 'zehanxtech AI Portfolio Projects',
      },
    ],
  },
  alternates: {
    canonical: `${baseURL}${work.path}`,
  },
};

const featuredProjects = [
  {
    id: 1,
    title: "Byte AI",
    description: "Advanced AI app builder that enables users to create intelligent applications without coding. Features drag-and-drop interface and real-time deployment.",
    category: "AI Development",
    tags: ["React", "AI", "No-Code"],
    url: "https://github.com/Ahmadjamil888/byte-ai"
  },
  {
    id: 2,
    title: "UAP - Universal Agentic Platform",
    description: "A comprehensive platform for building and managing AI agents. Supports multi-agent workflows and seamless integration with various APIs.",
    category: "AI Platform",
    tags: ["Next.js", "AI Agents", "TypeScript"],
    url: "https://github.com/Ahmadjamil888/uap"
  },
  {
    id: 3,
    title: "Pharma CoPilot",
    description: "AI-powered healthcare assistant for pharmaceutical research and drug discovery. Provides intelligent insights and data analysis.",
    category: "Healthcare AI",
    tags: ["Python", "Healthcare", "Machine Learning"],
    url: "https://github.com/Ahmadjamil888/pharma-copilot"
  },
  {
    id: 4,
    title: "Zehan AI",
    description: "Pakistan's first advanced AI assistant with natural language processing capabilities. Supports multiple languages and intelligent conversations.",
    category: "AI Assistant",
    tags: ["Python", "NLP", "AI"],
    url: "https://github.com/Ahmadjamil888/zehan-ai"
  },
  {
    id: 5,
    title: "AI App Builder",
    description: "Revolutionary platform for creating AI-powered applications with visual programming. Includes pre-built AI models and deployment tools.",
    category: "Development Platform",
    tags: ["React", "AI", "Visual Programming"],
    url: "https://github.com/Ahmadjamil888/ai-app-builder"
  },
  {
    id: 6,
    title: "Pakistan's First AI",
    description: "Groundbreaking AI system designed specifically for Pakistani context. Features Urdu language support and local cultural understanding.",
    category: "AI Innovation",
    tags: ["Python", "Urdu NLP", "Cultural AI"],
    url: "https://github.com/Ahmadjamil888/pakistan-first-ai"
  }
];

export default function Work() {
  return (
    <Column maxWidth="m">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify({
            "@context": "https://schema.org",
            "@type": "CollectionPage",
            "name": "zehanxtech Portfolio - AI Systems & Development Projects",
            "description": "Explore our portfolio of AI systems, machine learning projects, and innovative applications including Byte AI, UAP, Pharma CoPilot, and Pakistan's first AI.",
            "url": `${baseURL}${work.path}`,
            "mainEntity": {
              "@type": "ItemList",
              "name": "AI Development Projects",
              "itemListElement": featuredProjects.map((project, index) => ({
                "@type": "SoftwareApplication",
                "position": index + 1,
                "name": project.title,
                "description": project.description,
                "applicationCategory": project.category,
                "operatingSystem": "Web",
                "url": project.url,
                "author": {
                  "@type": "Organization",
                  "name": "zehanxtech"
                }
              }))
            },
            "breadcrumb": {
              "@type": "BreadcrumbList",
              "itemListElement": [
                {
                  "@type": "ListItem",
                  "position": 1,
                  "name": "Home",
                  "item": baseURL
                },
                {
                  "@type": "ListItem",
                  "position": 2,
                  "name": "Portfolio",
                  "item": `${baseURL}${work.path}`
                }
              ]
            }
          })
        }}
      />
      <Heading variant="display-strong-s" style={{ marginBottom: 24 }}>
        zehanxtech Portfolio - AI Systems & Development Projects
      </Heading>
      <div style={{ 
        display: "flex", 
        flexWrap: "wrap", 
        gap: "24px", 
        justifyContent: "center",
        padding: "20px 0"
      }}>
        {featuredProjects.map(project => (
          <div 
            key={project.id}
            style={{
              width: '320px',
              height: '350px',
              padding: '20px',
              color: 'white',
              background: 'linear-gradient(#212121, #212121) padding-box, linear-gradient(145deg, transparent 35%, #2563eb, #3b82f6) border-box',
              border: '2px solid transparent',
              borderRadius: '8px',
              display: 'flex',
              flexDirection: 'column',
              boxShadow: '0 4px 16px rgba(0,0,0,0.1)'
            }}
          >
            <div style={{ flex: 1 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '16px' }}>
                <span style={{ fontWeight: '600', color: '#717171', marginRight: '4px' }}>
                  {project.category}
                </span>
              </div>
              
              <h3 style={{ 
                fontSize: '24px', 
                margin: '24px 0 16px', 
                fontWeight: '600',
                lineHeight: '1.2',
                color: '#ffffff'
              }}>
                {project.title}
              </h3>
              
              <p style={{ 
                color: '#a1a1aa', 
                fontSize: '14px', 
                lineHeight: '1.5',
                marginBottom: '16px',
                minHeight: '60px'
              }}>
                {project.description}
              </p>
              
              {project.tags.length > 0 && (
                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: '16px' }}>
                  {project.tags.slice(0, 3).map((tag, index) => (
                    <span 
                      key={index}
                      style={{
                        backgroundColor: '#2563eb',
                        padding: '4px 8px',
                        fontWeight: '600',
                        textTransform: 'uppercase',
                        fontSize: '10px',
                        borderRadius: '50em'
                      }}
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              )}
            </div>
            
            <div style={{ 
              fontWeight: '600', 
              color: '#717171', 
              marginTop: '16px',
              paddingTop: '16px',
              borderTop: '1px solid #374151'
            }}>
              <a 
                href={project.url} 
                target="_blank" 
                rel="noopener noreferrer"
                style={{ color: '#2563eb', textDecoration: 'none' }}
              >
                View Project →
              </a>
            </div>
          </div>
        ))}
      </div>
    </Column>
  );
}