import { Badge } from "@/components/ui/badge";
import { Section } from "@/components/ui/section";
import { Calendar, Clock, ArrowRight, BrainCircuit, Code, TrendingUp } from "lucide-react";
import Link from "next/link";
import Navbar from "@/components/sections/navbar/default";
import Footer from "@/components/sections/footer/default";

export const metadata = {
  title: 'Blog - AI & Web Development Insights | Zehan X Technologies',
  description: 'Stay updated with the latest trends in AI, machine learning, and web development. Expert insights, tutorials, and industry analysis.',
};

const blogPosts = [
  {
    title: "The Future of AI in Web Development: Trends for 2025",
    excerpt: "Explore how artificial intelligence is revolutionizing web development, from automated code generation to intelligent user experiences.",
    category: "AI Trends",
    date: "2025-01-15",
    readTime: "8 min read",
    icon: <BrainCircuit className="size-5" />,
    featured: true
  },
  {
    title: "Building Scalable Next.js Applications with AI Integration",
    excerpt: "Learn best practices for integrating AI capabilities into Next.js applications while maintaining performance and scalability.",
    category: "Web Development",
    date: "2025-01-10",
    readTime: "12 min read",
    icon: <Code className="size-5" />,
    featured: true
  },
  {
    title: "Machine Learning Model Deployment: From Development to Production",
    excerpt: "A comprehensive guide to deploying ML models in production environments with Docker, Kubernetes, and cloud platforms.",
    category: "MLOps",
    date: "2025-01-05",
    readTime: "15 min read",
    icon: <TrendingUp className="size-5" />,
    featured: false
  },
  {
    title: "Deep Learning for Computer Vision: Real-World Applications",
    excerpt: "Discover how deep learning is transforming industries through computer vision applications in healthcare, manufacturing, and retail.",
    category: "Deep Learning",
    date: "2024-12-28",
    readTime: "10 min read",
    icon: <BrainCircuit className="size-5" />,
    featured: false
  },
  {
    title: "Optimizing React Performance with AI-Powered Code Analysis",
    excerpt: "Learn how AI tools can help identify performance bottlenecks and optimize React applications for better user experience.",
    category: "Performance",
    date: "2024-12-20",
    readTime: "7 min read",
    icon: <Code className="size-5" />,
    featured: false
  },
  {
    title: "The Rise of AI Chatbots: Building Intelligent Conversational Interfaces",
    excerpt: "Explore the latest developments in AI chatbot technology and learn how to build intelligent conversational interfaces.",
    category: "AI Chatbots",
    date: "2024-12-15",
    readTime: "9 min read",
    icon: <BrainCircuit className="size-5" />,
    featured: false
  }
];

const categories = ["All", "AI Trends", "Web Development", "MLOps", "Deep Learning", "Performance", "AI Chatbots"];

export default function Blog() {
  const featuredPosts = blogPosts.filter(post => post.featured);
  const regularPosts = blogPosts.filter(post => !post.featured);

  return (
    <main className="min-h-screen w-full overflow-hidden bg-background text-foreground">
      <Navbar />
      
      {/* Hero Section */}
      <Section className="pt-24 pb-16">
        <div className="max-w-container mx-auto text-center">
          <Badge variant="outline" className="mb-6">
            <BrainCircuit className="mr-2 size-4" />
            Tech Blog
          </Badge>
          <h1 className="text-4xl font-bold mb-6 sm:text-6xl">
            AI & Web Development Insights
          </h1>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Stay updated with the latest trends, tutorials, and expert insights in artificial intelligence, 
            machine learning, and modern web development.
          </p>
        </div>
      </Section>

      {/* Categories */}
      <Section className="py-8 border-b">
        <div className="max-w-container mx-auto">
          <div className="flex flex-wrap justify-center gap-2">
            {categories.map((category, index) => (
              <button
                key={index}
                className={`px-4 py-2 rounded-full text-sm transition-colors ${
                  index === 0 
                    ? 'bg-primary text-primary-foreground' 
                    : 'bg-muted hover:bg-muted/80'
                }`}
              >
                {category}
              </button>
            ))}
          </div>
        </div>
      </Section>

      {/* Featured Posts */}
      <Section className="py-16">
        <div className="max-w-container mx-auto">
          <h2 className="text-2xl font-bold mb-8">Featured Articles</h2>
          <div className="grid md:grid-cols-2 gap-8 mb-16">
            {featuredPosts.map((post, index) => (
              <article key={index} className="group cursor-pointer">
                <div className="p-6 rounded-lg border bg-background hover:shadow-lg transition-all duration-300">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="size-10 bg-primary/10 rounded-lg flex items-center justify-center text-primary">
                      {post.icon}
                    </div>
                    <div>
                      <span className="text-xs font-medium text-primary bg-primary/10 px-2 py-1 rounded">
                        {post.category}
                      </span>
                    </div>
                  </div>
                  
                  <h3 className="text-xl font-semibold mb-3 group-hover:text-primary transition-colors">
                    {post.title}
                  </h3>
                  
                  <p className="text-muted-foreground mb-4 leading-relaxed">
                    {post.excerpt}
                  </p>
                  
                  <div className="flex items-center justify-between text-sm text-muted-foreground">
                    <div className="flex items-center gap-4">
                      <span className="flex items-center gap-1">
                        <Calendar className="size-4" />
                        {new Date(post.date).toLocaleDateString('en-US', { 
                          month: 'short', 
                          day: 'numeric', 
                          year: 'numeric' 
                        })}
                      </span>
                      <span className="flex items-center gap-1">
                        <Clock className="size-4" />
                        {post.readTime}
                      </span>
                    </div>
                    <ArrowRight className="size-4 group-hover:translate-x-1 transition-transform" />
                  </div>
                </div>
              </article>
            ))}
          </div>

          {/* Regular Posts */}
          <h2 className="text-2xl font-bold mb-8">Latest Articles</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {regularPosts.map((post, index) => (
              <article key={index} className="group cursor-pointer">
                <div className="p-6 rounded-lg border bg-background hover:shadow-lg transition-all duration-300">
                  <div className="flex items-center gap-2 mb-3">
                    <div className="size-8 bg-primary/10 rounded-lg flex items-center justify-center text-primary">
                      {post.icon}
                    </div>
                    <span className="text-xs font-medium text-primary bg-primary/10 px-2 py-1 rounded">
                      {post.category}
                    </span>
                  </div>
                  
                  <h3 className="font-semibold mb-2 group-hover:text-primary transition-colors line-clamp-2">
                    {post.title}
                  </h3>
                  
                  <p className="text-muted-foreground text-sm mb-4 leading-relaxed line-clamp-3">
                    {post.excerpt}
                  </p>
                  
                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                    <div className="flex items-center gap-3">
                      <span className="flex items-center gap-1">
                        <Calendar className="size-3" />
                        {new Date(post.date).toLocaleDateString('en-US', { 
                          month: 'short', 
                          day: 'numeric'
                        })}
                      </span>
                      <span className="flex items-center gap-1">
                        <Clock className="size-3" />
                        {post.readTime}
                      </span>
                    </div>
                    <ArrowRight className="size-3 group-hover:translate-x-1 transition-transform" />
                  </div>
                </div>
              </article>
            ))}
          </div>
        </div>
      </Section>

      {/* Newsletter CTA */}
      <Section className="py-16 bg-muted/30">
        <div className="max-w-container mx-auto text-center">
          <h2 className="text-3xl font-bold mb-6">
            Stay Updated with AI & Tech Insights
          </h2>
          <p className="text-muted-foreground mb-8 max-w-2xl mx-auto">
            Subscribe to our newsletter and get the latest articles, tutorials, and industry insights 
            delivered directly to your inbox.
          </p>
          <div className="flex justify-center gap-4">
            <Link 
              href="/contact" 
              className="inline-flex items-center justify-center rounded-md bg-primary px-8 py-3 text-sm font-medium text-primary-foreground shadow transition-colors hover:bg-primary/90"
            >
              Subscribe Now
            </Link>
            <Link 
              href="/services" 
              className="inline-flex items-center justify-center rounded-md border border-input bg-background px-8 py-3 text-sm font-medium shadow-sm transition-colors hover:bg-accent hover:text-accent-foreground"
            >
              Our Services
            </Link>
          </div>
        </div>
      </Section>
      
      <Footer />
    </main>
  );
}