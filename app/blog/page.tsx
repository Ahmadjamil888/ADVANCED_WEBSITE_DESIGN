import { Badge } from "@/components/ui/badge";
import { Section } from "@/components/ui/section";
import { getAllBlogPosts } from "@/lib/blog";
import Link from "next/link";
import { Calendar, Clock, User, ArrowRight, BrainCircuit, Code, TrendingUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import Navbar from "@/components/sections/navbar/default";
import Footer from "@/components/sections/footer/default";

export const metadata = {
  title: 'Blog - AI & Web Development Insights | Zehan X Technologies',
  description: 'Stay updated with the latest trends in AI, machine learning, and web development. Expert insights, tutorials, and industry analysis.',
};

const getIconForTag = (tags: string[]) => {
  if (tags.includes('AI Trends') || tags.includes('Machine Learning')) return <BrainCircuit className="size-5" />;
  if (tags.includes('Deep Learning') || tags.includes('Neural Networks')) return <BrainCircuit className="size-5" />;
  if (tags.includes('Business Strategy') || tags.includes('Analytics')) return <TrendingUp className="size-5" />;
  return <Code className="size-5" />;
};

export default function Blog() {
  const blogPosts = getAllBlogPosts();
  const featuredPosts = blogPosts.slice(0, 2); // First 2 posts as featured
  const regularPosts = blogPosts.slice(2); // Rest as regular posts
  
  // Extract unique tags for categories
  const allTags = Array.from(new Set(blogPosts.flatMap(post => post.tags)));
  const categories = ["All", ...allTags];

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
              <Link key={index} href={`/blog/${post.slug}`}>
                <article className="group cursor-pointer enhanced-card p-6 rounded-lg hover-lift hover-glow">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="size-10 bg-blue-500/10 rounded-lg flex items-center justify-center text-blue-400">
                      {getIconForTag(post.tags)}
                    </div>
                    <div>
                      <span className="text-xs font-medium text-blue-400 bg-blue-500/10 px-2 py-1 rounded">
                        {post.tags[0] || 'AI'}
                      </span>
                    </div>
                  </div>
                  
                  <h3 className="text-xl font-semibold mb-3 group-hover:text-gradient-blue transition-colors">
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
                      <span className="flex items-center gap-1">
                        <User className="size-4" />
                        {post.author.split(',')[0]}
                      </span>
                    </div>
                    <ArrowRight className="size-4 group-hover:translate-x-1 transition-transform" />
                  </div>
                </article>
              </Link>
            ))}
          </div>

          {/* Regular Posts */}
          <h2 className="text-2xl font-bold mb-8">Latest Articles</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {regularPosts.map((post, index) => (
              <Link key={index} href={`/blog/${post.slug}`}>
                <article className="group cursor-pointer enhanced-card p-6 rounded-lg hover-lift">
                  <div className="flex items-center gap-2 mb-3">
                    <div className="size-8 bg-green-500/10 rounded-lg flex items-center justify-center text-green-400">
                      {getIconForTag(post.tags)}
                    </div>
                    <span className="text-xs font-medium text-green-400 bg-green-500/10 px-2 py-1 rounded">
                      {post.tags[0] || 'AI'}
                    </span>
                  </div>
                  
                  <h3 className="font-semibold mb-2 group-hover:text-gradient-green transition-colors line-clamp-2">
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
                </article>
              </Link>
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