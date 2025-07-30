import { getBlogPost, getBlogSlugs } from "@/lib/blog";
import { notFound } from "next/navigation";
import { Calendar, Clock, User, ArrowLeft, Share2, BookOpen } from "lucide-react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Section } from "@/components/ui/section";
import Navbar from "@/components/sections/navbar/default";
import Footer from "@/components/sections/footer/default";

interface BlogPostPageProps {
  params: {
    slug: string;
  };
}

export async function generateStaticParams() {
  const slugs = getBlogSlugs();
  return slugs.map((slug) => ({
    slug,
  }));
}

export async function generateMetadata({ params }: BlogPostPageProps) {
  const post = await getBlogPost(params.slug);
  
  if (!post) {
    return {
      title: 'Blog Post Not Found | Zehan X Technologies',
    };
  }

  return {
    title: `${post.title} | Zehan X Technologies Blog`,
    description: post.excerpt,
    openGraph: {
      title: post.title,
      description: post.excerpt,
      type: 'article',
      publishedTime: post.date,
      authors: [post.author],
      tags: post.tags,
    },
  };
}

export default async function BlogPost({ params }: BlogPostPageProps) {
  const post = await getBlogPost(params.slug);

  if (!post) {
    notFound();
  }

  return (
    <main className="min-h-screen w-full overflow-hidden bg-background text-foreground">
      <Navbar />
      
      {/* Back Navigation */}
      <Section className="pt-24 pb-8">
        <div className="max-w-4xl mx-auto">
          <Link href="/blog">
            <Button variant="ghost" className="mb-8 nav-item">
              <ArrowLeft className="mr-2 size-4" />
              Back to Blog
            </Button>
          </Link>
        </div>
      </Section>

      {/* Article Header */}
      <Section className="pb-12">
        <div className="max-w-4xl mx-auto">
          <div className="mb-6">
            <div className="flex flex-wrap gap-2 mb-4">
              {post.tags.map((tag, index) => (
                <Badge 
                  key={index} 
                  variant="outline" 
                  className="border-blue-500/20 bg-blue-500/10 text-blue-400"
                >
                  {tag}
                </Badge>
              ))}
            </div>
            
            <h1 className="text-4xl font-bold mb-6 sm:text-5xl lg:text-6xl text-gradient-blue">
              {post.title}
            </h1>
            
            <p className="text-xl text-muted-foreground mb-8 leading-relaxed">
              {post.excerpt}
            </p>
            
            <div className="flex flex-wrap items-center gap-6 text-muted-foreground">
              <div className="flex items-center gap-2">
                <User className="size-4" />
                <span className="text-sm">{post.author}</span>
              </div>
              <div className="flex items-center gap-2">
                <Calendar className="size-4" />
                <span className="text-sm">
                  {new Date(post.date).toLocaleDateString('en-US', {
                    year: 'numeric',
                    month: 'long',
                    day: 'numeric',
                  })}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <Clock className="size-4" />
                <span className="text-sm">{post.readTime}</span>
              </div>
              <div className="flex items-center gap-2">
                <BookOpen className="size-4" />
                <span className="text-sm">Article</span>
              </div>
            </div>
          </div>
        </div>
      </Section>

      {/* Article Content */}
      <Section className="pb-16">
        <div className="max-w-4xl mx-auto">
          <div className="enhanced-card p-8 rounded-lg">
            <div 
              className="prose prose-lg prose-slate dark:prose-invert max-w-none
                prose-headings:text-gradient-blue prose-headings:font-bold
                prose-h1:text-4xl prose-h2:text-3xl prose-h3:text-2xl
                prose-p:leading-relaxed prose-p:text-foreground/90
                prose-a:text-blue-400 prose-a:no-underline hover:prose-a:underline
                prose-strong:text-foreground prose-strong:font-semibold
                prose-code:bg-muted prose-code:px-2 prose-code:py-1 prose-code:rounded
                prose-pre:bg-muted prose-pre:border prose-pre:border-border
                prose-blockquote:border-l-blue-500 prose-blockquote:bg-blue-500/5
                prose-ul:text-foreground/90 prose-ol:text-foreground/90
                prose-li:text-foreground/90"
              dangerouslySetInnerHTML={{ __html: post.content }}
            />
          </div>
        </div>
      </Section>

      {/* Share Section */}
      <Section className="py-12 bg-muted/30">
        <div className="max-w-4xl mx-auto text-center">
          <h3 className="text-2xl font-bold mb-4">Share this article</h3>
          <p className="text-muted-foreground mb-6">
            Found this helpful? Share it with your network!
          </p>
          <div className="flex justify-center gap-4">
            <Button variant="outline" className="hover-lift">
              <Share2 className="mr-2 size-4" />
              Share
            </Button>
            <Link href="/blog">
              <Button className="btn-gradient-primary hover-lift">
                Read More Articles
              </Button>
            </Link>
          </div>
        </div>
      </Section>

      {/* CTA Section */}
      <Section className="py-16">
        <div className="max-w-4xl mx-auto text-center">
          <h3 className="text-3xl font-bold mb-6 text-gradient-green">
            Ready to Transform Your Business with AI?
          </h3>
          <p className="text-muted-foreground mb-8 max-w-2xl mx-auto">
            Our expert team at Zehan X Technologies can help you implement the AI solutions 
            discussed in this article. Let's build something amazing together.
          </p>
          <div className="flex justify-center gap-4">
            <Link href="/contact">
              <Button className="btn-gradient-secondary hover-lift glow-green">
                Get Started Today
              </Button>
            </Link>
            <Link href="/services">
              <Button variant="outline" className="hover-lift">
                Our Services
              </Button>
            </Link>
          </div>
        </div>
      </Section>
      
      <Footer />
    </main>
  );
}