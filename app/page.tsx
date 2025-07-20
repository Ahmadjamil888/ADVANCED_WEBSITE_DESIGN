import { Metadata } from "next";
import CTA from "../components/sections/cta/default";
import FAQ from "../components/sections/faq/default";
import Footer from "../components/sections/footer/default";
import Hero from "../components/sections/hero/default";
import Items from "../components/sections/items/default";
import Logos from "../components/sections/logos/default";
import Navbar from "../components/sections/navbar/default";
import Stats from "../components/sections/stats/default";

export const metadata: Metadata = {
  title: "Zehan X Technologies - Leading AI & Web Development Company",
  description: "Transform your business with cutting-edge AI solutions, machine learning models, and modern Next.js web applications. Expert AI consulting, deep learning, and full-stack development services.",
  keywords: [
    "AI development company",
    "machine learning services", 
    "Next.js development",
    "artificial intelligence solutions",
    "deep learning company",
    "AI consulting services",
    "custom AI models",
    "web development agency",
    "React development",
    "AI automation",
    "predictive analytics",
    "business intelligence",
    "AI chatbots",
    "computer vision",
    "natural language processing"
  ],
  openGraph: {
    title: "Zehan X Technologies - Leading AI & Web Development Company",
    description: "Transform your business with cutting-edge AI solutions, machine learning models, and modern Next.js web applications.",
    type: "website",
    url: "https://zehanx.com",
  },
  twitter: {
    title: "Zehan X Technologies - Leading AI & Web Development Company",
    description: "Transform your business with cutting-edge AI solutions, machine learning models, and modern Next.js web applications.",
    card: "summary_large_image",
  },
  alternates: {
    canonical: "https://zehanx.com",
  },
};

export default function Home() {
  return (
    <main className="min-h-screen w-full overflow-hidden bg-background text-foreground">
      <Navbar />
      <Hero />
      <Logos />
      <Items />
      <Stats />
      <FAQ />
      <CTA />
      <Footer />
    </main>
  );
}
