import Navbar from "../components/Navbar";
import { Footer } from "@/components";
import Features from "../sections/Features";

export const metadata = {
  title: "Features | zehanx Technologies",
  description: "Explore zehanx Technologies services - Web Development, AI, Machine Learning, Deep Learning, Neural Networks, Software Development, and App Development.",
};

export default function FeaturesPage() {
  return (
    <main className="relative min-h-screen bg-black pt-20">
      <Navbar />
      <Features />
      <Footer />
    </main>
  );
}
