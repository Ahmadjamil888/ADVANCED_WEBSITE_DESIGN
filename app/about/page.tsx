import Navbar from "../components/Navbar";
import Footer from "../components/Footer";
import About from "../sections/About";

export const metadata = {
  title: "About | zehanx Technologies",
  description: "Learn about zehanx Technologies. 6+ years of delivering excellence in Web Development, AI, ML, Deep Learning, Neural Networks, Software and App Development.",
};

export default function AboutPage() {
  return (
    <main className="relative min-h-screen bg-black pt-20">
      <Navbar />
      <About />
      <Footer />
    </main>
  );
}
