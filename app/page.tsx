import Navbar from "./components/Navbar";
import Footer from "./components/Footer";
import Hero from "./sections/Hero";
import Features from "./sections/Features";
import About from "./sections/About";
import Services from "./sections/Services";
import Insights from "./sections/Insights";
import Portfolio from "./sections/Portfolio";
import Team from "./sections/Team";
import Contact from "./sections/Contact";

export default function Home() {
  return (
    <main className="relative min-h-screen bg-black">
      <Navbar />
      <Hero />
      <Features />
      <About />
      <Services />
      <Insights />
      <Portfolio />
      <Team />
      <Contact />
      <Footer />
    </main>
  );
}
