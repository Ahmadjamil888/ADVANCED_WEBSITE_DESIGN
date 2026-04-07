import Navbar from "./components/Navbar";
import Footer from "./components/Footer";
import Hero from "./sections/Hero";
import RealResults from "./sections/RealResults";
import Features from "./sections/Features";
import Offers from "./sections/Offers";
import EarlyStageChaos from "./sections/EarlyStageChaos";
import About from "./sections/About";
import Services from "./sections/Services";
import Insights from "./sections/Insights";
import Portfolio from "./sections/Portfolio";
import Team from "./sections/Team";
import Testimonials from "./sections/Testimonials";
import Contact from "./sections/Contact";

export default function Home() {
  return (
    <main className="relative min-h-screen bg-black">
      <Navbar />
      <Hero />
      <RealResults />
      <Features />
      <Offers />
      <EarlyStageChaos />
      <About />
      <Services />
      <Insights />
      <Portfolio />
      <Team />
      <Testimonials />
      <Contact />
      <Footer />
    </main>
  );
}
