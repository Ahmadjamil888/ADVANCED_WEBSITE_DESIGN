<<<<<<< HEAD
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
=======
"use client";
import { useEffect } from "react";
import Lenis from "@studio-freight/lenis";
import {
	CTA,
	Footer,
	Hero,
	LogoTicker,
	Services,
	ProductShowcase,
	Testimonials,
} from "@/components";

export default function App() {
	useEffect(() => {
		const lenis = new Lenis();

		function raf(time: number) {
			lenis.raf(time);
			requestAnimationFrame(raf);
		}

		requestAnimationFrame(raf);
	}, []);
	return (
		<>
			<Hero />
			<LogoTicker />
			<ProductShowcase />
			<Services />
			<Testimonials />
			<CTA />
			<Footer />
		</>
	);
>>>>>>> 3bc9588be4435e479cd8b5adde3400babe24a484
}
