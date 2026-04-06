<<<<<<< HEAD
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
=======
"use client";

import { motion } from "framer-motion";
import { Navbar, Footer, Heading } from "@/components";
import { companyInfo, contactInfo } from "@/constants";
import { Code, Cpu, Globe, Users, Target, Rocket } from "lucide-react";

export default function AboutPage() {
	const stats = [
		{ icon: <Code className="w-6 h-6" />, value: "50+", label: "Projects Delivered" },
		{ icon: <Users className="w-6 h-6" />, value: "15+", label: "Team Members" },
		{ icon: <Globe className="w-6 h-6" />, value: "10+", label: "Countries Served" },
		{ icon: <Target className="w-6 h-6" />, value: "99%", label: "Client Satisfaction" },
	];

	const values = [
		{
			icon: <Rocket className="w-8 h-8 text-[#00ff88]" />,
			title: "Innovation First",
			description: "We stay at the cutting edge of technology, constantly exploring new solutions to give our clients a competitive advantage.",
		},
		{
			icon: <Code className="w-8 h-8 text-[#00ff88]" />,
			title: "Code Excellence",
			description: "Quality isn't negotiable. We write clean, maintainable, and scalable code that stands the test of time.",
		},
		{
			icon: <Users className="w-8 h-8 text-[#00ff88]" />,
			title: "Partnership Approach",
			description: "We don't just deliver projects—we build lasting relationships. Your success is our success.",
		},
		{
			icon: <Cpu className="w-8 h-8 text-[#00ff88]" />,
			title: "AI-Driven",
			description: "Machine Learning isn't just a service—it's in our DNA. We bring intelligence to every solution.",
		},
	];

	return (
		<div className="min-h-screen bg-[#0a0a0a]">
			<Navbar />
			<div className="pt-32 pb-20 padding-x">
				{/* Hero Section */}
				<div className="max-w-4xl mx-auto text-center mb-20">
					<motion.div
						initial={{ opacity: 0, y: 20 }}
						animate={{ opacity: 1, y: 0 }}
						transition={{ duration: 0.5 }}
						className="flex items-center justify-center gap-2 px-4 py-2 rounded-full border border-[#00ff88]/30 bg-[#00ff88]/10 w-fit mx-auto mb-6">
						<span className="text-[#00ff88] font-mono text-sm">&gt; who_we_are</span>
					</motion.div>
					<motion.h1
						initial={{ opacity: 0, y: 20 }}
						animate={{ opacity: 1, y: 0 }}
						transition={{ duration: 0.5, delay: 0.1 }}
						className="text-5xl md:text-6xl font-bold text-white mb-6">
						About <span className="text-[#00ff88]">{companyInfo.name}</span>
					</motion.h1>
					<motion.p
						initial={{ opacity: 0, y: 20 }}
						animate={{ opacity: 1, y: 0 }}
						transition={{ duration: 0.5, delay: 0.2 }}
						className="text-xl text-[#888] font-mono leading-relaxed">
						{companyInfo.description}
					</motion.p>
				</div>

				{/* Stats Section */}
				<motion.div
					initial={{ opacity: 0, y: 30 }}
					animate={{ opacity: 1, y: 0 }}
					transition={{ duration: 0.5, delay: 0.3 }}
					className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-20">
					{stats.map((stat, index) => (
						<div
							key={index}
							className="p-6 rounded-2xl bg-[#111] border border-[#222] text-center hover:border-[#00ff88]/30 transition-colors">
							<div className="text-[#00ff88] flex justify-center mb-3">{stat.icon}</div>
							<div className="text-3xl font-bold text-white mb-1">{stat.value}</div>
							<div className="text-sm text-[#888] font-mono">{stat.label}</div>
						</div>
					))}
				</motion.div>

				{/* Story Section */}
				<div className="grid md:grid-cols-2 gap-12 mb-20">
					<motion.div
						initial={{ opacity: 0, x: -30 }}
						animate={{ opacity: 1, x: 0 }}
						transition={{ duration: 0.5, delay: 0.4 }}>
						<h2 className="text-3xl font-bold text-white mb-4">Our Story</h2>
						<p className="text-[#888] font-mono leading-relaxed mb-4">
							Founded in {companyInfo.founded}, {companyInfo.name} began with a simple mission: 
							to make cutting-edge technology accessible to businesses of all sizes. 
							What started as a small team of passionate developers has grown into 
							a full-service technology partner.
						</p>
						<p className="text-[#888] font-mono leading-relaxed">
							Today, we&apos;re proud to have helped dozens of companies transform their 
							operations through intelligent software solutions. From predictive 
							analytics to full-scale digital transformation, we bring expertise 
							and dedication to every project.
						</p>
					</motion.div>
					<motion.div
						initial={{ opacity: 0, x: 30 }}
						animate={{ opacity: 1, x: 0 }}
						transition={{ duration: 0.5, delay: 0.5 }}
						className="p-8 rounded-2xl bg-gradient-to-br from-[#1a1a2e] to-[#0a0a0a] border border-[#222]">
						<h3 className="text-2xl font-bold text-white mb-4">Why Choose Us?</h3>
						<ul className="space-y-3">
							{[
								"Expert team with 10+ years combined experience",
								"End-to-end solutions from strategy to deployment",
								"Agile methodology for rapid iteration",
								"24/7 support and maintenance",
								"Transparent communication throughout",
							].map((item, i) => (
								<li key={i} className="flex items-start gap-3 text-[#aaa] font-mono">
									<span className="text-[#00ff88]">›</span>
									<span>{item}</span>
								</li>
							))}
						</ul>
					</motion.div>
				</div>

				{/* Values Section */}
				<div className="mb-20">
					<h2 className="text-3xl font-bold text-white text-center mb-12">Our Values</h2>
					<div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
						{values.map((value, index) => (
							<motion.div
								key={index}
								initial={{ opacity: 0, y: 20 }}
								animate={{ opacity: 1, y: 0 }}
								transition={{ duration: 0.5, delay: 0.1 * index }}
								className="p-6 rounded-2xl bg-[#111] border border-[#222] hover:border-[#00ff88]/30 transition-colors">
								<div className="mb-4">{value.icon}</div>
								<h3 className="text-xl font-bold text-white mb-2">{value.title}</h3>
								<p className="text-[#888] text-sm font-mono">{value.description}</p>
							</motion.div>
						))}
					</div>
				</div>

				{/* Contact CTA */}
				<motion.div
					initial={{ opacity: 0, y: 20 }}
					animate={{ opacity: 1, y: 0 }}
					transition={{ duration: 0.5, delay: 0.6 }}
					className="text-center p-12 rounded-3xl bg-gradient-to-r from-[#00ff88]/10 to-[#61afef]/10 border border-[#00ff88]/30">
					<h2 className="text-3xl font-bold text-white mb-4">Ready to Work Together?</h2>
					<p className="text-[#888] font-mono mb-6">
						Let&apos;s discuss how we can help transform your business.
					</p>
					<div className="flex flex-col sm:flex-row gap-4 justify-center">
						<a
							href={`mailto:${contactInfo.email}`}
							className="px-8 py-3 bg-[#00ff88] text-[#0a0a0a] font-mono font-semibold rounded-lg hover:bg-[#00cc6a] transition-colors">
							Email Us
						</a>
						<a
							href={`https://wa.me/${contactInfo.whatsapp}`}
							className="px-8 py-3 border border-[#00ff88] text-[#00ff88] font-mono font-semibold rounded-lg hover:bg-[#00ff88]/10 transition-colors">
							WhatsApp
						</a>
					</div>
				</motion.div>
			</div>
			<Footer />
		</div>
	);
>>>>>>> 3bc9588be4435e479cd8b5adde3400babe24a484
}
