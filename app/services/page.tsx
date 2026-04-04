"use client";

import { motion } from "framer-motion";
import { Navbar, Footer, Services } from "@/components";
import { servicesItems, contactInfo } from "@/constants";
import { Brain, Code, Database, ArrowRight, Check } from "lucide-react";

const iconMap: { [key: string]: React.ReactNode } = {
	brain: <Brain className="w-12 h-12 text-[#00ff88]" />,
	code: <Code className="w-12 h-12 text-[#00ff88]" />,
	database: <Database className="w-12 h-12 text-[#00ff88]" />,
};

export default function ServicesPage() {
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
						<span className="text-[#00ff88] font-mono text-sm">&gt; our_services</span>
					</motion.div>
					<motion.h1
						initial={{ opacity: 0, y: 20 }}
						animate={{ opacity: 1, y: 0 }}
						transition={{ duration: 0.5, delay: 0.1 }}
						className="text-5xl md:text-6xl font-bold text-white mb-6">
						What We <span className="text-[#00ff88]">Offer</span>
					</motion.h1>
					<motion.p
						initial={{ opacity: 0, y: 20 }}
						animate={{ opacity: 1, y: 0 }}
						transition={{ duration: 0.5, delay: 0.2 }}
						className="text-xl text-[#888] font-mono leading-relaxed">
						From machine learning models to full-stack applications, 
						we deliver solutions that drive real business impact.
					</motion.p>
				</div>

				{/* Services Grid */}
				<div className="grid md:grid-cols-3 gap-8 mb-20">
					{servicesItems.map((service, index) => (
						<motion.div
							key={service.id}
							initial={{ opacity: 0, y: 30 }}
							animate={{ opacity: 1, y: 0 }}
							transition={{ duration: 0.5, delay: index * 0.1 }}
							className={`p-8 rounded-2xl border transition-all hover:border-[#00ff88]/50 ${
								service.popular
									? "bg-gradient-to-b from-[#1a1a2e] to-[#0a0a0a] border-[#00ff88]/30"
									: "bg-[#111] border-[#222]"
							}`}>
							<div className="p-4 rounded-xl bg-[#00ff88]/10 border border-[#00ff88]/20 w-fit mb-6">
								{iconMap[service.icon]}
							</div>
							{service.popular && (
								<span className="inline-block px-3 py-1 bg-[#00ff88]/10 text-[#00ff88] text-sm font-mono rounded-full mb-4">
									Most Popular
								</span>
							)}
							<h3 className="text-2xl font-bold text-white mb-2">{service.title}</h3>
							<p className="text-[#00ff88] font-mono text-sm mb-4">{service.subtitle}</p>
							<p className="text-[#888] font-mono mb-6">{service.description}</p>
							<ul className="space-y-2 mb-8">
								{service.features.map((feature, i) => (
									<li key={i} className="flex items-center gap-2 text-[#aaa] text-sm font-mono">
										<Check className="w-4 h-4 text-[#00ff88]" />
										{feature}
									</li>
								))}
							</ul>
							<a
								href={`mailto:${contactInfo.email}?subject=Inquiry: ${service.title}`}
								className={`w-full flex items-center justify-center gap-2 py-3 rounded-lg font-mono font-semibold transition-colors ${
									service.popular
										? "bg-[#00ff88] text-[#0a0a0a] hover:bg-[#00cc6a]"
										: "border border-[#333] text-white hover:border-[#00ff88] hover:text-[#00ff88]"
								}`}>
								Get Started
								<ArrowRight className="w-4 h-4" />
							</a>
						</motion.div>
					))}
				</div>

				{/* Process Section */}
				<div className="mb-20">
					<h2 className="text-3xl font-bold text-white text-center mb-12">How We Work</h2>
					<div className="grid md:grid-cols-4 gap-6">
						{[
							{ step: "01", title: "Discovery", desc: "Understanding your needs and objectives" },
							{ step: "02", title: "Strategy", desc: "Planning the optimal solution architecture" },
							{ step: "03", title: "Development", desc: "Building with agile methodology" },
							{ step: "04", title: "Delivery", desc: "Deploying and maintaining excellence" },
						].map((item, index) => (
							<motion.div
								key={index}
								initial={{ opacity: 0, y: 20 }}
								animate={{ opacity: 1, y: 0 }}
								transition={{ duration: 0.5, delay: index * 0.1 }}
								className="p-6 rounded-2xl bg-[#111] border border-[#222] text-center">
								<div className="text-4xl font-bold text-[#00ff88]/30 mb-4">{item.step}</div>
								<h3 className="text-xl font-bold text-white mb-2">{item.title}</h3>
								<p className="text-[#888] text-sm font-mono">{item.desc}</p>
							</motion.div>
						))}
					</div>
				</div>

				{/* CTA */}
				<motion.div
					initial={{ opacity: 0, y: 20 }}
					animate={{ opacity: 1, y: 0 }}
					transition={{ duration: 0.5, delay: 0.4 }}
					className="text-center p-12 rounded-3xl bg-gradient-to-r from-[#00ff88]/10 to-[#61afef]/10 border border-[#00ff88]/30">
					<h2 className="text-3xl font-bold text-white mb-4">Not Sure What You Need?</h2>
					<p className="text-[#888] font-mono mb-6">
						Schedule a free consultation and we&apos;ll help you find the right solution.
					</p>
					<a
						href={`mailto:${contactInfo.email}?subject=Free Consultation`}
						className="inline-flex items-center gap-2 px-8 py-3 bg-[#00ff88] text-[#0a0a0a] font-mono font-semibold rounded-lg hover:bg-[#00cc6a] transition-colors">
						Book Consultation
						<ArrowRight className="w-4 h-4" />
					</a>
				</motion.div>
			</div>
			<Footer />
		</div>
	);
}
