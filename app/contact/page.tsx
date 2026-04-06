<<<<<<< HEAD
import Navbar from "../components/Navbar";
import Footer from "../components/Footer";
import Contact from "../sections/Contact";

export const metadata = {
  title: "Contact | zehanx Technologies",
  description: "Get in touch with zehanx Technologies. Ready to transform your ideas into reality? Contact us for Web, AI, ML, and App development solutions.",
};

export default function ContactPage() {
  return (
    <main className="relative min-h-screen bg-black pt-20">
      <Navbar />
      <Contact />
      <Footer />
    </main>
  );
=======
"use client";

import { motion } from "framer-motion";
import { Navbar, Footer } from "@/components";
import { contactInfo, companyInfo } from "@/constants";
import { Mail, Phone, MapPin, Send, Clock, MessageCircle } from "lucide-react";

export default function ContactPage() {
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
						<span className="text-[#00ff88] font-mono text-sm">&gt; get_in_touch</span>
					</motion.div>
					<motion.h1
						initial={{ opacity: 0, y: 20 }}
						animate={{ opacity: 1, y: 0 }}
						transition={{ duration: 0.5, delay: 0.1 }}
						className="text-5xl md:text-6xl font-bold text-white mb-6">
						Let&apos;s <span className="text-[#00ff88]">Connect</span>
					</motion.h1>
					<motion.p
						initial={{ opacity: 0, y: 20 }}
						animate={{ opacity: 1, y: 0 }}
						transition={{ duration: 0.5, delay: 0.2 }}
						className="text-xl text-[#888] font-mono leading-relaxed">
						Ready to start your project? We&apos;re here to help you transform your ideas into reality.
					</motion.p>
				</div>

				<div className="grid lg:grid-cols-2 gap-12 max-w-6xl mx-auto">
					{/* Contact Info */}
					<motion.div
						initial={{ opacity: 0, x: -30 }}
						animate={{ opacity: 1, x: 0 }}
						transition={{ duration: 0.5, delay: 0.3 }}>
						<h2 className="text-2xl font-bold text-white mb-8">Contact Information</h2>
						<div className="space-y-6">
							<a
								href={`mailto:${contactInfo.email}`}
								className="flex items-start gap-4 p-6 rounded-2xl bg-[#111] border border-[#222] hover:border-[#00ff88]/30 transition-colors group">
								<div className="p-3 rounded-xl bg-[#00ff88]/10 text-[#00ff88]">
									<Mail className="w-6 h-6" />
								</div>
								<div>
									<h3 className="text-lg font-semibold text-white mb-1">Email</h3>
									<p className="text-[#00ff88] font-mono">{contactInfo.email}</p>
									<p className="text-[#666] text-sm mt-1">We&apos;ll respond within 24 hours</p>
								</div>
							</a>

							<a
								href={`https://wa.me/${contactInfo.whatsapp}`}
								className="flex items-start gap-4 p-6 rounded-2xl bg-[#111] border border-[#222] hover:border-[#00ff88]/30 transition-colors group">
								<div className="p-3 rounded-xl bg-[#00ff88]/10 text-[#00ff88]">
									<MessageCircle className="w-6 h-6" />
								</div>
								<div>
									<h3 className="text-lg font-semibold text-white mb-1">WhatsApp</h3>
									<p className="text-[#00ff88] font-mono">{contactInfo.whatsapp}</p>
									<p className="text-[#666] text-sm mt-1">Quick responses for urgent inquiries</p>
								</div>
							</a>

							<div className="flex items-start gap-4 p-6 rounded-2xl bg-[#111] border border-[#222]">
								<div className="p-3 rounded-xl bg-[#00ff88]/10 text-[#00ff88]">
									<MapPin className="w-6 h-6" />
								</div>
								<div>
									<h3 className="text-lg font-semibold text-white mb-1">Location</h3>
									<p className="text-[#888] font-mono">{contactInfo.address}</p>
									<p className="text-[#666] text-sm mt-1">Serving clients globally</p>
								</div>
							</div>

							<div className="flex items-start gap-4 p-6 rounded-2xl bg-[#111] border border-[#222]">
								<div className="p-3 rounded-xl bg-[#00ff88]/10 text-[#00ff88]">
									<Clock className="w-6 h-6" />
								</div>
								<div>
									<h3 className="text-lg font-semibold text-white mb-1">Working Hours</h3>
									<p className="text-[#888] font-mono">Mon - Sat: 9AM - 6PM (PKT)</p>
									<p className="text-[#666] text-sm mt-1">Available for urgent projects</p>
								</div>
							</div>
						</div>
					</motion.div>

					{/* Quick Actions */}
					<motion.div
						initial={{ opacity: 0, x: 30 }}
						animate={{ opacity: 1, x: 0 }}
						transition={{ duration: 0.5, delay: 0.4 }}>
						<h2 className="text-2xl font-bold text-white mb-8">Quick Actions</h2>
						<div className="space-y-4">
							<a
								href={`mailto:${contactInfo.email}?subject=Project Inquiry`}
								className="flex items-center gap-4 p-6 rounded-2xl bg-gradient-to-r from-[#00ff88]/10 to-transparent border border-[#00ff88]/30 hover:from-[#00ff88]/20 transition-all">
								<Send className="w-6 h-6 text-[#00ff88]" />
								<div>
									<h3 className="text-lg font-semibold text-white">Start a Project</h3>
									<p className="text-[#888] text-sm font-mono">Tell us about your requirements</p>
								</div>
							</a>

							<a
								href={`mailto:${contactInfo.email}?subject=Free Consultation`}
								className="flex items-center gap-4 p-6 rounded-2xl bg-[#111] border border-[#222] hover:border-[#00ff88]/30 transition-all">
								<Phone className="w-6 h-6 text-[#00ff88]" />
								<div>
									<h3 className="text-lg font-semibold text-white">Free Consultation</h3>
									<p className="text-[#888] text-sm font-mono">30-min strategy call</p>
								</div>
							</a>

							<a
								href={`mailto:${contactInfo.email}?subject=Support Request`}
								className="flex items-center gap-4 p-6 rounded-2xl bg-[#111] border border-[#222] hover:border-[#00ff88]/30 transition-all">
								<MessageCircle className="w-6 h-6 text-[#00ff88]" />
								<div>
									<h3 className="text-lg font-semibold text-white">Get Support</h3>
									<p className="text-[#888] text-sm font-mono">Technical assistance & maintenance</p>
								</div>
							</a>
						</div>

						<div className="mt-8 p-6 rounded-2xl bg-gradient-to-br from-[#1a1a2e] to-[#0a0a0a] border border-[#222]">
							<h3 className="text-lg font-semibold text-white mb-2">Response Time</h3>
							<p className="text-[#888] font-mono text-sm">
								We typically respond to all inquiries within 24 hours. 
								For urgent matters, please reach out via WhatsApp for immediate assistance.
							</p>
						</div>
					</motion.div>
				</div>

				{/* FAQ Preview */}
				<motion.div
					initial={{ opacity: 0, y: 20 }}
					animate={{ opacity: 1, y: 0 }}
					transition={{ duration: 0.5, delay: 0.5 }}
					className="max-w-4xl mx-auto mt-20">
					<h2 className="text-2xl font-bold text-white text-center mb-8">Frequently Asked Questions</h2>
					<div className="grid md:grid-cols-2 gap-4">
						{[
							{ q: "What services do you offer?", a: "ML/AI solutions, software development, and data engineering." },
							{ q: "How long does a typical project take?", a: "Timeline varies from 2 weeks to 6 months depending on complexity." },
							{ q: "Do you provide ongoing support?", a: "Yes, we offer maintenance and support packages for all projects." },
							{ q: "Can you work with international clients?", a: "Absolutely! We work with clients globally and offer flexible hours." },
						].map((faq, i) => (
							<div key={i} className="p-6 rounded-xl bg-[#111] border border-[#222]">
								<h4 className="font-semibold text-white mb-2">{faq.q}</h4>
								<p className="text-[#888] text-sm font-mono">{faq.a}</p>
							</div>
						))}
					</div>
				</motion.div>
			</div>
			<Footer />
		</div>
	);
>>>>>>> 3bc9588be4435e479cd8b5adde3400babe24a484
}
