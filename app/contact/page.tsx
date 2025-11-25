"use client";
import React, { useState } from "react";
import Image from "next/image";
import { motion } from "framer-motion";
import { Button, Heading, Navbar, Footer } from "@/components";
import { textAnimation } from "@/motion";
import { contactInfo } from "@/constants";

export default function Contact() {
	const phares1 = ["Get in ", "Touch"];
	const phares2 = [
		"We'd love to hear from you. Reach out to us through any of the following channels.",
	];

	const contactMethods = [
		{
			id: 1,
			title: "Email",
			description: "Send us an email and we'll respond within 24 hours.",
			value: contactInfo.email,
			link: `mailto:${contactInfo.email}`,
			linkText: "Send Email",
		},
		{
			id: 2,
			title: "Phone",
			description: "Call us directly for immediate assistance.",
			value: contactInfo.phone,
			link: `tel:${contactInfo.phone}`,
			linkText: "Call Now",
		},
		{
			id: 3,
			title: "Company",
			description: "Learn more about our organization.",
			value: contactInfo.company,
			link: "/about",
			linkText: "About Us",
		},
	];

	const socialLinks = [
		{
			id: 1,
			name: "LinkedIn",
			description: "Connect with us on LinkedIn",
			link: "https://linkedin.com",
		},
		{
			id: 2,
			name: "GitHub",
			description: "Check out our projects",
			link: "https://github.com",
		},
		{
			id: 3,
			name: "Twitter",
			description: "Follow us for updates",
			link: "https://twitter.com",
		},
		{
			id: 4,
			name: "Facebook",
			description: "Like our page",
			link: "https://facebook.com",
		},
	];

	const faqs = [
		{
			id: 1,
			question: "What is your typical project timeline?",
			answer: "Project timelines vary based on scope and complexity. We typically provide detailed timelines after the discovery phase. Most projects range from 2-6 months.",
		},
		{
			id: 2,
			question: "Do you offer ongoing support?",
			answer: "Yes, we provide comprehensive maintenance and support packages. We can discuss your specific needs during our consultation.",
		},
		{
			id: 3,
			question: "What technologies do you specialize in?",
			answer: "We specialize in AI, ML, Data Science, Web Development, and Software Development. We work with modern technologies like React, Python, TensorFlow, and cloud platforms.",
		},
		{
			id: 4,
			question: "How do you ensure project quality?",
			answer: "We follow rigorous quality assurance processes including code reviews, automated testing, and comprehensive testing before deployment.",
		},
		{
			id: 5,
			question: "Can you work with existing systems?",
			answer: "Absolutely! We can integrate with your existing systems and provide seamless solutions that work with your current infrastructure.",
		},
		{
			id: 6,
			question: "What is your pricing model?",
			answer: "We offer flexible pricing models including fixed-price projects, time & materials, and retainer-based engagements. Contact us for a custom quote.",
		},
	];

	const [expandedFaq, setExpandedFaq] = useState<number | null>(null);

	return (
		<>
			<Navbar />
			<div className="pt-20">
				{/* Hero Section */}
				<div className="w-full padding-x py-20 bg-[radial-gradient(ellipse_200%_100%_at_bottom_left,#183EC2,#EAEEFE_80%)] xm:py-10 sm:py-10">
					<div className="w-full flex flex-col gap-10 items-center justify-center">
						<div className="overflow-hidden">
							<motion.div
								variants={textAnimation}
								initial="initial"
								whileInView="enter"
								viewport={{ once: true }}>
								<Heading
									classname="heading font-bold text-center"
									title={phares1}
								/>
							</motion.div>
						</div>
						<div className="overflow-hidden max-w-2xl">
							<motion.div
								variants={textAnimation}
								initial="initial"
								whileInView="enter"
								viewport={{ once: true }}>
								<Heading
									classname="paragraph font-normal text-center"
									title={phares2}
								/>
							</motion.div>
						</div>
					</div>
				</div>

				{/* Contact Methods */}
				<div className="w-full padding-x py-20 bg-white xm:py-10 sm:py-10">
					<div className="w-full flex flex-col gap-10">
						<div className="w-full flex items-center flex-col gap-3">
							<motion.div
								initial={{ opacity: 0, scale: 0 }}
								whileInView={{ opacity: 1, scale: 1 }}
								transition={{ duration: 1, type: "spring" }}
								viewport={{ once: true }}>
								<button className="w-fit py-2 px-3 rounded-full border border-[#2222221A] text-black font-dmSans text-sm font-medium leading-tight tracking-[-0.02188rem]">
									Contact Details
								</button>
							</motion.div>
						</div>

						<div className="w-full grid grid-cols-3 gap-8 xm:grid-cols-1 sm:grid-cols-1">
							{contactMethods.map((method) => (
								<motion.div
									key={method.id}
									className="p-8 rounded-2xl bg-gradient-to-br from-white to-[#f5f5f5] border border-[#F1F1F1] shadow-[0px_7px_14px_0px_#EAEAEA] hover:shadow-[0px_14px_28px_0px_#EAEAEA] transition-all duration-300"
									initial={{ opacity: 0, y: 20 }}
									whileInView={{ opacity: 1, y: 0 }}
									transition={{ duration: 0.6, delay: method.id * 0.1 }}
									viewport={{ once: true }}>
									<h3 className="text-[24px] font-bold text-black mb-2">{method.title}</h3>
									<p className="text-[#010D3E] text-sm mb-4">{method.description}</p>
									<p className="text-[#183EC2] font-bold mb-4 break-all">{method.value}</p>
									<a href={method.link}>
										<Button
											className="text-white bg-black py-2 px-4 w-full"
											title={method.linkText}
										/>
									</a>
								</motion.div>
							))}
						</div>
					</div>
				</div>

				{/* Social Links */}
				<div className="w-full padding-x py-20 bg-gradient-to-b from-white to-[#d2dcff] xm:py-10 sm:py-10">
					<div className="w-full flex flex-col gap-10">
						<div className="w-full flex items-center flex-col gap-3">
							<motion.div
								initial={{ opacity: 0, scale: 0 }}
								whileInView={{ opacity: 1, scale: 1 }}
								transition={{ duration: 1, type: "spring" }}
								viewport={{ once: true }}>
								<button className="w-fit py-2 px-3 rounded-full border border-[#2222221A] text-black font-dmSans text-sm font-medium leading-tight tracking-[-0.02188rem]">
									Follow Us
								</button>
							</motion.div>
							<h1 className="heading text-center font-bold">Connect on Social Media</h1>
						</div>

						<div className="w-full grid grid-cols-4 gap-8 xm:grid-cols-2 sm:grid-cols-2">
							{socialLinks.map((social) => (
								<motion.a
									key={social.id}
									href={social.link}
									target="_blank"
									rel="noopener noreferrer"
									className="p-8 rounded-2xl bg-white border border-[#F1F1F1] shadow-[0px_7px_14px_0px_#EAEAEA] hover:shadow-[0px_14px_28px_0px_#EAEAEA] transition-all duration-300 text-center"
									initial={{ opacity: 0, scale: 0.9 }}
									whileInView={{ opacity: 1, scale: 1 }}
									transition={{ duration: 0.6, delay: social.id * 0.1 }}
									viewport={{ once: true }}>
									<h3 className="text-[20px] font-bold text-black mb-2">{social.name}</h3>
									<p className="text-[#010D3E] text-sm">{social.description}</p>
								</motion.a>
							))}
						</div>
					</div>
				</div>

				{/* FAQ Section */}
				<div className="w-full padding-x py-20 bg-white xm:py-10 sm:py-10">
					<div className="w-full flex flex-col gap-10">
						<div className="w-full flex items-center flex-col gap-3">
							<motion.div
								initial={{ opacity: 0, scale: 0 }}
								whileInView={{ opacity: 1, scale: 1 }}
								transition={{ duration: 1, type: "spring" }}
								viewport={{ once: true }}>
								<button className="w-fit py-2 px-3 rounded-full border border-[#2222221A] text-black font-dmSans text-sm font-medium leading-tight tracking-[-0.02188rem]">
									FAQ
								</button>
							</motion.div>
							<h1 className="heading text-center font-bold">Frequently Asked Questions</h1>
						</div>

						<div className="w-full max-w-3xl mx-auto space-y-4">
							{faqs.map((faq) => (
								<motion.div
									key={faq.id}
									className="rounded-2xl border border-[#F1F1F1] shadow-[0px_7px_14px_0px_#EAEAEA] overflow-hidden"
									initial={{ opacity: 0, y: 10 }}
									whileInView={{ opacity: 1, y: 0 }}
									transition={{ duration: 0.6, delay: faq.id * 0.05 }}
									viewport={{ once: true }}>
									<button
										onClick={() => setExpandedFaq(expandedFaq === faq.id ? null : faq.id)}
										className="w-full p-6 flex items-center justify-between bg-white hover:bg-[#f9f9f9] transition-colors">
										<h3 className="text-[18px] font-bold text-black text-left">{faq.question}</h3>
										<span className={`text-2xl transition-transform duration-300 ${expandedFaq === faq.id ? 'rotate-180' : ''}`}>
											▼
										</span>
									</button>
									{expandedFaq === faq.id && (
										<motion.div
											initial={{ opacity: 0, height: 0 }}
											animate={{ opacity: 1, height: "auto" }}
											exit={{ opacity: 0, height: 0 }}
											transition={{ duration: 0.3 }}
											className="px-6 pb-6 bg-[#f9f9f9] border-t border-[#F1F1F1]">
											<p className="text-[#010D3E] leading-relaxed">{faq.answer}</p>
										</motion.div>
									)}
								</motion.div>
							))}
						</div>
					</div>
				</div>

				{/* Quick Contact CTA */}
				<div className="w-full padding-x py-20 bg-gradient-to-b from-white to-[#d2dcff] xm:py-10 sm:py-10">
					<div className="w-full flex flex-col gap-8 items-center justify-center text-center">
						<motion.div
							variants={textAnimation}
							initial="initial"
							whileInView="enter"
							viewport={{ once: true }}>
							<h2 className="heading font-bold">Still have questions?</h2>
							<p className="paragraph text-[#010D3E] mt-4">Reach out to us directly - we're here to help!</p>
						</motion.div>
						<motion.div
							variants={textAnimation}
							initial="initial"
							whileInView="enter"
							viewport={{ once: true }}
							className="flex gap-4 items-center mt-3 overflow-hidden flex-col xm:w-full sm:w-full">
							<a href={`mailto:${contactInfo.email}`} className="w-full xm:w-full sm:w-full">
								<Button
									className="text-white bg-black py-2 px-4 w-full"
									title={`Email: ${contactInfo.email}`}
								/>
							</a>
							<a href={`tel:${contactInfo.phone}`} className="w-full xm:w-full sm:w-full">
								<Button
									className="text-black bg-white py-2 px-4 w-full border border-black"
									title={`Call: ${contactInfo.phone}`}
								/>
							</a>
						</motion.div>
					</div>
				</div>
			</div>
			<Footer />
		</>
	);
}
