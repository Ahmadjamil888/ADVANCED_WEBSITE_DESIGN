"use client";
import { motion } from "framer-motion";
import { Button, Heading, Navbar, Footer } from "@/components";
import { textAnimation } from "@/motion";
import { contactInfo } from "@/constants";

export default function Privacy() {
	const phares1 = ["Privacy ", "Policy"];
	const phares2 = [
		"Your privacy is important to us. This policy explains how we collect, use, and protect your information.",
	];

	const sections = [
		{
			id: 1,
			title: "Information We Collect",
			content: [
				"We collect information you provide directly to us, such as when you contact us through our website, request our services, or subscribe to our newsletter.",
				"This may include your name, email address, phone number, company information, and any other details you choose to provide.",
				"We also automatically collect certain information about your device and how you interact with our website, including IP address, browser type, and pages visited.",
			],
		},
		{
			id: 2,
			title: "How We Use Your Information",
			content: [
				"We use the information we collect to provide, maintain, and improve our services and website.",
				"We may use your contact information to respond to your inquiries, send you updates about our services, and provide customer support.",
				"We analyze usage data to understand how our website is used and to optimize our services for better user experience.",
				"We may use your information to comply with legal obligations and enforce our terms of service.",
			],
		},
		{
			id: 3,
			title: "Data Protection",
			content: [
				"We implement appropriate technical and organizational measures to protect your personal information against unauthorized access, alteration, disclosure, or destruction.",
				"We use industry-standard encryption protocols to secure data transmission over the internet.",
				"Access to your personal information is restricted to employees and contractors who need to know that information to provide services to you.",
				"We regularly review and update our security practices to ensure the highest level of protection.",
			],
		},
		{
			id: 4,
			title: "Third-Party Sharing",
			content: [
				"We do not sell, trade, or rent your personal information to third parties.",
				"We may share your information with trusted service providers who assist us in operating our website and conducting our business, subject to confidentiality agreements.",
				"We may disclose your information when required by law or when we believe in good faith that disclosure is necessary to protect our rights, your safety, or the safety of others.",
			],
		},
		{
			id: 5,
			title: "Cookies and Tracking",
			content: [
				"Our website uses cookies to enhance your browsing experience and analyze site traffic.",
				"You can control cookie settings through your browser preferences. However, disabling cookies may affect the functionality of our website.",
				"We use analytics tools to track user behavior and improve our services. This data is aggregated and does not identify individuals.",
			],
		},
		{
			id: 6,
			title: "Your Rights",
			content: [
				"You have the right to access, correct, or delete your personal information at any time.",
				"You can opt out of receiving marketing communications from us by following the unsubscribe instructions in our emails.",
				"You have the right to request information about what personal data we hold about you.",
				"To exercise these rights, please contact us at the email address provided below.",
			],
		},
		{
			id: 7,
			title: "Changes to This Policy",
			content: [
				"We may update this privacy policy from time to time to reflect changes in our practices or for other operational, legal, or regulatory reasons.",
				"We will notify you of any material changes by posting the updated policy on our website and updating the effective date.",
				"Your continued use of our website following the posting of revised privacy policy means that you accept and agree to the changes.",
			],
		},
		{
			id: 8,
			title: "Contact Us",
			content: [
				"If you have any questions about this privacy policy or our privacy practices, please contact us at:",
				`Email: ${contactInfo.email}`,
				`Phone: ${contactInfo.phone}`,
				"We will respond to your inquiry within 24 hours.",
			],
		},
	];

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

				{/* Content Sections */}
				<div className="w-full padding-x py-20 bg-white xm:py-10 sm:py-10">
					<div className="w-full max-w-3xl mx-auto">
						<div className="space-y-12">
							{sections.map((section) => (
								<motion.div
									key={section.id}
									initial={{ opacity: 0, y: 20 }}
									whileInView={{ opacity: 1, y: 0 }}
									transition={{ duration: 0.6, delay: section.id * 0.05 }}
									viewport={{ once: true }}>
									<h2 className="text-[28px] font-bold text-black mb-4">{section.title}</h2>
									<div className="space-y-3">
										{section.content.map((paragraph, idx) => (
											<p key={idx} className="text-[#010D3E] leading-relaxed">
												{paragraph}
											</p>
										))}
									</div>
								</motion.div>
							))}
						</div>
					</div>
				</div>

				{/* CTA Section */}
				<div className="w-full padding-x py-20 bg-gradient-to-b from-white to-[#d2dcff] xm:py-10 sm:py-10">
					<div className="w-full flex flex-col gap-8 items-center justify-center text-center">
						<motion.div
							variants={textAnimation}
							initial="initial"
							whileInView="enter"
							viewport={{ once: true }}>
							<h2 className="heading font-bold">Questions About Our Privacy Policy?</h2>
							<p className="paragraph text-[#010D3E] mt-4">Contact us anytime for clarification or concerns.</p>
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
