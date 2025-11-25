"use client";
import { motion } from "framer-motion";
import { Button, Heading, Navbar, Footer } from "@/components";
import { textAnimation } from "@/motion";
import { contactInfo } from "@/constants";

export default function Terms() {
	const phares1 = ["Terms of ", "Service"];
	const phares2 = [
		"Please read these terms carefully before using our website and services.",
	];

	const sections = [
		{
			id: 1,
			title: "Acceptance of Terms",
			content: [
				"By accessing and using this website and our services, you accept and agree to be bound by the terms and provision of this agreement.",
				"If you do not agree to abide by the above, please do not use this service.",
				"We reserve the right to modify these terms at any time. Your continued use of the website following the posting of revised terms means that you accept and agree to the changes.",
			],
		},
		{
			id: 2,
			title: "Use License",
			content: [
				"Permission is granted to temporarily download one copy of the materials (information or software) on our website for personal, non-commercial transitory viewing only.",
				"This is the grant of a license, not a transfer of title, and under this license you may not:",
				"• Modify or copy the materials",
				"• Use the materials for any commercial purpose or for any public display",
				"• Attempt to decompile or reverse engineer any software contained on the website",
				"• Remove any copyright or other proprietary notations from the materials",
				"• Transfer the materials to another person or 'mirror' the materials on any other server",
			],
		},
		{
			id: 3,
			title: "Disclaimer",
			content: [
				"The materials on our website are provided on an 'as is' basis. We make no warranties, expressed or implied, and hereby disclaim and negate all other warranties including, without limitation, implied warranties or conditions of merchantability, fitness for a particular purpose, or non-infringement of intellectual property or other violation of rights.",
				"Further, we do not warrant or make any representations concerning the accuracy, likely results, or reliability of the use of the materials on its website or otherwise relating to such materials or on any sites linked to this site.",
			],
		},
		{
			id: 4,
			title: "Limitations",
			content: [
				"In no event shall our company or its suppliers be liable for any damages (including, without limitation, damages for loss of data or profit, or due to business interruption) arising out of the use or inability to use the materials on our website, even if we or our authorized representative has been notified orally or in writing of the possibility of such damage.",
				"Because some jurisdictions do not allow limitations on implied warranties, or limitations of liability for consequential or incidental damages, these limitations may not apply to you.",
			],
		},
		{
			id: 5,
			title: "Accuracy of Materials",
			content: [
				"The materials appearing on our website could include technical, typographical, or photographic errors.",
				"We do not warrant that any of the materials on our website are accurate, complete, or current.",
				"We may make changes to the materials contained on our website at any time without notice.",
				"We do not, however, make any commitment to update the materials.",
			],
		},
		{
			id: 6,
			title: "Links",
			content: [
				"We have not reviewed all of the sites linked to our website and are not responsible for the contents of any such linked site.",
				"The inclusion of any link does not imply endorsement by us of the site. Use of any such linked website is at the user's own risk.",
				"If you believe that any linked site infringes upon any of your copyrights, please contact us immediately.",
			],
		},
		{
			id: 7,
			title: "Modifications",
			content: [
				"We may revise these terms of service for our website at any time without notice.",
				"By using this website, you are agreeing to be bound by the then current version of these terms of service.",
			],
		},
		{
			id: 8,
			title: "Governing Law",
			content: [
				"These terms and conditions are governed by and construed in accordance with the laws of Pakistan, and you irrevocably submit to the exclusive jurisdiction of the courts in that location.",
			],
		},
		{
			id: 9,
			title: "Intellectual Property Rights",
			content: [
				"All content on our website, including text, graphics, logos, images, and software, is the property of Zehanx Technologies or its content suppliers and is protected by international copyright laws.",
				"You may not reproduce, distribute, transmit, display, or perform any content from our website without our prior written permission.",
				"Unauthorized use of any content may violate copyright, trademark, and other laws.",
			],
		},
		{
			id: 10,
			title: "User Conduct",
			content: [
				"You agree not to use our website or services for any unlawful purpose or in any way that could damage, disable, or impair our website.",
				"You agree not to attempt to gain unauthorized access to our website or systems.",
				"You agree not to transmit any harmful code, viruses, or malware through our website.",
				"You agree not to engage in any form of harassment, abuse, or discrimination.",
			],
		},
		{
			id: 11,
			title: "Service Availability",
			content: [
				"We strive to maintain continuous service availability, but we do not guarantee that our website or services will be available at all times.",
				"We may perform maintenance, updates, or other activities that could temporarily interrupt service.",
				"We are not liable for any damages resulting from service interruptions or unavailability.",
			],
		},
		{
			id: 12,
			title: "Contact Information",
			content: [
				"If you have any questions about these terms of service, please contact us at:",
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
							<h2 className="heading font-bold">Questions About Our Terms?</h2>
							<p className="paragraph text-[#010D3E] mt-4">Contact us for any clarification or concerns.</p>
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
