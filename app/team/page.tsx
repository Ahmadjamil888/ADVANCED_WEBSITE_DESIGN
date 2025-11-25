"use client";
import Image from "next/image";
import { motion } from "framer-motion";
import { Button, Heading, Navbar, Footer } from "@/components";
import { textAnimation } from "@/motion";
import { contactInfo } from "@/constants";

export default function Team() {
	const phares1 = ["Meet Our ", "Team"];
	const phares2 = [
		"Talented professionals dedicated to transforming your ideas into reality.",
	];

	const teamMembers = [
		{
			id: 1,
			name: "Ahmad Jamil",
			role: "Founder",
			description: "Visionary leader with expertise in AI and strategic business development. Ahmad drives innovation and sets the direction for Zehanx Technologies.",
			email: "ahmadjamildhami@gmail.com",
		},
		{
			id: 2,
			name: "Humayl Butt",
			role: "Co-Founder",
			description: "Technical architect with deep expertise in software development and cloud infrastructure. Humayl ensures our solutions are scalable and robust.",
			email: "master.gamer69@gmail.com",
		},
		{
			id: 3,
			name: "Ahmad Ibrahim",
			role: "Co-Founder",
			description: "Data science and machine learning specialist. Ahmad Ibrahim leads our AI initiatives and ensures cutting-edge solutions for our clients.",
			email: "rafaqatyaseen@gmail.com",
		},
	];

	const values = [
		{
			id: 1,
			title: "Collaboration",
			description: "We work together seamlessly to deliver exceptional results.",
		},
		{
			id: 2,
			title: "Innovation",
			description: "We constantly push boundaries to find better solutions.",
		},
		{
			id: 3,
			title: "Excellence",
			description: "We maintain the highest standards in everything we do.",
		},
		{
			id: 4,
			title: "Integrity",
			description: "We operate with transparency and honesty always.",
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

				{/* Team Members */}
				<div className="w-full padding-x py-20 bg-white xm:py-10 sm:py-10">
					<div className="w-full flex flex-col gap-10">
						<div className="w-full flex items-center flex-col gap-3">
							<motion.div
								initial={{ opacity: 0, scale: 0 }}
								whileInView={{ opacity: 1, scale: 1 }}
								transition={{ duration: 1, type: "spring" }}
								viewport={{ once: true }}>
								<button className="w-fit py-2 px-3 rounded-full border border-[#2222221A] text-black font-dmSans text-sm font-medium leading-tight tracking-[-0.02188rem]">
									Leadership Team
								</button>
							</motion.div>
						</div>

						<div className="w-full grid grid-cols-3 gap-8 xm:grid-cols-1 sm:grid-cols-1">
							{teamMembers.map((member) => (
								<motion.div
									key={member.id}
									className="p-8 rounded-2xl bg-gradient-to-br from-white to-[#f5f5f5] border border-[#F1F1F1] shadow-[0px_7px_14px_0px_#EAEAEA] hover:shadow-[0px_14px_28px_0px_#EAEAEA] transition-all duration-300"
									initial={{ opacity: 0, y: 20 }}
									whileInView={{ opacity: 1, y: 0 }}
									transition={{ duration: 0.6, delay: member.id * 0.1 }}
									viewport={{ once: true }}>
									<div className="w-12 h-12 rounded-full bg-[#183EC2] mb-4"></div>
									<h3 className="text-[24px] font-bold text-black mb-2">{member.name}</h3>
									<p className="text-[#183EC2] font-bold mb-3">{member.role}</p>
									<p className="text-[#010D3E] leading-relaxed mb-6">{member.description}</p>
									<a href={`mailto:${member.email}`}>
										<Button
											className="text-white bg-black py-2 px-4 w-full"
											title="Contact"
										/>
									</a>
								</motion.div>
							))}
						</div>
					</div>
				</div>

				{/* Team Values */}
				<div className="w-full padding-x py-20 bg-gradient-to-b from-white to-[#d2dcff] xm:py-10 sm:py-10">
					<div className="w-full flex flex-col gap-10">
						<div className="w-full flex items-center flex-col gap-3">
							<motion.div
								initial={{ opacity: 0, scale: 0 }}
								whileInView={{ opacity: 1, scale: 1 }}
								transition={{ duration: 1, type: "spring" }}
								viewport={{ once: true }}>
								<button className="w-fit py-2 px-3 rounded-full border border-[#2222221A] text-black font-dmSans text-sm font-medium leading-tight tracking-[-0.02188rem]">
									Team Values
								</button>
							</motion.div>
							<h1 className="heading text-center font-bold">What We Stand For</h1>
						</div>

						<div className="w-full grid grid-cols-2 gap-8 xm:grid-cols-1 sm:grid-cols-1">
							{values.map((value) => (
								<motion.div
									key={value.id}
									className="p-8 rounded-2xl bg-white border border-[#F1F1F1] shadow-[0px_7px_14px_0px_#EAEAEA]"
									initial={{ opacity: 0, y: 20 }}
									whileInView={{ opacity: 1, y: 0 }}
									transition={{ duration: 0.6, delay: value.id * 0.1 }}
									viewport={{ once: true }}>
									<h3 className="text-[24px] font-bold text-black mb-3">{value.title}</h3>
									<p className="text-[#010D3E] leading-relaxed">{value.description}</p>
								</motion.div>
							))}
						</div>
					</div>
				</div>

				{/* Team Culture */}
				<div className="w-full padding-x py-20 bg-white xm:py-10 sm:py-10">
					<div className="w-full flex flex-col gap-10">
						<div className="w-full flex items-center flex-col gap-3">
							<h1 className="heading text-center font-bold">Our Culture</h1>
						</div>

						<div className="w-full max-w-3xl mx-auto">
							<motion.div
								className="p-8 rounded-2xl bg-gradient-to-br from-[#183EC2] to-[#001E7F] text-white"
								initial={{ opacity: 0, scale: 0.9 }}
								whileInView={{ opacity: 1, scale: 1 }}
								transition={{ duration: 0.6 }}
								viewport={{ once: true }}>
								<h3 className="text-[24px] font-bold mb-4">Building Tomorrow, Together</h3>
								<p className="leading-relaxed mb-4">
									At Zehanx Technologies, we believe in fostering an environment where innovation thrives and every team member can contribute their best. We're committed to continuous learning, supporting each other's growth, and delivering solutions that make a real difference.
								</p>
								<p className="leading-relaxed">
									Our team is passionate about technology and dedicated to transforming concepts into reality. We work collaboratively to solve complex problems and create value for our clients and ourselves.
								</p>
							</motion.div>
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
							<h2 className="heading font-bold">Want to Join Our Team?</h2>
							<p className="paragraph text-[#010D3E] mt-4">We're always looking for talented individuals to join us on our journey.</p>
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
