"use client";
import Image from "next/image";
import { motion } from "framer-motion";
import { Button, Heading, Navbar, Footer } from "@/components";
import { textAnimation, imageAnimation } from "@/motion";
import { cog, cylinder } from "@/public";
import { contactInfo } from "@/constants";

export default function About() {
	const phares1 = ["About ", "Zehanx Technologies"];
	const phares2 = [
		"We are a team of passionate developers, data scientists, and AI engineers dedicated to transforming ideas into innovative solutions.",
	];

	const values = [
		{
			id: 1,
			title: "Innovation",
			description: "We push boundaries and embrace cutting-edge technologies to deliver next-generation solutions.",
		},
		{
			id: 2,
			title: "Excellence",
			description: "We maintain the highest standards of quality in every project we undertake.",
		},
		{
			id: 3,
			title: "Collaboration",
			description: "We work closely with our clients to understand their vision and bring it to life.",
		},
		{
			id: 4,
			title: "Integrity",
			description: "We operate with transparency and honesty in all our business dealings.",
		},
	];

	const team = [
		{
			id: 1,
			name: "Expertise",
			description: "Our team brings years of experience in AI, ML, Data Science, and Software Development.",
		},
		{
			id: 2,
			name: "Dedication",
			description: "We are committed to delivering exceptional results and exceeding client expectations.",
		},
		{
			id: 3,
			name: "Innovation",
			description: "We constantly explore new technologies and methodologies to stay ahead of the curve.",
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

				{/* Mission & Vision */}
				<div className="w-full padding-x py-20 bg-white xm:py-10 sm:py-10">
					<div className="w-full flex flex-col gap-20 xm:gap-10 sm:gap-10">
						<div className="w-full flex gap-10 xm:flex-col sm:flex-col">
							<motion.div
								className="w-1/2 xm:w-full sm:w-full"
								variants={textAnimation}
								initial="initial"
								whileInView="enter"
								viewport={{ once: true }}>
								<h2 className="heading font-bold mb-4">Our Mission</h2>
								<p className="paragraph text-[#010D3E] leading-relaxed">
									To empower businesses by transforming their concepts into reality through innovative AI, Machine Learning, Data Science, and Software Development solutions. We believe in the power of technology to solve complex problems and drive meaningful change.
								</p>
							</motion.div>
							<motion.div
								className="w-1/2 xm:w-full sm:w-full"
								variants={textAnimation}
								initial="initial"
								whileInView="enter"
								viewport={{ once: true }}>
								<h2 className="heading font-bold mb-4">Our Vision</h2>
								<p className="paragraph text-[#010D3E] leading-relaxed">
									To be the leading technology partner for businesses seeking to leverage AI and advanced software solutions. We envision a future where technology seamlessly integrates with business strategy to create unprecedented value.
								</p>
							</motion.div>
						</div>
					</div>
				</div>

				{/* Core Values */}
				<div className="w-full padding-x py-20 bg-gradient-to-b from-white to-[#d2dcff] xm:py-10 sm:py-10">
					<div className="w-full flex flex-col gap-10">
						<div className="w-full flex items-center flex-col gap-3">
							<motion.div
								initial={{ opacity: 0, scale: 0 }}
								whileInView={{ opacity: 1, scale: 1 }}
								transition={{ duration: 1, type: "spring" }}
								viewport={{ once: true }}>
								<button className="w-fit py-2 px-3 rounded-full border border-[#2222221A] text-black font-dmSans text-sm font-medium leading-tight tracking-[-0.02188rem]">
									Our Values
								</button>
							</motion.div>
							<h1 className="heading text-center font-bold">Core Values</h1>
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

				{/* Why Choose Us */}
				<div className="w-full padding-x py-20 bg-white xm:py-10 sm:py-10">
					<div className="w-full flex flex-col gap-10">
						<div className="w-full flex items-center flex-col gap-3">
							<h1 className="heading text-center font-bold">Why Choose Us</h1>
						</div>

						<div className="w-full grid grid-cols-3 gap-8 xm:grid-cols-1 sm:grid-cols-1">
							{team.map((item) => (
								<motion.div
									key={item.id}
									className="p-8 rounded-2xl bg-gradient-to-br from-[#183EC2] to-[#001E7F] text-white"
									initial={{ opacity: 0, scale: 0.9 }}
									whileInView={{ opacity: 1, scale: 1 }}
									transition={{ duration: 0.6, delay: item.id * 0.1 }}
									viewport={{ once: true }}>
									<h3 className="text-[24px] font-bold mb-3">{item.name}</h3>
									<p className="leading-relaxed">{item.description}</p>
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
							<h2 className="heading font-bold">Ready to Partner With Us?</h2>
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
