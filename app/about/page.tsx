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
		"We are a pioneering AI research company dedicated to developing mindful artificial intelligence and machine learning models that advance technology while maintaining ethical responsibility.",
	];

	const values = [
		{
			id: 1,
			title: "Mindful AI",
			description: "We develop AI models that are not just powerful but also ethical, responsible, and beneficial to humanity.",
		},
		{
			id: 2,
			title: "Research Excellence",
			description: "We advance the frontiers of AI/ML through rigorous research and innovative model architectures.",
		},
		{
			id: 3,
			title: "Ethical Responsibility",
			description: "We ensure our AI models are developed with fairness, transparency, and accountability at their core.",
		},
		{
			id: 4,
			title: "Innovation",
			description: "We push the boundaries of what's possible in AI while maintaining our commitment to responsible development.",
		},
	];

	const team = [
		{
			id: 1,
			name: "AI Research",
			description: "Our team specializes in cutting-edge AI research, developing novel architectures and algorithms.",
		},
		{
			id: 2,
			name: "Model Development",
			description: "We create sophisticated machine learning models tailored to solve complex real-world problems.",
		},
		{
			id: 3,
			name: "Ethical AI",
			description: "We ensure all our models are developed with ethical considerations and responsible AI principles.",
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
									To advance the field of artificial intelligence by developing mindful AI models that push the boundaries of innovation while maintaining ethical responsibility. We believe AI should enhance human capabilities and contribute positively to society.
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
									To become a global leader in responsible AI development, creating models that not only solve complex problems but do so with transparency, fairness, and consideration for their impact on humanity and society.
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
							<h2 className="heading font-bold">Ready to Explore Mindful AI Solutions?</h2>
						</motion.div>
						<motion.div
							variants={textAnimation}
							initial="initial"
							whileInView="enter"
							viewport={{ once: true }}
							className="flex gap-4 items-center mt-3 overflow-hidden flex-col xm:w-full sm:w-full">
							<a href={`mailto:${contactInfo.email}?subject=Partnership Inquiry`} className="w-full xm:w-full sm:w-full">
								<Button
									className="text-white bg-black py-2 px-4 w-full"
									title={`Email: ${contactInfo.email}`}
								/>
							</a>
							<a href="/contact" className="w-full xm:w-full sm:w-full">
								<Button
									className="text-black bg-white py-2 px-4 w-full border border-black"
									title="Contact Us"
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
