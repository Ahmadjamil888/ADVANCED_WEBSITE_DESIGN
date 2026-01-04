"use client";
import Image from "next/image";
import { motion } from "framer-motion";
import { Button, Heading, Navbar, Footer } from "@/components";
import { textAnimation } from "@/motion";
import { contactInfo } from "@/constants";

export default function Services() {
	const phares1 = ["Our ", "Services"];
	const phares2 = [
		"Comprehensive solutions tailored to transform your business with cutting-edge technology.",
	];

	const services = [
		{
			id: 1,
			title: "Artificial Intelligence",
			description: "Harness the power of AI to automate processes, gain insights, and create intelligent systems.",
			features: [
				"Machine Learning Models",
				"Natural Language Processing",
				"Computer Vision Solutions",
				"Predictive Analytics",
				"AI Consulting & Strategy",
				"Custom AI Solutions",
			],
		},
		{
			id: 2,
			title: "Machine Learning",
			description: "Build intelligent systems that learn and improve from experience without explicit programming.",
			features: [
				"Supervised Learning",
				"Unsupervised Learning",
				"Deep Learning",
				"Model Training & Optimization",
				"Feature Engineering",
				"ML Pipeline Development",
			],
		},
		{
			id: 3,
			title: "Data Science",
			description: "Transform raw data into actionable insights that drive business decisions.",
			features: [
				"Data Analysis & Visualization",
				"Statistical Modeling",
				"Big Data Processing",
				"Business Intelligence",
				"Data Pipeline Development",
				"Reporting & Dashboards",
			],
		},
		{
			id: 4,
			title: "Web Development",
			description: "Create responsive, scalable, and user-friendly web applications.",
			features: [
				"Frontend Development",
				"Backend Development",
				"Full-Stack Solutions",
				"Progressive Web Apps",
				"E-commerce Solutions",
				"API Development",
			],
		},
		{
			id: 5,
			title: "Software Development",
			description: "Build robust software solutions that meet your unique business requirements.",
			features: [
				"Desktop Applications",
				"Mobile App Development",
				"Cloud Solutions",
				"DevOps & Infrastructure",
				"Quality Assurance & Testing",
				"Maintenance & Support",
			],
		},
		{
			id: 6,
			title: "Consulting",
			description: "Expert guidance to help you navigate your digital transformation journey.",
			features: [
				"Technology Strategy",
				"Architecture Design",
				"Process Optimization",
				"Team Training",
				"Best Practices",
				"Custom Solutions",
			],
		},
	];

	const process = [
		{
			id: 1,
			step: "01",
			title: "Discovery",
			description: "We understand your business goals, challenges, and requirements through detailed consultation.",
		},
		{
			id: 2,
			step: "02",
			title: "Planning",
			description: "We develop a comprehensive strategy and roadmap tailored to your specific needs.",
		},
		{
			id: 3,
			step: "03",
			title: "Development",
			description: "Our expert team builds your solution using best practices and cutting-edge technologies.",
		},
		{
			id: 4,
			step: "04",
			title: "Testing",
			description: "Rigorous quality assurance ensures your solution meets the highest standards.",
		},
		{
			id: 5,
			step: "05",
			title: "Deployment",
			description: "We seamlessly deploy your solution and ensure smooth integration with your systems.",
		},
		{
			id: 6,
			step: "06",
			title: "Support",
			description: "Ongoing maintenance and support to keep your solution running optimally.",
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

				{/* Services Grid */}
				<div className="w-full padding-x py-20 bg-white xm:py-10 sm:py-10">
					<div className="w-full flex flex-col gap-10">
						<div className="w-full flex items-center flex-col gap-3">
							<motion.div
								initial={{ opacity: 0, scale: 0 }}
								whileInView={{ opacity: 1, scale: 1 }}
								transition={{ duration: 1, type: "spring" }}
								viewport={{ once: true }}>
								<button className="w-fit py-2 px-3 rounded-full border border-[#2222221A] text-black font-dmSans text-sm font-medium leading-tight tracking-[-0.02188rem]">
									What We Offer
								</button>
							</motion.div>
						</div>

						<div className="w-full grid grid-cols-3 gap-8 xm:grid-cols-1 sm:grid-cols-1">
							{services.map((service) => (
								<motion.div
									key={service.id}
									className="p-8 rounded-2xl bg-gradient-to-br from-white to-[#f5f5f5] border border-[#F1F1F1] shadow-[0px_7px_14px_0px_#EAEAEA] hover:shadow-[0px_14px_28px_0px_#EAEAEA] transition-all duration-300"
									initial={{ opacity: 0, y: 20 }}
									whileInView={{ opacity: 1, y: 0 }}
									transition={{ duration: 0.6, delay: service.id * 0.1 }}
									viewport={{ once: true }}>
									<h3 className="text-[24px] font-bold text-black mb-3">{service.title}</h3>
									<p className="text-[#010D3E] leading-relaxed mb-6">{service.description}</p>
									<div className="space-y-2">
										{service.features.map((feature, idx) => (
											<div key={idx} className="flex items-start gap-2">
												<span className="text-[#183EC2] font-bold mt-1">✓</span>
												<p className="text-[#010D3E] text-sm">{feature}</p>
											</div>
										))}
									</div>
								</motion.div>
							))}
						</div>
					</div>
				</div>

				{/* Our Process */}
				<div className="w-full padding-x py-20 bg-gradient-to-b from-white to-[#d2dcff] xm:py-10 sm:py-10">
					<div className="w-full flex flex-col gap-10">
						<div className="w-full flex items-center flex-col gap-3">
							<motion.div
								initial={{ opacity: 0, scale: 0 }}
								whileInView={{ opacity: 1, scale: 1 }}
								transition={{ duration: 1, type: "spring" }}
								viewport={{ once: true }}>
								<button className="w-fit py-2 px-3 rounded-full border border-[#2222221A] text-black font-dmSans text-sm font-medium leading-tight tracking-[-0.02188rem]">
									Our Process
								</button>
							</motion.div>
							<h1 className="heading text-center font-bold">How We Work</h1>
						</div>

						<div className="w-full grid grid-cols-3 gap-8 xm:grid-cols-1 sm:grid-cols-1">
							{process.map((item) => (
								<motion.div
									key={item.id}
									className="p-8 rounded-2xl bg-white border border-[#F1F1F1] shadow-[0px_7px_14px_0px_#EAEAEA]"
									initial={{ opacity: 0, scale: 0.9 }}
									whileInView={{ opacity: 1, scale: 1 }}
									transition={{ duration: 0.6, delay: item.id * 0.1 }}
									viewport={{ once: true }}>
									<div className="text-5xl font-bold text-[#183EC2] mb-4">{item.step}</div>
									<h3 className="text-[24px] font-bold text-black mb-3">{item.title}</h3>
									<p className="text-[#010D3E] leading-relaxed">{item.description}</p>
								</motion.div>
							))}
						</div>
					</div>
				</div>

				{/* Technology Stack */}
				<div className="w-full padding-x py-20 bg-white xm:py-10 sm:py-10">
					<div className="w-full flex flex-col gap-10">
						<div className="w-full flex items-center flex-col gap-3">
							<motion.div
								initial={{ opacity: 0, scale: 0 }}
								whileInView={{ opacity: 1, scale: 1 }}
								transition={{ duration: 1, type: "spring" }}
								viewport={{ once: true }}>
								<button className="w-fit py-2 px-3 rounded-full border border-[#2222221A] text-black font-dmSans text-sm font-medium leading-tight tracking-[-0.02188rem]">
									Technologies
								</button>
							</motion.div>
							<h1 className="heading text-center font-bold">Technologies We Use</h1>
						</div>

						<div className="w-full grid grid-cols-2 gap-8 xm:grid-cols-1 sm:grid-cols-1">
							<motion.div
								className="p-8 rounded-2xl bg-gradient-to-br from-[#183EC2] to-[#001E7F] text-white"
								initial={{ opacity: 0, x: -20 }}
								whileInView={{ opacity: 1, x: 0 }}
								transition={{ duration: 0.6 }}
								viewport={{ once: true }}>
								<h3 className="text-[24px] font-bold mb-4">AI & ML</h3>
								<div className="space-y-2">
									{["TensorFlow", "PyTorch", "Scikit-learn", "XGBoost", "OpenAI APIs"].map((tech, idx) => (
										<p key={idx} className="text-sm">✓ {tech}</p>
									))}
								</div>
							</motion.div>

							<motion.div
								className="p-8 rounded-2xl bg-gradient-to-br from-[#183EC2] to-[#001E7F] text-white"
								initial={{ opacity: 0, x: 20 }}
								whileInView={{ opacity: 1, x: 0 }}
								transition={{ duration: 0.6 }}
								viewport={{ once: true }}>
								<h3 className="text-[24px] font-bold mb-4">Web & Backend</h3>
								<div className="space-y-2">
									{["React", "Next.js", "Node.js", "Python", "PostgreSQL"].map((tech, idx) => (
										<p key={idx} className="text-sm">✓ {tech}</p>
									))}
								</div>
							</motion.div>

							<motion.div
								className="p-8 rounded-2xl bg-gradient-to-br from-[#183EC2] to-[#001E7F] text-white"
								initial={{ opacity: 0, x: -20 }}
								whileInView={{ opacity: 1, x: 0 }}
								transition={{ duration: 0.6, delay: 0.2 }}
								viewport={{ once: true }}>
								<h3 className="text-[24px] font-bold mb-4">Data & Analytics</h3>
								<div className="space-y-2">
									{["Pandas", "NumPy", "Tableau", "Power BI", "Apache Spark"].map((tech, idx) => (
										<p key={idx} className="text-sm">✓ {tech}</p>
									))}
								</div>
							</motion.div>

							<motion.div
								className="p-8 rounded-2xl bg-gradient-to-br from-[#183EC2] to-[#001E7F] text-white"
								initial={{ opacity: 0, x: 20 }}
								whileInView={{ opacity: 1, x: 0 }}
								transition={{ duration: 0.6, delay: 0.2 }}
								viewport={{ once: true }}>
								<h3 className="text-[24px] font-bold mb-4">Cloud & DevOps</h3>
								<div className="space-y-2">
									{["AWS", "Google Cloud", "Azure", "Docker", "Kubernetes"].map((tech, idx) => (
										<p key={idx} className="text-sm">✓ {tech}</p>
									))}
								</div>
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
							<h2 className="heading font-bold">Ready to Get Started?</h2>
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
