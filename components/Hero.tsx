import Image from "next/image";
import { motion, useScroll, useTransform } from "framer-motion";
import { Button, Heading, Navbar } from "@/components";
import { imageAnimation, textAnimation } from "@/motion";
import { ArrowRight, cog, cylinder, noodle } from "@/public";
import { companyInfo } from "@/constants";
import { useRef } from "react";

export default function Hero() {
	const container = useRef(null);

	const { scrollYProgress } = useScroll({
		target: container,
		offset: ["start end", "end start"],
	});
	const cq = useTransform(scrollYProgress, [0, 1], [0, 200]);
	const mq = useTransform(scrollYProgress, [0, 1], [0, -200]);

	const phares1 = ["Intelligent Solutions"];
	const phares2 = ["for the Digital Age"];
	const phares3 = [
		"We craft cutting-edge Machine Learning models and robust software systems that drive innovation and transform businesses.",
	];
	return (
		<div
			ref={container}
			className="w-full h-screen xm:min-h-screen sm:min-h-screen bg-gradient-to-br from-[#0a0a0a] via-[#1a1a2e] to-[#0a0a0a]">
			<Navbar />
			<div className="w-full padding-x h-full items-center flex gap-4 justify-between overflow-hidden xm:flex-col sm:flex-col xm:pt-20 sm:pt-20">
				<div className="w-1/2 xm:w-full sm:w-full flex flex-col gap-6 xm:gap-4 sm:gap-4 relative xm:flex-col sm:flex-col">
					<div className="overflow-hidden">
						<motion.div
							className="w-fit py-2 px-4 rounded-full border border-[#00ff88]/30 bg-[#00ff88]/10 text-[#00ff88] font-mono text-sm font-medium leading-tight"
							variants={textAnimation}
							initial="initial"
							whileInView="enter"
							viewport={{ once: true }}>
							<span className="terminal-green">&gt;</span> {companyInfo.tagline}
						</motion.div>
					</div>
					<div>
						<Heading
							classname="heading font-bold text-white"
							title={phares1}
						/>
						<Heading
							classname="heading font-bold text-[#00ff88]"
							title={phares2}
						/>
					</div>
					<div>
						<Heading
							classname="paragraph font-normal text-[#888]"
							title={phares3}
						/>
					</div>
					<div className="overflow-hidden">
						<motion.div
							className="flex gap-4 items-center"
							variants={textAnimation}
							initial="initial"
							whileInView="enter"
							viewport={{ once: true }}>
							<Button
								className="text-[#0a0a0a] bg-[#00ff88] py-2 px-4 font-mono font-semibold hover:bg-[#00cc6a] transition-colors"
								title="Start Your Project"
							/>
							<div className="flex items-center gap-2">
								<Button
									className="text-[#00ff88] hover:text-white transition-colors"
									title="View Our Work"
								/>
								<Image
									src={ArrowRight}
									alt="ArrowRight"
									width={20}
									height={20}
									className="text-[#00ff88]"
								/>
							</div>
						</motion.div>
					</div>
				</div>
				<div className="w-1/2 xm:w-full sm:w-full h-full relative items-center justify-center flex">
					<motion.div
						animate={{ y: [-30, 30] }}
						transition={{
							duration: 2,
							repeat: Infinity,
							repeatType: "mirror",
							ease: "easeInOut",
						}}
						className="w-full flex items-center justify-center">
						<Image
							src={cog}
							alt="AI and ML Visualization"
							width={800}
							height={400}
							className="w-[70%] xm:w-full sm:w-full h-auto object-cover opacity-80"
						/>
					</motion.div>
					<motion.div
						className="absolute -right-16 bottom-10 rotate-[30deg] xm:hidden sm:hidden"
						variants={imageAnimation}
						initial="initial"
						whileInView="enter"
						viewport={{ once: true }}
						style={{ y: mq }}>
						<Image
							src={noodle}
							alt="Code visualization"
							width={200}
							height={200}
							className="opacity-60"
						/>
					</motion.div>
					<motion.div
						className="absolute -left-20 top-20 xm:hidden sm:hidden"
						variants={imageAnimation}
						initial="initial"
						whileInView="enter"
						viewport={{ once: true }}
						style={{ y: cq }}>
						<Image
							src={cylinder}
							alt="Data flow"
							width={200}
							height={200}
							className="opacity-60"
						/>
					</motion.div>
				</div>
			</div>
		</div>
	);
}
