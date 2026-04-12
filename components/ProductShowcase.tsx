"use client";

import Image from "next/image";
import { motion, useScroll, useTransform } from "framer-motion";
import { TextMask } from "@/animations";
import { productImage, pyramid, tube } from "@/public";
import { useRef } from "react";
import { Zap, Cpu, Globe } from "lucide-react";

export default function ProductShowcase() {
	const container = useRef(null);

	const { scrollYProgress } = useScroll({
		target: container,
		offset: ["start end", "end start"],
	});
	const cq = useTransform(scrollYProgress, [0, 1], [0, 200]);
	const mq = useTransform(scrollYProgress, [0, 1], [0, -200]);
	const phares1 = ["Transforming Ideas Into"];
	const phares2 = ["Intelligent Solutions"];
	const phares3 = [
		"From concept to deployment, we architect solutions",
		"that leverage the latest in AI, ML, and software engineering.",
	];
	const phares4 = [
		"From concept to deployment, we architect solutions that leverage the latest in AI, ML, and software engineering.",
	];
	return (
		<div className="w-full padding-x py-20 bg-gradient-to-b from-[#0a0a0a] via-[#111] to-[#0a0a0a]">
			<div className="w-full flex flex-col gap-10">
				<div className="w-full flex items-center flex-col gap-3">
					<motion.div
						initial={{ opacity: 0, scale: 0 }}
						whileInView={{ opacity: 1, scale: 1 }}
						transition={{
							duration: 1,
							type: "spring",
						}}
						viewport={{ once: true }}
						className="flex items-center gap-2 px-4 py-2 rounded-full border border-[#00ff88]/30 bg-[#00ff88]/10">
						<Zap className="w-4 h-4 text-[#00ff88]" />
						<span className="text-[#00ff88] font-mono text-sm">Our Process</span>
					</motion.div>
					<div>
						<h1 className="heading text-center font-bold leading-tight tracking-[-2.7px] text-white">
							<TextMask>{phares1}</TextMask>
						</h1>
						<h1 className="heading text-center font-bold leading-tight tracking-[-2.7px] text-[#00ff88]">
							<TextMask>{phares2}</TextMask>
						</h1>
					</div>
					<div>
						<h1 className="text-[#888] font-mono paragraph font-normal leading-tight text-center block xm:hidden sm:hidden">
							<TextMask>{phares3}</TextMask>
						</h1>
						<h1 className="text-[#888] font-mono paragraph font-normal leading-tight text-center hidden xm:block sm:block">
							<TextMask>{phares4}</TextMask>
						</h1>
					</div>
				</div>
				<div
					className="relative"
					ref={container}>
					<motion.div
						initial={{ opacity: 0, scale: 0.5 }}
						whileInView={{ opacity: 1, scale: 1 }}
						transition={{
							duration: 1,
							type: "spring",
						}}
						viewport={{ once: true }}
						className="w-full flex items-center justify-center">
						<div className="relative w-full max-w-4xl">
							<div className="absolute inset-0 bg-gradient-to-r from-[#00ff88]/20 to-[#61afef]/20 blur-3xl" />
							<Image
								src={productImage}
								alt="Our Solutions"
								className="w-full h-full object-cover rounded-xl border border-[#333] relative z-10"
							/>
						</div>
					</motion.div>
					<motion.div
						initial={{ opacity: 0, scale: 0 }}
						whileInView={{ opacity: 1, scale: 1 }}
						transition={{
							duration: 1,
							type: "spring",
						}}
						viewport={{ once: true }}
						className="absolute -left-40 bottom-0 xm:hidden sm:hidden"
						style={{ y: mq }}>
						<div className="p-6 rounded-2xl bg-[#111] border border-[#222]">
							<Cpu className="w-12 h-12 text-[#00ff88]" />
							<p className="text-white font-mono mt-2">AI/ML</p>
						</div>
					</motion.div>
					<motion.div
						initial={{ opacity: 0, scale: 0 }}
						whileInView={{ opacity: 1, scale: 1 }}
						transition={{
							duration: 1,
							type: "spring",
						}}
						viewport={{ once: true }}
						style={{ y: cq }}
						className="absolute -right-32 -top-20 xm:hidden sm:hidden">
						<div className="p-6 rounded-2xl bg-[#111] border border-[#222]">
							<Globe className="w-12 h-12 text-[#61afef]" />
							<p className="text-white font-mono mt-2">Global Scale</p>
						</div>
					</motion.div>
				</div>
			</div>
		</div>
	);
}
