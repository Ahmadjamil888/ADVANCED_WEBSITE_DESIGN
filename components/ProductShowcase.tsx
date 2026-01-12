import Image from "next/image";
import { motion, useScroll, useTransform } from "framer-motion";
import { TextMask } from "@/animations";
import { daakKhanaImage, pyramid, tube } from "@/public";
import { useRef } from "react";

export default function ProductShowcase() {
	const container = useRef(null);

	const { scrollYProgress } = useScroll({
		target: container,
		offset: ["start end", "end start"],
	});
	const cq = useTransform(scrollYProgress, [0, 1], [0, 200]);
	const mq = useTransform(scrollYProgress, [0, 1], [0, -200]);
	const phares1 = ["Our expertise in", "enterprise software solutions"];
	const phares2 = [
		"We specialize in B2B solutions, AI/ML technologies,",
		"cybersecurity, and proprietary software products to deliver",
		"comprehensive solutions tailored to your business needs.",
	];
	const phares3 = [
		"We specialize in B2B solutions, AI/ML technologies, cybersecurity, and proprietary software products to deliver comprehensive solutions tailored to your business needs.",
	];
	return (
		<div className="w-full padding-x py-10 bg-gradient-to-b from-white to-[#d2dcff]">
			<div className="w-full flex flex-col gap-10">
				<div className="w-full flex items-center flex-col gap-3">
					<motion.div
						initial={{ opacity: 0, scale: 0 }}
						whileInView={{ opacity: 1, scale: 1 }}
						transition={{
							duration: 1,
							type: "spring",
						}}
						viewport={{ once: true }}>
						<button className="w-fit py-2 px-3 rounded-full border border-[#2222221A] text-black font-dmSans text-sm font-medium leading-tight tracking-[-0.02188rem]">
							Software Solutions
						</button>
					</motion.div>
					<div>
						<h1 className="heading text-center font-bold leading-tight tracking-[-2.7px] bg-gradient-to-b from-black to-[#001E7F] bg-clip-text">
							<TextMask>{phares1}</TextMask>
						</h1>
					</div>
					<div>
						<h1 className="text-[#010D3E] font-dmSans paragraph font-normal leading-tight text-center block xm:hidden sm:hidden">
							<TextMask>{phares2}</TextMask>
						</h1>
						<h1 className="text-[#010D3E] font-dmSans paragraph font-normal leading-tight text-center hidden xm:block sm:block">
							<TextMask>{phares3}</TextMask>
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
						<Image
							src={daakKhanaImage}
							alt="daak khana product image"
							className="w-4/5 h-full object-cover rounded-[50px]"
						/>
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
						<Image
							src={tube}
							alt="tube-hero-img"
							width={250}
							height={250}
						/>
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
						<Image
							src={pyramid}
							alt="pyramid-hero-img"
							width={250}
							height={250}
						/>
					</motion.div>
				</div>
			</div>
		</div>
	);
}