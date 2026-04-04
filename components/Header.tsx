"use client";
import Link from "next/link";
import Image from "next/image";
import { useState } from "react";
import { Button } from "@/components";
import { navVariants } from "@/motion";
import { MobileMenu } from "@/animations";
import { navigationItems, companyInfo } from "@/constants";
import { logoLight, arrowRightWhite } from "@/public";
import { motion, useMotionValueEvent, useScroll } from "framer-motion";

export default function Navbar() {
	const { scrollY } = useScroll();
	const [hidden, setHidden] = useState(false);

	useMotionValueEvent(scrollY, "change", (latest) => {
		if (latest > 0) {
			setHidden(true);
		} else {
			setHidden(false);
		}
	});
	return (
		<>
			<motion.div
				initial="initial"
				whileInView="enter"
				variants={navVariants}
				className="fixed w-full top-0 z-50 xm:hidden sm:hidden">
				<motion.div
					className="w-full flex items-center justify-center gap-3 py-3 bg-[#0a0a0a] border-b border-[#222]"
					variants={navVariants}
					animate={hidden ? "hidden" : "vissible"}>
					<div className="xm:hidden sm:hidden">
						<h1 className="text-[#888] text-[18px] font-normal leading-tight">
							{companyInfo.tagline} - Let's build something extraordinary.
						</h1>
					</div>
					<div className="flex gap-2 items-center">
						<Link href="/contact">
							<button className="text-[#00ff88] text-[16px] leading-tight font-normal hover:text-white transition-colors">
								Get in touch
							</button>
						</Link>
						<Image
							src={arrowRightWhite}
							alt="arrowRightWhite"
							width={20}
							height={20}
							className="text-[#00ff88]"
						/>
					</div>
				</motion.div>
				<motion.div
					className="w-full flex items-center justify-between gap-2 padding-x py-3 backdrop-blur-md bg-[#0a0a0a]/80 border-b border-[#222]"
					animate={hidden ? { y: -48 } : { y: 0 }}
					transition={{ duration: 0.5, ease: "easeInOut" }}>
					<div className="flex items-center gap-3">
						<Image
							src={logoLight}
							alt="Zehanx Technologies Logo"
							width={45}
							height={45}
							className="rounded-sm"
						/>
						<span className="text-white font-mono text-lg font-semibold hidden md:block">
							{companyInfo.name}
						</span>
					</div>
					<div className="flex items-center gap-4 xm:hidden sm:hidden">
						{navigationItems.map((item) => (
							<Link
								href={item.href}
								key={item.id}
								className="text-[18px] font-normal leading-tight text-[#888] hover:text-[#00ff88] transition-colors">
								{item.title}
							</Link>
						))}
						<Link href="/contact">
							<Button
								className="text-[#0a0a0a] bg-[#00ff88] px-4 py-2 font-mono font-semibold hover:bg-[#00cc6a] transition-colors"
								title="Start Project"
							/>
						</Link>
					</div>
				</motion.div>
			</motion.div>
			<div className="fixed w-full top-0 z-50 hidden xm:block sm:block">
				<MobileMenu />
			</div>
		</>
	);
}
