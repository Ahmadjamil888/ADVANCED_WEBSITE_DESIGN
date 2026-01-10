"use client";
import Links from "./Links";
import Link from "next/link";
import { navigationItems } from "@/constants";
import { motion } from "framer-motion";
import React, { useState } from "react";
import { curve, menuSlide } from "@/motion";
import { usePathname } from "next/navigation";

export default function Navbar() {
	const pathname = usePathname();
	const [selectedIndicator, setSelectedIndicator] = useState(pathname);

	return (
		<motion.div
			variants={menuSlide}
			initial="initial"
			animate="enter"
			exit="exit"
			className="h-screen bg-[#292929] fixed right-0 top-0 text-white">
			<div className="box-border h-full z-[999] relative py-[30px] px-[60px]  sm:px-[40px] xm:px-[40px] flex flex-col justify-between">
				<div
					onMouseLeave={() => {
						setSelectedIndicator(pathname);
					}}
					className="flex flex-col text-[50px] md:text-[45px] sm:text-[40px] xm:text-[30px] mt-[60px]">
					<div className="text-[#999999] border-b-[1px] border-[#999999] uppercase text-[12px] mb-[20px]">
						<p>Navigation</p>
					</div>
					{navigationItems.map((data, index) => {
						if (data.title === "Products") {
							return (
								<div key={index} className="flex flex-col">
									<Links
										className="text-white"
										data={{ ...data, index }}
										isActive={selectedIndicator == data.href}
										setSelectedIndicator={setSelectedIndicator}></Links>
									<div className="ml-8 mt-2 text-[30px] md:text-[25px] sm:text-[20px] xm:text-[18px]">
										<div className="text-[#999999] uppercase text-[10px] mb-[10px]">
											<p>our products</p>
										</div>
										<Link
											href="https://daakkhana.up.railway.app/"
											target="_blank"
											rel="noopener noreferrer"
											className="block py-2 text-white hover:text-[#cccccc] transition-colors cursor-pointer z-[1000] relative">
											Daak Khana - Courier Marketplace
										</Link>
										<Link
											href="https://vector-e55x.vercel.app"
											target="_blank"
											rel="noopener noreferrer"
											className="block py-2 text-white hover:text-[#cccccc] transition-colors cursor-pointer z-[1000] relative">
											Vector - AI Workspace for Data Scientists
										</Link>
									</div>
								</div>
							);
						}
						return (
							<Links
								className="text-white"
								key={index}
								data={{ ...data, index }}
								isActive={selectedIndicator == data.href}
								setSelectedIndicator={setSelectedIndicator}></Links>
						);
					})}
				</div>
			</div>
			<svg className="absolute top-0 left-[-99px] w-[100px] h-full fill-[#292929] stroke-none">
				<motion.path
					variants={curve}
					initial="initial"
					animate="enter"
					exit="exit"
				/>
			</svg>
		</motion.div>
	);
}
