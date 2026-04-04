import Image from "next/image";
import { Heading } from "@/components";
import { motion } from "framer-motion";
import { textAnimation } from "@/motion";
import { testimonials } from "@/constants";
import { Quote } from "lucide-react";

export default function Testimonials() {
	const phares = ["Client Success Stories"];
	const phares1 = [
		"See how we've helped businesses transform",
		"with intelligent technology solutions.",
	];
	return (
		<div className="w-full flex flex-col items-center padding-x py-20 gap-20 bg-gradient-to-b from-[#0a0a0a] to-[#111] xm:gap-10 sm:gap-10">
			<div className="flex flex-col items-center gap-5">
				<motion.div
					initial={{ opacity: 0, y: 20 }}
					whileInView={{ opacity: 1, y: 0 }}
					transition={{ duration: 0.5 }}
					viewport={{ once: true }}
					className="flex items-center gap-2 px-4 py-2 rounded-full border border-[#00ff88]/30 bg-[#00ff88]/10">
					<Quote className="w-4 h-4 text-[#00ff88]" />
					<span className="text-[#00ff88] font-mono text-sm">Testimonials</span>
				</motion.div>
				<div>
					<Heading
						classname="heading font-bold text-white xm:text-center sm:text-center"
						title={phares}
					/>
				</div>
				<div>
					<Heading
						classname="paragraph text-[#888] text-center font-normal xm:text-center sm:text-center"
						title={phares1}
					/>
				</div>
			</div>
			<motion.div className="w-full flex gap-5 xm:flex-col sm:flex-col xm:w-full py-10 sm:w-full overflow-hidden h-[750px] [mask-image:linear-gradient(to_bottom,transparent,black_25%,black_75%,transparent)] mt-10">
				<motion.div
					animate={{ y: "-50%" }}
					transition={{
						repeat: Infinity,
						repeatType: "loop",
						ease: "linear",
						duration: 13,
					}}
					className="w-1/3 flex flex-col h-fit xm:w-full sm:w-full">
					{[...testimonials.slice(0, 3), ...testimonials.slice(0, 3)].map(
						(item) => (
							<div
								className="flex flex-col gap-5"
								key={item.id}>
								<div className="p-10 mb-5 border border-[#222] rounded-[20px] bg-[#111] flex flex-col gap-5 hover:border-[#00ff88]/30 transition-colors">
									<Quote className="w-8 h-8 text-[#00ff88]/50" />
									<p className="text-[#ccc] font-mono text-lg font-normal leading-relaxed">
										{item.text}
									</p>
									<div className="flex items-center gap-5">
										<Image
											src={item.src}
											alt={item.name}
											width={60}
											height={60}
											className="rounded-full border-2 border-[#00ff88]/30"
										/>
										<div className="flex flex-col">
											<h1 className="text-white font-mono text-lg font-semibold leading-tight">
												{item.name}
											</h1>
											<p className="text-[#00ff88] font-mono text-sm leading-tight">
												{item.username}
											</p>
										</div>
									</div>
								</div>
							</div>
						),
					)}
				</motion.div>
				<motion.div
					className="w-1/3 flex flex-col h-fit xm:hidden sm:hidden"
					animate={{ y: "-50%" }}
					transition={{
						repeat: Infinity,
						repeatType: "loop",
						ease: "linear",
						duration: 15,
					}}>
					{[...testimonials.slice(3, 6), ...testimonials.slice(3, 6)].map(
						(item) => (
							<div
								className="flex flex-col gap-5"
								key={item.id}>
								<div className="p-10 mb-5 border border-[#222] rounded-[20px] bg-[#111] flex flex-col gap-5 hover:border-[#00ff88]/30 transition-colors">
									<Quote className="w-8 h-8 text-[#00ff88]/50" />
									<p className="text-[#ccc] font-mono text-lg font-normal leading-relaxed">
										{item.text}
									</p>
									<div className="flex items-center gap-5">
										<Image
											src={item.src}
											alt={item.name}
											width={60}
											height={60}
											className="rounded-full border-2 border-[#00ff88]/30"
										/>
										<div className="flex flex-col">
											<h1 className="text-white font-mono text-lg font-semibold leading-tight">
												{item.name}
											</h1>
											<p className="text-[#00ff88] font-mono text-sm leading-tight">
												{item.username}
											</p>
										</div>
									</div>
								</div>
							</div>
						),
					)}
				</motion.div>
				<motion.div
					className="w-1/3 flex flex-col h-fit xm:hidden sm:hidden"
					animate={{ y: "-50%" }}
					transition={{
						repeat: Infinity,
						repeatType: "loop",
						ease: "linear",
						duration: 16,
					}}>
					{[...testimonials.slice(6), ...testimonials.slice(6)].map((item) => (
						<div
							className="flex flex-col gap-5"
							key={item.id}>
							<div className="p-10 mb-5 border border-[#222] rounded-[20px] bg-[#111] flex flex-col gap-5 hover:border-[#00ff88]/30 transition-colors">
								<Quote className="w-8 h-8 text-[#00ff88]/50" />
								<p className="text-[#ccc] font-mono text-lg font-normal leading-relaxed">
									{item.text}
								</p>
								<div className="flex items-center gap-5">
									<Image
										src={item.src}
										alt={item.name}
										width={60}
										height={60}
										className="rounded-full border-2 border-[#00ff88]/30"
									/>
									<div className="flex flex-col">
										<h1 className="text-white font-mono text-lg font-semibold leading-tight">
											{item.name}
										</h1>
										<p className="text-[#00ff88] font-mono text-sm leading-tight">
											{item.username}
										</p>
									</div>
								</div>
							</div>
						</div>
					))}
				</motion.div>
			</motion.div>
		</div>
	);
}
