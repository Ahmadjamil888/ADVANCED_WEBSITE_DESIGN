import { motion } from "framer-motion";
import { servicesItems, companyInfo } from "@/constants";
import { Heading, Button } from "@/components";
import { Brain, Code, Database, Sparkles } from "lucide-react";

const iconMap: { [key: string]: React.ReactNode } = {
	brain: <Brain className="w-8 h-8 text-[#00ff88]" />,
	code: <Code className="w-8 h-8 text-[#00ff88]" />,
	database: <Database className="w-8 h-8 text-[#00ff88]" />,
};

export default function Services() {
	const phares = ["Our Services"];
	const phares1 = [
		"From Machine Learning models to full-stack applications,",
		"we deliver solutions that drive real business impact.",
	];
	return (
		<div className="w-full padding-x py-20 bg-[#0a0a0a] xm:py-10 sm:py-10">
			<div className="w-full flex flex-col gap-10">
				<div className="w-full flex items-center flex-col gap-3">
					<motion.div
						initial={{ opacity: 0, y: 20 }}
						whileInView={{ opacity: 1, y: 0 }}
						transition={{ duration: 0.5 }}
						viewport={{ once: true }}
						className="flex items-center gap-2 px-4 py-2 rounded-full border border-[#00ff88]/30 bg-[#00ff88]/10">
						<Sparkles className="w-4 h-4 text-[#00ff88]" />
						<span className="text-[#00ff88] font-mono text-sm">What We Do</span>
					</motion.div>
					<div>
						<Heading
							classname="heading font-bold text-white text-center"
							title={phares}
						/>
					</div>
					<div>
						<Heading
							classname="paragraph text-[#888] text-center"
							title={phares1}
						/>
					</div>
				</div>
				<div className="w-full flex justify-center items-stretch gap-8 xm:flex-col sm:flex-col">
					{servicesItems.map((item) => (
						<motion.div
							key={item.id}
							initial={{ opacity: 0, y: 30 }}
							whileInView={{ opacity: 1, y: 0 }}
							transition={{ duration: 0.5, delay: item.id * 0.1 }}
							viewport={{ once: true }}
							className={`w-full flex flex-col gap-8 rounded-2xl border p-10 xm:p-8 sm:p-8 transition-all duration-300 hover:border-[#00ff88]/50 ${
								item.popular
									? "bg-gradient-to-b from-[#1a1a2e] to-[#0a0a0a] border-[#00ff88]/30"
									: "bg-[#111] border-[#222]"
							}`}>
							<div className="w-full flex flex-col gap-6">
								<div className="flex items-center justify-between">
									<div className="flex items-center gap-4">
										<div className="p-3 rounded-xl bg-[#00ff88]/10 border border-[#00ff88]/20">
											{iconMap[item.icon]}
										</div>
										<div>
											<h3 className="text-white text-[24px] font-bold leading-tight">
												{item.title}
											</h3>
											<p className="text-[#00ff88] text-sm font-mono">
												{item.subtitle}
											</p>
										</div>
									</div>
									{item.popular && (
										<motion.span
											className="border border-[#00ff88]/30 rounded-md text-sm px-3 py-1 bg-[#00ff88]/10 text-[#00ff88] font-mono"
											animate={{ opacity: [0.7, 1, 0.7] }}
											transition={{
												duration: 2,
												repeat: Infinity,
												ease: "easeInOut",
											}}>
											Popular
										</motion.span>
									)}
								</div>
								<p className="text-[#888] text-[16px] leading-relaxed">
									{item.description}
								</p>
								<Button
									title="Learn More"
									className={`w-full py-3 rounded-lg font-semibold font-mono transition-all ${
										item.popular
											? "bg-[#00ff88] text-[#0a0a0a] hover:bg-[#00cc6a]"
											: "bg-transparent border border-[#333] text-white hover:border-[#00ff88] hover:text-[#00ff88]"
									}`}
								/>
							</div>
							<div className="w-full flex flex-col gap-4 pt-6 border-t border-[#222]">
								<p className="text-[#666] text-sm font-mono mb-2">Features:</p>
								{item.features.map((feature, index) => (
									<div
										className="w-full flex gap-3 items-center"
										key={index}>
										<span className="text-[#00ff88] font-mono text-sm">›</span>
										<p className="text-[#aaa] text-[14px] leading-tight">
											{feature}
										</p>
									</div>
								))}
							</div>
						</motion.div>
					))}
				</div>
			</div>
		</div>
	);
}
