import Image from "next/image";
import { motion } from "framer-motion";
import { Button, Heading } from "@/components";
import { ArrowRight, spring, star } from "@/public";
import { imageAnimation, textAnimation } from "@/motion";
import { contactInfo } from "@/constants";
import { Mail, MessageCircle } from "lucide-react";

export default function CallToAction() {
	const phares1 = ["Ready to Build Something?"];
	const phares2 = [
		"Let's discuss how Zehanx Technologies can transform",
		"your ideas into intelligent, scalable solutions.",
	];
	const phares3 = [
		"Ready to build? Contact us today and let's create something extraordinary together.",
	];
	return (
		<div className="w-full padding-x py-20 relative bg-gradient-to-b from-[#0a0a0a] via-[#1a1a2e] to-[#0a0a0a]">
			<div className="w-full flex items-center gap-5">
				<motion.div
					variants={imageAnimation}
					initial="initial"
					whileInView="enter"
					viewport={{ once: true }}
					className="xm:hidden sm:hidden">
					<Image
						src={star}
						alt="AI Innovation"
						width={400}
						height={400}
						className="opacity-60"
					/>
				</motion.div>
				<div className="w-full flex items-center flex-col gap-6">
					<motion.div
						initial={{ opacity: 0, y: 20 }}
						whileInView={{ opacity: 1, y: 0 }}
						transition={{ duration: 0.5 }}
						viewport={{ once: true }}
						className="flex items-center gap-2 px-4 py-2 rounded-full border border-[#00ff88]/30 bg-[#00ff88]/10">
						<MessageCircle className="w-4 h-4 text-[#00ff88]" />
						<span className="text-[#00ff88] font-mono text-sm">Get In Touch</span>
					</motion.div>
					<div>
						<Heading
							classname="heading font-bold text-white xm:text-center sm:text-center"
							title={phares1}
						/>
					</div>
					<div>
						<Heading
							classname="paragraph font-normal text-[#888] text-center block xm:hidden sm:hidden"
							title={phares2}
						/>
						<Heading
							classname="paragraph font-normal text-[#888] text-center hidden xm:block sm:block"
							title={phares3}
						/>
					</div>
					<motion.div
						variants={textAnimation}
						initial="initial"
						whileInView="enter"
						viewport={{ once: true }}
						className="flex flex-col sm:flex-row gap-4 items-center mt-3 overflow-hidden">
						<a 
							href={`mailto:${contactInfo.email}`}
							className="flex items-center gap-2 text-[#0a0a0a] bg-[#00ff88] py-3 px-6 rounded-lg font-mono font-semibold hover:bg-[#00cc6a] transition-colors">
							<Mail className="w-4 h-4" />
							<span>{contactInfo.email}</span>
						</a>
						<a 
							href={`https://wa.me/${contactInfo.whatsapp}`}
							className="flex items-center gap-2 text-[#00ff88] border border-[#00ff88]/30 py-3 px-6 rounded-lg font-mono hover:bg-[#00ff88]/10 transition-colors">
							<MessageCircle className="w-4 h-4" />
							<span>WhatsApp: {contactInfo.whatsapp}</span>
						</a>
					</motion.div>
				</div>
				<motion.div
					className="xm:hidden sm:hidden"
					variants={imageAnimation}
					initial="initial"
					whileInView="enter"
					viewport={{ once: true }}>
					<Image
						src={spring}
						alt="Software Solutions"
						width={400}
						height={400}
						className="opacity-60"
					/>
				</motion.div>
			</div>
		</div>
	);
}
