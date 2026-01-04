import Image from "next/image";
import { motion } from "framer-motion";
import { Button, Heading } from "@/components";
import { ArrowRight, spring, star } from "@/public";
import { imageAnimation, textAnimation } from "@/motion";
import { contactInfo } from "@/constants";

export default function CallToAction() {
	const phares1 = ["Ready to transform your ideas?"];
	const phares2 = [
		"Get in touch with our team today. We're ready to help",
		"you bring your concepts to reality with innovative",
		"solutions.",
	];
	const phares3 = [
		"Get in touch with our team today. We're ready to help you bring your concepts to reality with innovative solutions.",
	];
	return (
		<div className="w-full padding-x py-10 relative bg-gradient-to-b from-white to-[#d2dcff]">
			<div className="w-full flex items-center gap-5">
				<motion.div
					variants={imageAnimation}
					initial="initial"
					whileInView="enter"
					viewport={{ once: true }}
					className="xm:hidden sm:hidden">
					<Image
						src={star}
						alt="star-hero-img"
						width={400}
						height={400}
					/>
				</motion.div>
				<div className="w-full flex items-center flex-col gap-3">
					<div>
						<Heading
							classname="heading font-bold xm:text-center sm:text-center"
							title={phares1}
						/>
					</div>
					<div>
						<Heading
							classname="paragraph font-normal text-center block xm:hidden sm:hidden"
							title={phares2}
						/>
						<Heading
							classname="paragraph font-normal text-center hidden xm:block sm:block"
							title={phares3}
						/>
					</div>
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
				<motion.div
					className="xm:hidden sm:hidden"
					variants={imageAnimation}
					initial="initial"
					whileInView="enter"
					viewport={{ once: true }}>
					<Image
						src={spring}
						alt="spring-hero-img"
						width={400}
						height={400}
					/>
				</motion.div>
			</div>
		</div>
	);
}
