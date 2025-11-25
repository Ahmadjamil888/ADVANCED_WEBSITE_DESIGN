import Link from "next/link";
import Image from "next/image";
import { logo } from "@/public";
import { footerItems, footerSocialsItems, contactInfo } from "@/constants";

export default function Footer() {
	return (
		<div className="w-full bg-black py-10 padding-x">
			<div className="w-full flex items-center justify-center flex-col gap-7">
				<div>
					<Image
						src={logo}
						alt="Zehanx Technologies Logo"
						width={40}
						height={40}
					/>
				</div>
				<div className="text-center">
					<h3 className="text-white font-bold text-lg">{contactInfo.company}</h3>
					<p className="text-[#BCBCBC] text-sm italic">{contactInfo.slogan}</p>
				</div>
				<div className="flex items-center gap-4  xm:flex-col sm:flex-col">
					{footerItems.map((item) => (
						<Link
							href={item.href}
							key={item.id}
							className="paragraph font-normal leading-tight text-[#BCBCBC]">
							{item.title}
						</Link>
					))}
				</div>
				<div className="flex items-center gap-4 xm:flex-col sm:flex-col text-center">
					<a href={`mailto:${contactInfo.email}`} className="text-[#BCBCBC] paragraph font-normal hover:text-white">
						{contactInfo.email}
					</a>
					<span className="text-[#BCBCBC] xm:hidden sm:hidden">•</span>
					<a href={`tel:${contactInfo.phone}`} className="text-[#BCBCBC] paragraph font-normal hover:text-white">
						{contactInfo.phone}
					</a>
				</div>
				<div className="flex items-center gap-4">
					{footerSocialsItems.map((item) => (
						<Link
							href={item.href}
							key={item.id}>
							<Image
								src={item.src}
								alt="social"
								width={30}
								height={30}
							/>
						</Link>
					))}
				</div>
				<div className="flex items-center">
					<p className="text-[#BCBCBC] paragraph font-normal">
						© 2025 {contactInfo.company}. All rights reserved.
					</p>
				</div>
			</div>
		</div>
	);
}
