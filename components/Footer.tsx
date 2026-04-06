import Link from "next/link";
import Image from "next/image";
import { logoDark } from "@/public";
import { footerItems, footerSocialsItems, contactInfo, companyInfo } from "@/constants";
import { Mail, Phone } from "lucide-react";

export default function Footer() {
	return (
		<div className="w-full bg-[#0a0a0a] py-10 padding-x border-t border-[#222]">
			<div className="w-full flex items-center justify-center flex-col gap-7">
				<div className="flex items-center gap-3">
					<Image
						src={logoDark}
						alt="Zehanx Technologies Logo"
						width={50}
						height={50}
						className="rounded-sm"
					/>
					<span className="text-white font-mono text-xl font-semibold">
						{companyInfo.name}
					</span>
				</div>
				<p className="text-[#888] text-center max-w-md">
					{companyInfo.description}
				</p>
				<div className="flex items-center gap-6 xm:flex-col sm:flex-col">
					{footerItems.map((item) => (
						<Link
							href={item.href}
							key={item.id}
							className="paragraph font-normal leading-tight text-[#888] hover:text-[#00ff88] transition-colors">
							{item.title}
						</Link>
					))}
				</div>
				<div className="flex items-center gap-6">
					<a 
						href={`mailto:${contactInfo.email}`}
						className="flex items-center gap-2 text-[#888] hover:text-[#00ff88] transition-colors">
						<Mail className="w-4 h-4" />
						<span className="font-mono text-sm">{contactInfo.email}</span>
					</a>
					<a 
						href={`https://wa.me/${contactInfo.whatsapp}`}
						className="flex items-center gap-2 text-[#888] hover:text-[#00ff88] transition-colors">
						<Phone className="w-4 h-4" />
						<span className="font-mono text-sm">WhatsApp: {contactInfo.whatsapp}</span>
					</a>
				</div>
				<div className="flex items-center gap-4">
					{footerSocialsItems.map((item) => (
						<Link
							href={item.href}
							key={item.id}
							className="text-[#888] hover:text-[#00ff88] transition-colors"
							aria-label={item.label}>
							<Image
								src={item.src}
								alt={item.label}
								width={24}
								height={24}
								className="opacity-60 hover:opacity-100"
							/>
						</Link>
					))}
				</div>
				<div className="flex items-center flex-col gap-2">
					<p className="text-[#666] paragraph font-normal font-mono text-sm">
						© 2024 {companyInfo.name}. All rights reserved.
					</p>
					<p className="text-[#444] text-xs font-mono">
						Built with code, powered by innovation.
					</p>
				</div>
			</div>
		</div>
	);
}
