import LogoMarquee from "./LogoMarquee";
import { logoMarqueeItems } from "@/constants";

export default function LogoTicker() {
	return (
		<div className="w-full py-10 bg-[#0a0a0a] [mask-image:linear-gradient(to_left,transparent,black_25%,black_75%,transparent)]">
			<LogoMarquee baseVelocity={1.5}>
				{logoMarqueeItems.map((item) => (
					<div
						className={`flex items-center justify-center px-8 py-4 mx-4 rounded-xl bg-[#111] border border-[#222] hover:border-[#00ff88]/30 transition-colors ${item.id == 6 && "mr-14"}`}
						key={item.id}>
						<span className="text-[#00ff88] font-mono text-lg font-semibold whitespace-nowrap">
							{item.name}
						</span>
					</div>
				))}
			</LogoMarquee>
		</div>
	);
}
