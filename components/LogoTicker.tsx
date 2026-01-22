import Image from "next/image";
import LogoMarquee from "./LogoMarquee";
import { logoMarqueeItems } from "@/constants";

export default function LogoTicker() {
	return (
		<div className="w-full [mask-image:linear-gradient(to_left,transparent,black_25%,black_75%,transparent)]">
			<LogoMarquee baseVelocity={1.5}>
				{logoMarqueeItems.map((item) => (
					<div
						className={`w-[250px] flex ${item.id == 5 && "mr-14"}`}
						key={item.id}>
						<Image
							src={item.src}
							alt="logo"
						/>
					</div>
				))}
			</LogoMarquee>
		</div>
	);
}
