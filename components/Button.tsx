import Link from "next/link";
import { TbuttonProps } from "@/types";

export default function Button({ className, title, href }: TbuttonProps) {
	if (href) {
		return (
			<Link href={href}>
				<div
					className={`text-[18px] leading-tight rounded-[8px] font-normal ${className}`}>
					{title}
				</div>
			</Link>
		);
	}
	return (
		<button
			className={`text-[18px] leading-tight rounded-[8px] font-normal ${className}`}>
			{title}
		</button>
	);
}
