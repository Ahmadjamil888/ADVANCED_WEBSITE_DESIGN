import "@/styles/globals.css";
import type { Metadata } from "next";
export const metadata: Metadata = {
	title: "Zehanx Technologies | AI, ML & Software Development",
	description: "Zehanx Technologies - From concepts to reality. Specializing in AI, ML, Data Science, Web Development, and Software Development.",
	icons: {
		icon: "/unnamed.png",
	},
};
export default function RootLayout({
	children,
}: {
	children: React.ReactNode;
}) {
	return (
		<html lang="en">
			<body>{children}</body>
		</html>
	);
}
