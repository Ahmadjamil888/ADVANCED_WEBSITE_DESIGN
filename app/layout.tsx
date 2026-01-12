import "@/styles/globals.css";
import type { Metadata } from "next";
export const metadata: Metadata = {
	title: "Zehanx Technologies | B2B Software Solutions, AI/ML, Cybersecurity & Enterprise Systems",
	description: "Zehanx Technologies - Empowering businesses with comprehensive B2B software solutions in AI/ML, cybersecurity, and enterprise systems.",
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
