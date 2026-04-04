import "@/styles/globals.css";
import type { Metadata } from "next";
export const metadata: Metadata = {
	title: "Zehanx Technologies | Machine Learning & Software Solutions",
	description: "Zehanx Technologies provides cutting-edge Machine Learning solutions, custom software development, and digital transformation services.",
};
export default function RootLayout({
	children,
}: {
	children: React.ReactNode;
}) {
	return (
		<html lang="en">
			<body className="bg-[#0a0a0a] text-white">{children}</body>
		</html>
	);
}
