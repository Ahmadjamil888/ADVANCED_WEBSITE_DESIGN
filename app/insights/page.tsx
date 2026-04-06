import Navbar from "../components/Navbar";
import Footer from "../components/Footer";
import Insights from "../sections/Insights";

export const metadata = {
  title: "Insights | zehanx Technologies",
  description: "Discover zehanx Technologies insights and statistics. 6+ years of experience, 100+ projects delivered, 50+ happy clients, 24/7 support.",
};

export default function InsightsPage() {
  return (
    <main className="relative min-h-screen bg-black pt-20">
      <Navbar />
      <Insights />
      <Footer />
    </main>
  );
}
