import Navbar from "../components/Navbar";
import Footer from "../components/Footer";
import Services from "../sections/Services";

export const metadata = {
  title: "Services | zehanx Technologies",
  description: "Explore our comprehensive technology services - Web Development, AI, Machine Learning, Deep Learning, Neural Networks, Software Development, and Mobile App Development.",
};

export default function ServicesPage() {
  return (
    <main className="relative min-h-screen bg-black pt-20">
      <Navbar />
      <Services />
      <Footer />
    </main>
  );
}
