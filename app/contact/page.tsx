import Navbar from "../components/Navbar";
import Footer from "../components/Footer";
import Contact from "../sections/Contact";

export const metadata = {
  title: "Contact | zehanx Technologies",
  description: "Get in touch with zehanx Technologies. Ready to transform your ideas into reality? Contact us for Web, AI, ML, and App development solutions.",
};

export default function ContactPage() {
  return (
    <main className="relative min-h-screen bg-black pt-20">
      <Navbar />
      <Contact />
      <Footer />
    </main>
  );
}
