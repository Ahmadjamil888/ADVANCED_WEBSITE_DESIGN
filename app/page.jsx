import {
  Contact,
  Description,
  Header,
  Navbar,
  Project,
  Thumbnail,
  Transition,
} from '@/layout';

/** @type {import('next').Metadata} */
export const metadata = {
  title: 'ZehanX Technologies - Research in AI/ML/DL, Security & Emerging Tech',
  description:
    'ZehanX Technologies - A research-oriented company specializing in Artificial Intelligence, Machine Learning, Deep Learning, tokenization, signal processing, and cybersecurity solutions.',
};

export default function Home() {
  return (
    <Transition>
      <Navbar />
      <Header />
      <main>
        <Description />
        <Thumbnail />
        <Project />
      </main>
      <Contact />
    </Transition>
  );
}
