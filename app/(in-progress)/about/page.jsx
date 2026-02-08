import { InProgress } from '@/components';
import { Transition } from '@/layout';

/** @type {import('next').Metadata} */
export const metadata = {
  title: 'About',
  description:
    'Learn about ZehanX Technologies - A research-oriented company specializing in Artificial Intelligence, Machine Learning, Deep Learning, tokenization, signal processing, and cybersecurity solutions.',
};

export default function About() {
  return (
    <Transition>
      <InProgress>About Page</InProgress>
    </Transition>
  );
}
