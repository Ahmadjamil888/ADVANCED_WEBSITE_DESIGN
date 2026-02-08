import { InProgress } from '@/components';
import { Transition } from '@/layout';

/** @type {import('next').Metadata} */
export const metadata = {
  title: 'Contact',
  description:
    'Contact ZehanX Technologies - Reach out to our research team for collaborations, partnerships, and inquiries about our work in AI/ML/DL, tokenization, signals, and cybersecurity.',
};

export default function Contact() {
  return (
    <Transition>
      <InProgress>Contact Page</InProgress>
    </Transition>
  );
}
