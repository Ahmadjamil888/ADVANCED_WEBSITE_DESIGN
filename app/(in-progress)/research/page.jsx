import { InProgress } from '@/components';
import { Transition } from '@/layout';

/** @type {import('next').Metadata} */
export const metadata = {
  title: 'Research',
  description:
    'Research initiatives at ZehanX Technologies in AI/ML/DL, tokenization, signal processing, and cybersecurity. Explore our cutting-edge research projects and innovations.',
};

export default function Research() {
  return (
    <Transition>
      <InProgress>Research Page</InProgress>
    </Transition>
  );
}
