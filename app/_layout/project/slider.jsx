'use client';

import Image from 'next/image';

import { Center } from '@/components';

/**
 * @param {Object} props
 * @param {'image' | 'video'} props.type
 * @param {string} props.source
 */
export function ProjectSlider({ type, source }) {
  const content =
    type === 'image' ? (
      <Image
        src={source}
        className='object-cover'
        fill={true}
        alt='project items'
      />
    ) : (
      <video
        src={source}
        loop={true}
        controls={false}
        muted={true}
        autoPlay
        width='100%'
        height='100%'
        className='!static !bg-transparent'
      />
    );

  return (
    <Center
      className='relative w-1/4 rounded'
      style={
        {
          minWidth: '150px',
          height: '20vw',
        }
      }
    >
      {content}
    </Center>
  );
}
