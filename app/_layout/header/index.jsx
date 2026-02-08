'use client';

import { motion } from 'framer-motion';
import { MoveDownRight } from 'lucide-react';
import Image from 'next/image';

import { ParallaxSlider } from '@/components';

import { slideUp } from './variants';

export function Header() {
  return (
    <motion.header
      className='relative h-screen overflow-hidden bg-secondary-foreground text-background'
      variants={slideUp}
      initial='initial'
      animate='enter'
    >
      <div className="absolute inset-0">
        <Image
          src='/Gemini_Generated_Image_exhwpmexhwpmexhw.png'
          className='object-cover w-full h-full md:scale-125 md:object-contain'
          fill={true}
          sizes='100vw'
          alt='ZehanX Technologies Background'
        />
        <div className="absolute bottom-0 right-0 w-1/2 h-1/2 bg-black opacity-50"></div>
      </div>

      <div className='relative flex h-full flex-col justify-end gap-2 md:flex-col-reverse md:justify-normal'>
        <div className='select-none'>
          <h1 className='text-[max(9em,15vw)]'>
            <ParallaxSlider repeat={4} baseVelocity={2}>
              <span className='pe-12'>
                ZEHANX
                <span className='spacer'>—</span>
              </span>
            </ParallaxSlider>
          </h1>
        </div>

        <div className='md:ml-auto'>
          <div className='mx-10 max-md:my-12 md:mx-36'>
            <div className='mb-4 md:mb-20'>
              <MoveDownRight size={28} strokeWidth={1.25} />
            </div>

            <h4 className='text-[clamp(1.55em,2.5vw,2.75em)]'>
              <span className='block'>Research in</span>
              <span className='block'>AI/ML/DL & Security</span>
            </h4>
          </div>
        </div>
      </div>
    </motion.header>
  );
}
