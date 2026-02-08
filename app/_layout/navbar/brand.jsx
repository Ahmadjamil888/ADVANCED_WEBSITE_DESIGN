'use client';

import Image from 'next/image';

export function NavbarBrand() {
  return (
    <div className='group flex cursor-pointer pb-5'>
      <div className='transition-transform duration-500 ease-in-expo group-hover:rotate-[360deg]'>
        <Image 
          src='/Screenshot 2026-02-08 230507.png' 
          alt='ZehanX Technologies Logo'
          width={40}
          height={40}
          className='rounded-full object-contain'
        />
      </div>

      <div className='relative ms-2 flex overflow-hidden whitespace-nowrap transition-all duration-500 ease-in-expo group-hover:pe-8'>
        <h5 className='transition-transform duration-500 ease-in-expo group-hover:-translate-x-full'>
          ZehanX
        </h5>
        <h5 className='ps-1 transition-transform duration-500 ease-in-expo group-hover:-translate-x-14'>
          Technologies
        </h5>
      </div>
    </div>
  );
}
