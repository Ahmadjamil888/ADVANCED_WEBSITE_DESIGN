'use client';

import AdSense from './AdSense';

export default function BlogAdBanner() {
  return (
    <div className="my-8 flex justify-center">
      <div className="max-w-4xl w-full">
        <div className="text-xs text-muted-foreground text-center mb-2">Advertisement</div>
        <AdSense
          adSlot="1234567890" // Replace with your actual ad slot ID
          adFormat="rectangle"
          style={{
            display: 'block',
            width: '100%',
            height: '250px',
          }}
          className="border border-border/20 rounded-lg"
        />
      </div>
    </div>
  );
}