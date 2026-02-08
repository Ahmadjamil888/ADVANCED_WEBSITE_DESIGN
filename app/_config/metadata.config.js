/** @type {import('next').Metadata} */
export const rootMetadata = {
  metadataBase: new URL('https://www.zehanx.com/'),
  title: {
    template: '%s | ZehanX Technologies',
    default: 'ZehanX Technologies - Research in AI/ML/DL, Tokenization, Signals, Cybersecurity',
  },
  description:
    'ZehanX Technologies - A research-oriented company specializing in Artificial Intelligence, Machine Learning, Deep Learning, tokenization, signal processing, and cybersecurity solutions.',
  generator: 'ZehanX Technologies',
  applicationName: 'ZehanX Technologies',
  referrer: 'origin-when-cross-origin',
  keywords: ['AI', 'Machine Learning', 'Infrastructure', 'API'],
  authors: [
    { name: 'ZehanX Technologies', url: 'https://www.zehanx.com' },
  ],
  creator: 'ZehanX Technologies',
  publisher: 'ZehanX Technologies',
  twitter: {
    card: 'summary_large_image',
    title: 'ZehanX Technologies',
    description:
      'ZehanX Technologies - A research-oriented company specializing in Artificial Intelligence, Machine Learning, Deep Learning, tokenization, signal processing, and cybersecurity solutions.',
    siteId: '1467726470533754880',
    creator: '@ZehanXTech',
    creatorId: '1467726470533754880',
    images: {
      url: 'https://www.zehanx.com/logo.png',
      alt: 'ZehanX Technologies Logo',
    },
  },
  robots: {
    index: false,
    follow: true,
    nocache: true,
    googleBot: {
      index: true,
      follow: false,
      noimageindex: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
};
