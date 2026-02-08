import plugin from 'tailwindcss/plugin';

/** @type {import('tailwindcss').Config} */
export const tailwindPlugin = plugin(
  //? Add CSS variable definitions to the base layer
  function ({ addBase }) {
    addBase({
      ':root': {
        '--background': '210 10% 5%',
        '--foreground': '210 5% 90%',
        '--card': '210 8% 10%',
        '--card-foreground': '210 5% 85%',
        '--popover': '210 10% 5%',
        '--popover-foreground': '210 5% 90%',
        '--primary': '200 80% 50%',
        '--primary-foreground': '0 0% 98%',
        '--secondary': '210 15% 25%',
        '--secondary-foreground': '210 5% 85%',
        '--muted': '210 8% 15%',
        '--muted-foreground': '210 5% 65%',
        '--accent': '210 10% 15%',
        '--accent-foreground': '210 5% 85%',
        '--destructive': '0 70% 40%',
        '--destructive-foreground': '0 0% 98%',
        '--border': '210 10% 18%',
        '--input': '210 10% 18%',
        '--ring': '200 80% 50%',
        '--radius': '0.5rem',
      },
    });
  },
  //? Extend the Tailwind theme with 'themable' utilities
  {
    theme: {
      container: {
        center: true,
        padding: '2rem',
        screens: {
          '2xl': '1400px',
        },
      },
      extend: {
        colors: {
          border: 'hsl(var(--border))',
          input: 'hsl(var(--input))',
          ring: 'hsl(var(--ring))',
          background: 'hsl(var(--background))',
          foreground: 'hsl(var(--foreground))',
          primary: {
            DEFAULT: 'hsl(var(--primary))',
            foreground: 'hsl(var(--primary-foreground))',
          },
          secondary: {
            DEFAULT: 'hsl(var(--secondary))',
            foreground: 'hsl(var(--secondary-foreground))',
          },
          destructive: {
            DEFAULT: 'hsl(var(--destructive))',
            foreground: 'hsl(var(--destructive-foreground))',
          },
          muted: {
            DEFAULT: 'hsl(var(--muted))',
            foreground: 'hsl(var(--muted-foreground))',
          },
          accent: {
            DEFAULT: 'hsl(var(--accent))',
            foreground: 'hsl(var(--accent-foreground))',
          },
          popover: {
            DEFAULT: 'hsl(var(--popover))',
            foreground: 'hsl(var(--popover-foreground))',
          },
          card: {
            DEFAULT: 'hsl(var(--card))',
            foreground: 'hsl(var(--card-foreground))',
          },
        },
        borderRadius: {
          lg: 'var(--radius)',
          md: 'calc(var(--radius) - 2px)',
          sm: 'calc(var(--radius) - 4px)',
        },
        keyframes: {
          'accordion-down': {
            from: { height: 0 },
            to: { height: 'var(--radix-accordion-content-height)' },
          },
          'accordion-up': {
            from: { height: 'var(--radix-accordion-content-height)' },
            to: { height: 0 },
          },
        },
        animation: {
          'accordion-down': 'accordion-down 0.2s ease-out',
          'accordion-up': 'accordion-up 0.2s ease-out',
        },
        fontFamily: {
          neue_montreal: ['var(--font-neue-montreal)'],
        },
        transitionDuration: {
          1500: '1500ms',
          2000: '2000ms',
          2500: '2500ms',
          3000: '3000ms',
        },
        transitionTimingFunction: {
          'in-expo': 'cubic-bezier(0.1, 0, 0.3, 1)',
        },
      },
    },
  },
);
