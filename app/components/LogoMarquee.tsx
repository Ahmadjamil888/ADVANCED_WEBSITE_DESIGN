"use client";

// Big tech company logos as SVGs
const logos = [
  {
    name: "Google",
    svg: <svg viewBox="0 0 272 92" fill="currentColor" className="h-5 sm:h-6 w-auto"><path d="M115.75 47.18c0 12.77-9.99 22.18-22.25 22.18s-22.25-9.41-22.25-22.18C71.25 34.32 81.24 25 93.5 25s22.25 9.32 22.25 22.18zm-9.74 0c0-7.98-5.79-13.44-12.51-13.44S80.99 39.2 80.99 47.18c0 7.9 5.79 13.44 12.51 13.44s12.51-5.55 12.51-13.44z"/><path d="M163.75 47.18c0 12.77-9.99 22.18-22.25 22.18s-22.25-9.41-22.25-22.18c0-12.85 9.99-22.18 22.25-22.18s22.25 9.32 22.25 22.18zm-9.74 0c0-7.98-5.79-13.44-12.51-13.44s-12.51 5.46-12.51 13.44c0 7.9 5.79 13.44 12.51 13.44s12.51-5.55 12.51-13.44z"/><path d="M209.75 26.34v39.82c0 16.38-9.66 23.07-21.08 23.07-10.75 0-17.22-7.19-19.66-13.07l8.48-3.53c1.51 3.61 5.21 7.87 11.17 7.87 7.31 0 11.84-4.51 11.84-13v-3.19h-.34c-2.18 2.69-6.38 5.04-11.68 5.04-11.09 0-21.25-9.66-21.25-22.09 0-12.52 10.16-22.26 21.25-22.26 5.29 0 9.49 2.35 11.68 4.96h.34v-3.61h9.25zm-8.56 20.92c0-7.81-5.21-13.52-11.84-13.52-6.72 0-12.35 5.71-12.35 13.52 0 7.73 5.63 13.36 12.35 13.36 6.63 0 11.84-5.63 11.84-13.36z"/><path d="M225 3v65h-9.5V3h9.5z"/><path d="M262.02 54.48l7.56 5.04c-2.44 3.61-8.32 9.83-18.48 9.83-12.6 0-22.01-9.74-22.01-22.18 0-13.19 9.49-22.18 20.92-22.18 11.51 0 17.14 9.16 18.98 14.11l1.01 2.52-29.75 12.28c2.27 4.45 5.8 6.72 10.75 6.72 4.96 0 8.4-2.44 10.92-6.14zm-23.27-7.98l19.82-8.23c-1.09-2.77-4.37-4.7-8.23-4.7-4.95 0-11.84 4.37-11.59 12.93z"/><path d="M35.29 41.41V32H67c.31 1.64.47 3.58.47 5.68 0 7.06-1.93 15.79-8.15 22.01-6.05 6.3-13.78 9.66-24.02 9.66C16.32 69.35.36 53.89.36 34.91.36 15.93 16.32.47 35.3.47c10.5 0 17.98 4.12 23.6 9.49l-6.64 6.64c-4.03-3.78-9.49-6.72-16.97-6.72-13.86 0-24.7 11.17-24.7 25.03 0 13.86 10.84 25.03 24.7 25.03 8.99 0 14.11-3.61 17.39-6.89 2.66-2.66 4.41-6.46 5.1-11.65l-22.49.01z"/></svg>,
  },
  {
    name: "Microsoft",
    svg: <svg viewBox="0 0 23 23" fill="currentColor" className="h-5 sm:h-6 w-auto"><path d="M0 0h11v11H0z"/><path d="M12 0h11v11H12z"/><path d="M0 12h11v11H0z"/><path d="M12 12h11v11H12z"/></svg>,
  },
  {
    name: "Apple",
    svg: <svg viewBox="0 0 814 1000" fill="currentColor" className="h-5 sm:h-6 w-auto"><path d="M788.1 340.9c-5.8 4.5-108.2 62.2-108.2 190.5 0 148.4 130.3 200.9 134.2 202.2-.6 3.2-20.7 71.9-68.7 141.9-42.8 61.6-87.5 123.1-155.5 123.1s-85.5-39.5-164-39.5c-76.5 0-103.7 40.8-165.9 40.8s-105.6-57-155.5-127C46.5 791.2 0 663 0 541.8c0-194.4 126.4-297.5 250.8-297.5 66.1 0 121.2 43.6 162.8 43.6 39.5 0 102.3-46 193.7-46 31.2 0 130.8.9 180.8 99.4zm-234-181.5c31.1-37.2 52.3-88.8 52.3-140.6 0-7.1-.6-14.4-1.8-20.2-50.1 1.9-109.1 33.4-144.7 75.2-28.6 33.2-55.4 85.2-55.4 137.7 0 8.2.8 16.5 2.3 22.4 3.2.5 6.5.6 9.9.6 45.1 0 102-28.6 136.7-75z"/></svg>,
  },
  {
    name: "Amazon",
    svg: <svg viewBox="0 0 120 40" fill="currentColor" className="h-5 sm:h-6 w-auto"><text x="0" y="30" fontFamily="Arial Black, sans-serif" fontSize="28" fontWeight="900" fill="currentColor">amazon</text><path d="M95 25 Q 110 30 118 28" stroke="currentColor" strokeWidth="3" fill="none"/></svg>,
  },
  {
    name: "Meta",
    svg: <svg viewBox="0 0 100 40" fill="currentColor" className="h-5 sm:h-6 w-auto"><text x="0" y="30" fontFamily="Arial, sans-serif" fontSize="26" fontWeight="600" fill="currentColor">Meta</text></svg>,
  },
  {
    name: "Netflix",
    svg: <svg viewBox="0 0 100 40" fill="currentColor" className="h-5 sm:h-6 w-auto"><text x="0" y="30" fontFamily="Arial Black, sans-serif" fontSize="24" fontWeight="900" letterSpacing="-1" fill="currentColor">NETFLIX</text></svg>,
  },
  {
    name: "OpenAI",
    svg: <svg viewBox="0 0 100 40" fill="currentColor" className="h-5 sm:h-6 w-auto"><text x="0" y="28" fontFamily="Arial, sans-serif" fontSize="22" fontWeight="600" fill="currentColor">OpenAI</text></svg>,
  },
  {
    name: "Adobe",
    svg: <svg viewBox="0 0 100 40" fill="currentColor" className="h-5 sm:h-6 w-auto"><text x="0" y="30" fontFamily="Arial, sans-serif" fontSize="24" fontWeight="700" fill="currentColor">Adobe</text></svg>,
  },
  {
    name: "Slack",
    svg: <svg viewBox="0 0 100 40" fill="currentColor" className="h-5 sm:h-6 w-auto"><text x="0" y="30" fontFamily="Arial, sans-serif" fontSize="24" fontWeight="700" fill="currentColor">slack</text></svg>,
  },
];

export default function LogoMarquee() {
  // Triple the logos for seamless infinite scroll
  const tripledLogos = [...logos, ...logos, ...logos];

  return (
    <div className="relative w-full overflow-hidden py-8 sm:py-12">
      {/* Gradient overlays for fade effect */}
      <div className="absolute left-0 top-0 bottom-0 w-16 sm:w-32 bg-gradient-to-r from-black to-transparent z-10 pointer-events-none" />
      <div className="absolute right-0 top-0 bottom-0 w-16 sm:w-32 bg-gradient-to-l from-black to-transparent z-10 pointer-events-none" />
      
      {/* Scrolling container */}
      <div className="flex animate-scroll">
        {tripledLogos.map((logo, index) => (
          <div
            key={`${logo.name}-${index}`}
            className="flex-shrink-0 mx-8 sm:mx-12 opacity-70 hover:opacity-100 transition-opacity duration-300"
          >
            <div className="text-white">
              {logo.svg}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
