'use client';

import CorporateSection from './corporate-section';

export default function InnovationLabSection() {
  return (
    <CorporateSection
      title="Innovation Laboratory"
      subtitle="Research & Development"
      description="Our dedicated R&D team continuously explores emerging technologies to keep you ahead of the curve. From quantum computing to advanced neural networks, we're building tomorrow's solutions today."
      imageUrl="https://images.unsplash.com/photo-1518709268805-4e9042af2176?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8M3x8aW5ub3ZhdGlvbiUyMGxhYnxlbnwwfHwwfHx8MA%3D%3D"
      imageAlt="Innovation Laboratory"
      buttonText="Explore Innovation"
      buttonHref="/services"
      features={[
        "Cutting-edge research in emerging technologies",
        "Prototype development and proof of concepts",
        "Technology roadmap planning and strategy",
        "Innovation workshops and consulting"
      ]}
      reverse={true}
    />
  );
}