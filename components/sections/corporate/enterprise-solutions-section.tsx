'use client';

import CorporateSection from './corporate-section';

export default function EnterpriseSolutionsSection() {
  return (
    <CorporateSection
      title="Enterprise-Grade Solutions"
      subtitle="Scalable Technology"
      description="Built for enterprise scale with security, performance, and reliability at the core. Our solutions handle millions of users while maintaining peak performance."
      imageUrl="https://images.unsplash.com/photo-1560472354-b33ff0c44a43?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8M3x8ZW50ZXJwcmlzZSUyMHRlY2hub2xvZ3l8ZW58MHx8MHx8fDA%3D"
      imageAlt="Enterprise Technology Solutions"
      buttonText="View Enterprise Features"
      buttonHref="/services"
      features={[
        "99.9% uptime with enterprise-grade infrastructure",
        "Advanced security protocols and compliance",
        "Scalable architecture for growing businesses",
        "Dedicated support and maintenance"
      ]}
      reverse={true}
    />
  );
}