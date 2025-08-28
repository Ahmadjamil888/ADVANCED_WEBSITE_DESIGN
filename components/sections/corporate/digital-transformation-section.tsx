'use client';

import CorporateSection from './corporate-section';

export default function DigitalTransformationSection() {
  return (
    <CorporateSection
      title="Digital Transformation Excellence"
      subtitle="Future-Ready Business"
      description="Modernize your operations with our comprehensive digital transformation services. From legacy system migration to cloud-native applications, we guide your journey to digital excellence."
      imageUrl="https://images.unsplash.com/photo-1551434678-e076c223a692?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8ZGlnaXRhbCUyMHRyYW5zZm9ybWF0aW9ufGVufDB8fDB8fHww"
      imageAlt="Digital Transformation"
      buttonText="Start Transformation"
      buttonHref="/contact"
      features={[
        "Legacy system modernization and migration",
        "Cloud-native application development",
        "Process automation and optimization",
        "Change management and training support"
      ]}
    />
  );
}