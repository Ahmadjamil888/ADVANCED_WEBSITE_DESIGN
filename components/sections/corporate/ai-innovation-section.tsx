'use client';

import CorporateSection from './corporate-section';

export default function AIInnovationSection() {
  return (
    <CorporateSection
      title="AI Innovation at Scale"
      subtitle="Artificial Intelligence"
      description="Transform your business with cutting-edge AI solutions. Our advanced machine learning models and deep learning algorithms deliver intelligent automation that drives real results."
      imageUrl="https://plus.unsplash.com/premium_photo-1661284972849-bff06ac39c30?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MXx8dGVhbSUyMEFJfGVufDB8fDB8fHww"
      imageAlt="AI Innovation Team"
      buttonText="Explore AI Solutions"
      buttonHref="/services"
      features={[
        "Custom machine learning models tailored to your business",
        "Predictive analytics for data-driven decision making",
        "Intelligent automation to streamline operations",
        "24/7 AI monitoring and optimization"
      ]}
    />
  );
}