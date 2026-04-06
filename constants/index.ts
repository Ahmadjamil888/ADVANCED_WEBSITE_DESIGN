import { avatar1, avatar2, avatar3, avatar4, avatar5, avatar6, avatar7, avatar8, avatar9, instagram, linkedin, x, youtube } from "@/public";

/**
 * Navigation items for Zehanx Technologies
 */
export const navigationItems = [
   {
      id: 1,
      title: 'About',
      href: '/about'
   },
   {
      id: 2,
      title: 'Services',
      href: '/services'
   },
   {
      id: 3,
      title: 'Solutions',
      href: '/solutions'
   },
   {
      id: 4,
      title: 'Contact',
      href: '/contact'
   }
];

/**
 * Services offered by Zehanx Technologies (transformed from pricing)
 */
export const servicesItems = [
   {
      id: 1,
      title: "Machine Learning",
      subtitle: "AI-Powered Solutions",
      description: "Custom ML models, predictive analytics, and intelligent automation for your business.",
      features: [
         "Custom Model Development",
         "Predictive Analytics",
         "Natural Language Processing",
         "Computer Vision Solutions",
         "MLOps & Model Deployment",
      ],
      icon: "brain",
      popular: false,
   },
   {
      id: 2,
      title: "Software Development",
      subtitle: "Full-Stack Solutions",
      description: "End-to-end software development from web apps to enterprise systems.",
      features: [
         "Web Application Development",
         "Mobile App Development",
         "API Design & Integration",
         "Cloud Architecture",
         "DevOps & CI/CD",
         "Database Solutions",
         "Legacy System Modernization",
      ],
      icon: "code",
      popular: true,
   },
   {
      id: 3,
      title: "Data Engineering",
      subtitle: "Data Infrastructure",
      description: "Build robust data pipelines and infrastructure for data-driven decisions.",
      features: [
         "Data Pipeline Development",
         "ETL/ELT Processes",
         "Data Warehousing",
         "Big Data Solutions",
         "Real-time Data Processing",
         "Data Quality & Governance",
         "Business Intelligence",
      ],
      icon: "database",
      popular: false,
   },
];

/**
 * Client testimonials for Zehanx Technologies
 */
export const testimonials = [
   {
      id: 1,
      text: "Zehanx transformed our legacy system into a modern ML-powered platform. Their expertise in both software and AI is unmatched.",
      src: avatar1,
      name: "Ahmed Khan",
      username: "CTO, TechVentures",
   },
   {
      id: 2,
      text: "The machine learning models they built for our predictive maintenance have reduced downtime by 40%. Exceptional work!",
      src: avatar2,
      name: "Sarah Mitchell",
      username: "Operations Director, IndustrialCo",
   },
   {
      id: 3,
      text: "From concept to deployment, Zehanx delivered our fintech app on time with outstanding quality. Highly recommended.",
      src: avatar3,
      name: "James Wilson",
      username: "Founder, FinFlow",
   },
   {
      id: 4,
      text: "Their data engineering team built our entire analytics infrastructure. Now we make decisions 10x faster.",
      src: avatar4,
      name: "Fatima Ali",
      username: "Data Director, RetailMax",
   },
   {
      id: 5,
      text: "Professional, responsive, and incredibly skilled. Zehanx is our go-to partner for all software development needs.",
      src: avatar5,
      name: "Michael Chen",
      username: "Product Manager, CloudScale",
   },
   {
      id: 6,
      text: "The computer vision solution they created for our quality control process has revolutionized our manufacturing.",
      src: avatar6,
      name: "Emma Thompson",
      username: "Quality Lead, AutoTech",
   },
   {
      id: 7,
      text: "Zehanx doesn't just write code - they solve business problems. Their strategic approach to ML saved us months of work.",
      src: avatar7,
      name: "David Park",
      username: "CEO, InnovateLab",
   },
   {
      id: 8,
      text: "Our e-commerce recommendation engine built by Zehanx increased conversions by 25%. ROI was visible within weeks.",
      src: avatar8,
      name: "Lisa Anderson",
      username: "Marketing VP, ShopSmart",
   },
   {
      id: 9,
      text: "The team at Zehanx combines technical excellence with genuine business understanding. A rare find in the industry.",
      src: avatar9,
      name: "Robert Martinez",
      username: "IT Director, HealthPlus",
   },
];

/**
 * Technology partners / tools logos (keeping the marquee)
 */
export const logoMarqueeItems = [
   {
      id: 1,
      name: "TensorFlow"
   },
   {
      id: 2,
      name: "PyTorch"
   },
   {
      id: 3,
      name: "Python"
   },
   {
      id: 4,
      name: "React"
   },
   {
      id: 5,
      name: "AWS"
   },
   {
      id: 6,
      name: "Docker"
   }
];

/**
 * Footer navigation items
 */
export const footerItems = [
   {
      id: 1,
      title: 'About',
      href: '/about'
   },
   {
      id: 2,
      title: 'Services',
      href: '/services'
   },
   {
      id: 3,
      title: 'Solutions',
      href: '/solutions'
   },
   {
      id: 4,
      title: 'Contact',
      href: '/contact'
   },
   {
      id: 5,
      title: 'Careers',
      href: '/careers'
   }
];

/**
 * Social media links
 */
export const footerSocialsItems = [
   {
      id: 1,
      src: linkedin,
      href: "https://linkedin.com",
      label: "LinkedIn"
   },
   {
      id: 2,
      src: x,
      href: "https://twitter.com",
      label: "Twitter"
   },
   {
      id: 3,
      src: instagram,
      href: "https://instagram.com",
      label: "Instagram"
   },
   {
      id: 4,
      src: youtube,
      href: "https://youtube.com",
      label: "YouTube"
   },
];

/**
 * Contact information
 */
export const contactInfo = {
   email: "zehanxtech@gmail.com",
   whatsapp: "03338188722",
   phone: "+92 333 8188722",
   address: "Pakistan"
};

/**
 * Company information
 */
export const companyInfo = {
   name: "Zehanx Technologies",
   tagline: "Machine Learning & Software Solutions",
   description: "We transform businesses through intelligent technology solutions. From AI-powered applications to robust software systems, we deliver excellence.",
   founded: "2020",
   team: "15+ experts"
};

// Legacy exports for compatibility
export const pricingTiers = servicesItems;
export const pricingItems = servicesItems;