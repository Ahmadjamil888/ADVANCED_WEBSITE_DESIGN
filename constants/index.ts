import { acme, apex, avatar1, avatar2, avatar3, avatar4, avatar5, avatar6, avatar7, avatar8, avatar9, celestial, echo, instagram, linkedin, pin, pulse, quantum, x, youtube } from "@/public";
import { features } from "process";

/**
 * An array of navigation items, each with an id, title, and href.
 * These items are used to build the main navigation menu of the application.
 */
export const navigationItems = [
   {
      id: 1,
      title: 'Services',
      href: '/services'
   },
   {
      id: 2,
      title: 'About',
      href: '/about'
   },
   {
      id: 3,
      title: 'Team',
      href: '/team'
   },
   {
      id: 4,
      title: 'Contact',
      href: '/contact'
   }
];

export const pricingTiers = [
   {
      id: 1,
      title: "Free",
      monthlyPrice: 0,
      buttonText: "Get started for free",
      popular: false,
      inverse: false,
      features: [
         "Up to 5 project members",
         "Unlimited tasks and projects",
         "2GB storage",
         "Integrations",
         "Basic support",
      ],
   },
   {
      id: 2,
      title: "Pro",
      monthlyPrice: 9,
      buttonText: "Sign up now",
      popular: true,
      inverse: true,
      features: [
         "Up to 50 project members",
         "Unlimited tasks and projects",
         "50GB storage",
         "Integrations",
         "Priority support",
         "Advanced support",
         "Export support",
      ],
   },
   {
      id: 3,
      title: "Business",
      monthlyPrice: 19,
      buttonText: "Sign up now",
      popular: false,
      inverse: false,
      features: [
         "Up to 5 project members",
         "Unlimited tasks and projects",
         "200GB storage",
         "Integrations",
         "Dedicated account manager",
         "Custom fields",
         "Advanced analytics",
         "Export capabilities",
         "API access",
         "Advanced security features",
      ],
   },
];

export const testimonials = [
   {
      id: 1,
      text: "Zehanx Technologies delivered AI solutions that perfectly fit our workflow. The platform is user-friendly and powerful.",
      src: avatar9,
      name: "Ahsan Khalid",
      username: "@ahsan_k",
   },
   {
      id: 2,
      text: "As a product designer, I’m always exploring innovative tech—and Zehanx instantly impressed me with its clean development and smart automation.",
      src: avatar1,
      name: "Sana Farooq",
      username: "@sanafx",
   },
   {
      id: 3,
      text: "Our productivity skyrocketed after integrating Zehanx’s AI tools. Tasks that took hours now complete in minutes.",
      src: avatar2,
      name: "Bilal Ahmed",
      username: "@bilaldev",
   },
   {
      id: 4,
      text: "Their software has transformed how I manage projects, automate processes, and meet deadlines—truly next-level tech.",
      src: avatar3,
      name: "Maria Sheikh",
      username: "@mariasheikh",
   },
   {
      id: 5,
      text: "Integration was smooth and extremely fast. Zehanx provided complete support, making onboarding effortless.",
      src: avatar4,
      name: "Humza Tariq",
      username: "@humzata",
   },
   {
      id: 6,
      text: "Their AI-powered systems help us track operations efficiently and prevent workflow bottlenecks. It’s a game-changer.",
      src: avatar5,
      name: "Laiba Noor",
      username: "@laibatech",
   },
   {
      id: 7,
      text: "The customization and API integrations offered by Zehanx are top-tier. It fits seamlessly into our existing stack.",
      src: avatar6,
      name: "Rehan Malik",
      username: "@rehanm_dev",
   },
   {
      id: 8,
      text: "Switching to Zehanx completely streamlined our team communication and project tracking. Excellent experience.",
      src: avatar7,
      name: "Farhan Ali",
      username: "@farhan.codes",
   },
   {
      id: 9,
      text: "Task assignment, AI-based tracking, and centralized documentation have made our team far more efficient.",
      src: avatar8,
      name: "Imran Siddiqui",
      username: "@imransid_tech",
   },
];

export const logoMarqueeItems = [
   {
      id: 1,
      src: apex
   },
   {
      id: 2,
      src: acme
   },
   {
      id: 3,
      src: celestial
   },
   {
      id: 4,
      src: echo
   },
   {
      id: 5,
      src: pulse
   },
   {
      id: 6,
      src: quantum
   }
];

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
      title: 'Team',
      href: '/team'
   },
   {
      id: 4,
      title: 'Contact',
      href: '/contact'
   },
   {
      id: 5,
      title: 'Privacy',
      href: '/privacy'
   },
   {
      id: 6,
      title: 'Terms',
      href: '/terms'
   }
];

export const contactInfo = {
   email: 'zehanxtech@gmail.com',
   phone: '+92 344 2693910',
   company: 'Zehanx Technologies',
   slogan: 'From concepts to reality'
};

export const footerSocialsItems = [
   {
      id: 1,
      src: instagram,
      href: "/"
   },
   {
      id: 2,
      src: linkedin,
      href: "/"
   },
   {
      id: 3,
      src: pin,
      href: "/"
   },
   {
      id: 4,
      src: x,
      href: "/"
   },
   {
      id: 5,
      src: youtube,
      href: "/"
   },
];

export const pricingItems = [
   {
      id: 1,
      title: 'Artificial Intelligence',
      price: null,
      btn: "Learn More",
      features: [
         {
            id: 1,
            feature: "Machine Learning Models",
         },
         {
            id: 2,
            feature: "Natural Language Processing",
         },
         {
            id: 3,
            feature: "Computer Vision Solutions",
         },
         {
            id: 4,
            feature: "Predictive Analytics",
         },
         {
            id: 5,
            feature: "AI Consulting",
         },
      ]
   },
   {
      id: 2,
      title: 'Data Science',
      price: null,
      btn: "Learn More",
      features: [
         {
            id: 1,
            feature: "Data Analysis & Visualization",
         },
         {
            id: 2,
            feature: "Statistical Modeling",
         },
         {
            id: 3,
            feature: "Big Data Processing",
         },
         {
            id: 4,
            feature: "Business Intelligence",
         },
         {
            id: 5,
            feature: "Data Pipeline Development",
         },
         {
            id: 6,
            feature: "Reporting & Dashboards",
         },
         {
            id: 7,
            feature: "Data Strategy Consulting",
         },
      ]
   },
   {
      id: 3,
      title: 'Software Development',
      price: null,
      btn: "Learn More",
      features: [
         {
            id: 1,
            feature: "Web Application Development",
         },
         {
            id: 2,
            feature: "Mobile App Development",
         },
         {
            id: 3,
            feature: "Desktop Applications",
         },
         {
            id: 4,
            feature: "API Development",
         },
         {
            id: 5,
            feature: "Cloud Solutions",
         },
         {
            id: 6,
            feature: "DevOps & Infrastructure",
         },
         {
            id: 7,
            feature: "Quality Assurance & Testing",
         },
      ]
   },
];