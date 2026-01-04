import { acme, apex, avatar1, avatar2, avatar3, avatar4, avatar5, avatar6, avatar7, avatar8, avatar9, celestial, echo, instagram, linkedin, pin, pulse, quantum, x, youtube } from "@/public";
import { features } from "process";

/**
 * An array of navigation items, each with an id, title, and href.
 * These items are used to build the main navigation menu of the application.
 */
export const navigationItems = [
   {
      id: 1,
      title: 'Products',
      href: '/products'
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
      text: "Zehanx Technologies' mindful AI models have revolutionized our data processing. The accuracy and efficiency are unmatched.",
      src: avatar9,
      name: "Ahsan Khalid",
      username: "@ahsan_k",
   },
   {
      id: 2,
      text: "The LLM solutions developed by Zehanx have transformed our customer service operations. The contextual understanding is remarkable.",
      src: avatar1,
      name: "Sana Farooq",
      username: "@sanafx",
   },
   {
      id: 3,
      text: "Implementing Zehanx's ML models reduced our processing time by 80%. Their mindful approach to AI development is evident in every solution.",
      src: avatar2,
      name: "Bilal Ahmed",
      username: "@bilaldev",
   },
   {
      id: 4,
      text: "Zehanx's AI models have become integral to our decision-making processes. The insights generated are incredibly valuable.",
      src: avatar3,
      name: "Maria Sheikh",
      username: "@mariasheikh",
   },
   {
      id: 5,
      text: "The custom AI solutions from Zehanx Technologies have given us a competitive edge. Their models are both powerful and ethical.",
      src: avatar4,
      name: "Humza Tariq",
      username: "@humzata",
   },
   {
      id: 6,
      text: "Zehanx's mindful AI approach ensures our models are not just accurate but also responsible. This is exactly what we needed.",
      src: avatar5,
      name: "Laiba Noor",
      username: "@laibatech",
   },
   {
      id: 7,
      text: "The AI models developed by Zehanx have exceeded our expectations in both performance and reliability. Truly exceptional work.",
      src: avatar6,
      name: "Rehan Malik",
      username: "@rehanm_dev",
   },
   {
      id: 8,
      text: "Zehanx Technologies has set a new standard for AI development. Their mindful approach to model creation is impressive.",
      src: avatar7,
      name: "Farhan Ali",
      username: "@farhan.codes",
   },
   {
      id: 9,
      text: "The AI solutions from Zehanx have transformed how we analyze data and make predictions. Accuracy and efficiency combined.",
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
      title: 'Products',
      href: '/products'
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
   slogan: 'Developing mindful AI models for a better future'
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
      title: 'Mindful AI Models',
      price: null,
      btn: "Learn More",
      features: [
         {
            id: 1,
            feature: "Custom Neural Networks",
         },
         {
            id: 2,
            feature: "Ethical AI Frameworks",
         },
         {
            id: 3,
            feature: "Responsible ML Models",
         },
         {
            id: 4,
            feature: "Bias Detection & Mitigation",
         },
         {
            id: 5,
            feature: "Transparent AI Systems",
         },
      ]
   },
   {
      id: 2,
      title: 'Large Language Models',
      price: null,
      btn: "Learn More",
      features: [
         {
            id: 1,
            feature: "Custom LLM Training",
         },
         {
            id: 2,
            feature: "Contextual Understanding",
         },
         {
            id: 3,
            feature: "Domain-Specific Models",
         },
         {
            id: 4,
            feature: "Multilingual Capabilities",
         },
         {
            id: 5,
            feature: "Fine-tuning Services",
         },
         {
            id: 6,
            feature: "Model Optimization",
         },
         {
            id: 7,
            feature: "Deployment Solutions",
         },
      ]
   },
   {
      id: 3,
      title: 'AI/ML Research',
      price: null,
      btn: "Learn More",
      features: [
         {
            id: 1,
            feature: "Cutting-Edge Research",
         },
         {
            id: 2,
            feature: "Model Architecture Design",
         },
         {
            id: 3,
            feature: "Algorithm Development",
         },
         {
            id: 4,
            feature: "Performance Optimization",
         },
         {
            id: 5,
            feature: "Validation & Testing",
         },
         {
            id: 6,
            feature: "Documentation & Papers",
         },
         {
            id: 7,
            feature: "Open Source Contributions",
         },
      ]
   },
];