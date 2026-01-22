import { acme, apex, avatar1, avatar2, avatar3, avatar4, avatar5, avatar6, avatar7, avatar8, avatar9, celestial, echo, instagram, linkedin, pin, pulse, quantum, x, youtube, usmanHospital, aurionTech, irtcop, apsacs, nicLahore } from "@/public";

/**
 * An array of navigation items, each with an id, title, and href.
 * These items are used to build the main navigation menu of the application.
 */
export const navigationItems = [
   {
      id: 1,
      title: 'Solutions',
      href: '/services'
   },
   {
      id: 2,
      title: 'About',
      href: '/about'
   },
   {
      id: 3,
      title: 'Products',
      href: '/products'
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
      text: "Zehanx Technologies' software solutions have transformed our business operations. The integration and efficiency gains are remarkable.",
      src: avatar9,
      name: "Ahsan Khalid",
      username: "@ahsan_k",
   },
   {
      id: 2,
      text: "The ML and Gen AI solutions from Zehanx have revolutionized our data analysis capabilities. The insights and automation are game-changing.",
      src: avatar1,
      name: "Sana Farooq",
      username: "@sanafx",
   },
   {
      id: 3,
      text: "Implementing Zehanx's cybersecurity solutions reduced our risk exposure by 90%. Their expertise in enterprise security is exceptional.",
      src: avatar2,
      name: "Bilal Ahmed",
      username: "@bilaldev",
   },
   {
      id: 4,
      text: "Zehanx's software products have become essential to our daily operations. The reliability and support are top-notch.",
      src: avatar3,
      name: "Maria Sheikh",
      username: "@mariasheikh",
   },
   {
      id: 5,
      text: "The custom neural network solutions from Zehanx Technologies gave us a significant competitive advantage. Professional and reliable service.",
      src: avatar4,
      name: "Humza Tariq",
      username: "@humzata",
   },
   {
      id: 6,
      text: "Zehanx's approach to NLP solutions ensures our systems are both powerful and ethically sound. Exactly what we needed for our platform.",
      src: avatar5,
      name: "Laiba Noor",
      username: "@laibatech",
   },
   {
      id: 7,
      text: "The software solutions from Zehanx have exceeded our expectations in both performance and scalability. Truly exceptional work.",
      src: avatar6,
      name: "Rehan Malik",
      username: "@rehanm_dev",
   },
   {
      id: 8,
      text: "Zehanx Technologies has set a new standard for enterprise software development. Their BSC solutions are particularly impressive.",
      src: avatar7,
      name: "Farhan Ali",
      username: "@farhan.codes",
   },
   {
      id: 9,
      text: "The comprehensive software suite from Zehanx has transformed our workflow efficiency. ML, cybersecurity, and business solutions all in one place.",
      src: avatar8,
      name: "Imran Siddiqui",
      username: "@imransid_tech",
   },
];

export const logoMarqueeItems = [
   {
      id: 1,
      src: usmanHospital
   },
   {
      id: 2,
      src: aurionTech
   },
   {
      id: 3,
      src: irtcop
   },
   {
      id: 4,
      src: apsacs
   },
   {
      id: 5,
      src: nicLahore
   }
];

export const footerItems = [
   {
      id: 1,
      title: 'Solutions',
      href: '/services'
   },
   {
      id: 2,
      title: 'About',
      href: '/about'
   },
   {
      id: 3,
      title: 'Products',
      href: '/products'
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
   slogan: 'Innovative software solutions in AI, ML, Cybersecurity and Enterprise Systems'
};

export const footerSocialsItems = [
   {
      id: 1,
      src: instagram,
      href: "https://instagram.com/zehanxtech"
   },
   {
      id: 2,
      src: linkedin,
      href: "https://linkedin.com/company/zehanx-technologies"
   },
   {
      id: 3,
      src: pin,
      href: "https://pinterest.com/zehanxtech"
   },
   {
      id: 4,
      src: x,
      href: "https://x.com/zehanxtech"
   },
   {
      id: 5,
      src: youtube,
      href: "https://youtube.com/@zehanxtech"
   },
];

export const pricingItems = [
   {
      id: 1,
      title: 'Enterprise Software Solutions',
      price: null,
      btn: "Learn More",
      features: [
         {
            id: 1,
            feature: "Custom Business Solutions",
         },
         {
            id: 2,
            feature: "Scalable Enterprise Systems",
         },
         {
            id: 3,
            feature: "Integration & Migration",
         },
         {
            id: 4,
            feature: "Business Process Automation",
         },
         {
            id: 5,
            feature: "Analytics & Reporting",
         },
      ]
   },
   {
      id: 2,
      title: 'AI/ML & Gen AI',
      price: null,
      btn: "Learn More",
      features: [
         {
            id: 1,
            feature: "Neural Network Development",
         },
         {
            id: 2,
            feature: "Deep Learning Models",
         },
         {
            id: 3,
            feature: "NLP & Computer Vision",
         },
         {
            id: 4,
            feature: "Generative AI Solutions",
         },
         {
            id: 5,
            feature: "Custom Model Training",
         },
         {
            id: 6,
            feature: "AI System Integration",
         },
         {
            id: 7,
            feature: "MLOps & Deployment",
         },
      ]
   },
   {
      id: 3,
      title: 'Cybersecurity & Software',
      price: null,
      btn: "Learn More",
      features: [
         {
            id: 1,
            feature: "Enterprise Security Solutions",
         },
         {
            id: 2,
            feature: "Threat Detection & Response",
         },
         {
            id: 3,
            feature: "Secure Software Development",
         },
         {
            id: 4,
            feature: "Vulnerability Assessment",
         },
         {
            id: 5,
            feature: "Security Compliance",
         },
         {
            id: 6,
            feature: "Proprietary Software Solutions",
         },
         {
            id: 7,
            feature: "24/7 Security Monitoring",
         },
      ]
   },
];