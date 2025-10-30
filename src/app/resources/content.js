import { Logo } from "@/once-ui/components";

const person = {
  firstName: "zehanx",
  lastName: "tech",
  get name() {
    return `${this.firstName}${this.lastName}`;
  },
  role: "AI & Web Development Company",
  avatar: "/logo.jpg",
  email: "zehanxtech@gmail.com",
  location: "Asia/Karachi", // Expecting the IANA time zone identifier, e.g., 'Europe/Vienna'
  languages: ["English", "Urdu"], // optional: Leave the array empty if you don't want to display languages
};

const newsletter = {
  display: true,
  title: <>Subscribe to zehanxtech's Newsletter</>,
  description: (
    <>
      We occasionally share insights about AI development, web technologies, and cutting-edge solutions.
    </>
  ),
};

const social = [
  // Links are automatically displayed.
  // Import new icons in /once-ui/icons.ts
  {
    name: "GitHub",
    icon: "github",
    link: "https://github.com/Ahmadjamil888",
  },
  {
    name: "LinkedIn",
    icon: "linkedin",
    link: "https://www.linkedin.com/company/zehanxtech",
  },
  {
    name: "YouTube",
    icon: "youtube",
    link: "https://www.youtube.com/@zehanxtech",
  },
  {
    name: "Email",
    icon: "email",
    link: `mailto:${person.email}`,
  },
  {
    name: "Phone",
    icon: "phone",
    link: "tel:+923442693910",
  },
];

const home = {
  path: "/",
  image: "/images/og/home.jpg",
  label: "Home",
  title: `${person.name} - AI & Web Development`,
  description: `zehanxtech - Building AI for Better of Humanity. We specialize in AI solutions, web development, and cutting-edge technology.`,
  headline: <>Building AI for Better of Humanity</>,
  featured: {
    display: true,
    title: <>Recent projects: <strong className="ml-4">by zehanxtech</strong></>,
    href: "https://github.com/Ahmadjamil888?tab=repositories",
  },
  subline: (
    <>
      We're zehanxtech, an AI & Web Development company where we craft intelligent
      <br /> solutions and cutting-edge web applications for the future.
    </>
  ),
};

const about = {
  path: "/about",
  label: "About",
  title: `About – ${person.name}`,
  description: `Meet ${person.name}, ${person.role} from ${person.location}`,
  tableOfContent: {
    display: true,
    subItems: false,
  },
  avatar: {
    display: true,
  },
  calendar: {
    display: true,
    link: "mailto:zehanxtech@gmail.com",
  },
  intro: {
    display: true,
    title: "Introduction",
    description: (
      <>
        Welcome to zehanxtech, where we're passionate about building AI solutions for the betterment of humanity. 
        We specialize in creating cutting-edge AI applications, modern web development, and innovative digital solutions 
        that transform businesses and improve lives.
      </>
    ),
  },
  work: {
    display: true, // set to false to hide this section
    title: "Our Journey",
    experiences: [
      {
        company: "zehanxtech",
        timeframe: "2024 - Present",
        role: "AI & Web Development Company",
        achievements: [
          <>
            Founded zehanxtech with a mission to build AI solutions for the betterment of humanity.
          </>,
          <>
            Developed cutting-edge AI applications including chatbots, machine learning models, and intelligent web applications.
          </>,
          <>
            Specialized in modern web development using Next.js, React, and advanced AI integration.
          </>,
        ],
        images: [],
      },
      {
        company: "Technology Innovation",
        timeframe: "2023 - 2024",
        role: "Research & Development",
        achievements: [
          <>
            Researched and developed AI-powered solutions for various industries including healthcare, finance, and education.
          </>,
          <>
            Built scalable web applications with focus on performance, accessibility, and user experience.
          </>,
        ],
        images: [],
      },
    ],
  },
  studies: {
    display: true, // set to false to hide this section
    title: "Expertise",
    institutions: [
      {
        name: "Artificial Intelligence",
        description: <>Advanced AI development, machine learning, and neural networks.</>,
      },
      {
        name: "Web Development",
        description: <>Modern web technologies, full-stack development, and cloud solutions.</>,
      },
      {
        name: "Digital Innovation",
        description: <>Cutting-edge technology research and innovative solution development.</>,
      },
    ],
  },
  technical: {
    display: true, // set to false to hide this section
    title: "Technical Skills",
    skills: [
      {
        title: "AI & Machine Learning",
        description: <>Advanced AI development, neural networks, natural language processing, and computer vision solutions.</>,
        images: [],
      },
      {
        title: "Web Development",
        description: <>Full-stack development with Next.js, React, Node.js, TypeScript, and modern web technologies.</>,
        images: [],
      },
      {
        title: "Cloud & DevOps",
        description: <>Cloud deployment, CI/CD pipelines, containerization, and scalable infrastructure solutions.</>,
        images: [],
      },
      {
        title: "UI/UX Design",
        description: <>Modern design systems, user experience optimization, and responsive web design.</>,
        images: [],
      },
    ],
  },
};

const blog = {
  path: "/blog",
  label: "Blog",
  title: "Writing about design and tech...",
  description: `Read what ${person.name} has been up to recently`,
  // Create new blog posts by adding a new .mdx file to app/blog/posts
  // All posts will be listed on the /blog route
};

const work = {
  path: "/work",
  label: "Portfolio",
  title: `Projects – ${person.name}`,
  description: `AI and web development projects by ${person.name}`,
  // Create new project pages by adding a new .mdx file to app/blog/posts
  // All projects will be listed on the /home and /work routes
};

const gallery = {
  path: "/gallery",
  label: "Gallery",
  title: `Photo gallery – ${person.name}`,
  description: `A photo collection by ${person.name}`,
  // Images by https://lorant.one
  // These are placeholder images, replace with your own
  images: [
    {
      src: "/images/gallery/horizontal-1.jpg",
      alt: "image",
      orientation: "horizontal",
    },
    {
      src: "/images/gallery/horizontal-2.jpg",
      alt: "image",
      orientation: "horizontal",
    },
    {
      src: "/images/gallery/horizontal-3.jpg",
      alt: "image",
      orientation: "horizontal",
    },
    {
      src: "/images/gallery/horizontal-4.jpg",
      alt: "image",
      orientation: "horizontal",
    },
    {
      src: "/images/gallery/vertical-1.jpg",
      alt: "image",
      orientation: "vertical",
    },
    {
      src: "/images/gallery/vertical-2.jpg",
      alt: "image",
      orientation: "vertical",
    },
    {
      src: "/images/gallery/vertical-3.jpg",
      alt: "image",
      orientation: "vertical",
    },
    {
      src: "/images/gallery/vertical-4.jpg",
      alt: "image",
      orientation: "vertical",
    },
  ],
};

export { person, social, newsletter, home, about, blog, work, gallery };
