import { motion } from "framer-motion";

const items = [
  {
    number: "01",
    title: "Enterprise & Govt Software",
    description:
      "We design and build secure, scalable custom software solutions tailored specifically for government agencies and large-scale enterprises.",
    gradient: "from-cyan-500/10 to-cyan-500/0",
    accent: "#06b6d4",
    icon: (
      <svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg" className="w-full h-full">
        <path opacity="0.15" d="M45.9998 10H31.9998L19.7271 26L31.9998 56L59.9998 26L45.9998 10Z" fill="currentColor" />
        <path d="M18 10H46L60 26L32 56L4 26L18 10Z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M44.2726 26L31.9998 56L19.7271 26L31.9998 10L44.2726 26Z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M4 26H60" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
  },
  {
    number: "02",
    title: "AI & Machine Learning",
    description:
      "From Deep Learning models to Natural Language Processing (NLP), we deliver cutting-edge AI services that automate processes and generate insights.",
    gradient: "from-violet-500/10 to-violet-500/0",
    accent: "#8b5cf6",
    icon: (
      <svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg" className="w-full h-full">
        <path opacity="0.15" d="M49.0005 54C52.8665 54 56.0005 50.866 56.0005 47C56.0005 43.134 52.8665 40 49.0005 40C45.1345 40 42.0005 43.134 42.0005 47C42.0005 50.866 45.1345 54 49.0005 54Z" fill="currentColor" />
        <path opacity="0.15" d="M15 24C18.866 24 22 20.866 22 17C22 13.134 18.866 10 15 10C11.134 10 8 13.134 8 17C8 20.866 11.134 24 15 24Z" fill="currentColor" />
        <path d="M49.0005 54C52.8665 54 56.0005 50.866 56.0005 47C56.0005 43.134 52.8665 40 49.0005 40C45.1345 40 42.0005 43.134 42.0005 47C42.0005 50.866 45.1345 54 49.0005 54Z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M49.0006 40L49.0002 29.9703C49.0001 26.7878 47.7358 23.7358 45.4855 21.4855L36 12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M36 22V12H46" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M15 24C18.866 24 22 20.866 22 17C22 13.134 18.866 10 15 10C11.134 10 8 13.134 8 17C8 20.866 11.134 24 15 24Z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M15 24L15.0004 34.0297C15.0005 37.2122 16.2648 40.2642 18.5151 42.5145L28.0006 52" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M28.0003 42V52H18.0003" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
  },
  {
    number: "03",
    title: "Deep Tech & DL Solutions",
    description:
      "We leverage state-of-the-art Neural Networks and deep learning frameworks to solve complex computations and predictive analytics.",
    gradient: "from-emerald-500/10 to-emerald-500/0",
    accent: "#10b981",
    icon: (
      <svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg" className="w-full h-full">
        <path opacity="0.15" d="M50 38.2155L40.8007 47.4147C40.5565 47.659 40.2532 47.8358 39.9204 47.928C39.5875 48.0201 39.2365 48.0246 38.9014 47.9408L24.4122 44.3185C24.1404 44.2506 23.886 44.1263 23.6653 43.9537L10 33.268L18.1436 17.9475L30.9736 14.2071C31.4319 14.0735 31.9229 14.1082 32.3578 14.3051L41 18.2155H35.8284C35.5658 18.2155 35.3057 18.2672 35.0631 18.3677C34.8204 18.4682 34.5999 18.6155 34.4142 18.8012L24.6306 28.5848C24.428 28.7875 24.2713 29.0313 24.1711 29.2997C24.0709 29.5682 24.0295 29.855 24.0498 30.1408C24.0702 30.4267 24.1517 30.7048 24.2888 30.9564C24.426 31.208 24.6156 31.4271 24.8448 31.5991L26.2 32.6155C27.5848 33.654 29.269 34.2155 31 34.2155C32.731 34.2155 34.4152 33.654 35.8 32.6155L39 30.2155L50 38.2155Z" fill="currentColor" />
        <path d="M60.1794 30.4462L54 33.5359L46 18.2154L52.2423 15.0942C52.7113 14.8597 53.2536 14.8188 53.7525 14.9802C54.2514 15.1416 54.6669 15.4925 54.9096 15.9573L61.0578 27.7316C61.1808 27.967 61.2556 28.2246 61.2779 28.4892C61.3002 28.7539 61.2696 29.0204 61.1878 29.2731C61.1061 29.5258 60.9748 29.7596 60.8016 29.9611C60.6285 30.1625 60.417 30.3274 60.1794 30.4462Z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M10.0011 33.2681L3.8217 30.1784C3.58414 30.0596 3.37261 29.8947 3.19947 29.6932C3.02633 29.4918 2.89504 29.258 2.81327 29.0052C2.7315 28.7525 2.70088 28.4861 2.7232 28.2214C2.74552 27.9568 2.82033 27.6992 2.94327 27.4638L9.09151 15.6895C9.33421 15.2247 9.74973 14.8738 10.2486 14.7124C10.7475 14.551 11.2898 14.5919 11.7588 14.8264L18.0011 17.9475L10.0011 33.2681Z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M54 33.5359L50 38.2154L40.8007 47.4147C40.5565 47.6589 40.2532 47.8357 39.9204 47.9279C39.5875 48.0201 39.2365 48.0245 38.9014 47.9407L24.4122 44.3184C24.1404 44.2505 23.886 44.1262 23.6653 43.9537L10 33.2679" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M50 38.2155L39 30.2155L35.8 32.6155C34.4152 33.654 32.7309 34.2155 31 34.2155C29.269 34.2155 27.5848 33.654 26.2 32.6155L24.8448 31.5991C24.6156 31.4271 24.4259 31.208 24.2888 30.9564C24.1516 30.7048 24.0701 30.4267 24.0498 30.1408C24.0295 29.855 24.0709 29.5682 24.1711 29.2997C24.2713 29.0313 24.428 28.7875 24.6306 28.5849L34.4142 18.8012C34.5999 18.6155 34.8204 18.4682 35.063 18.3677C35.3057 18.2672 35.5658 18.2155 35.8284 18.2155H46" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M18.1433 17.9475L30.9733 14.2071C31.4316 14.0735 31.9226 14.1082 32.3576 14.3051L40.9997 18.2155" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M28 53.2154L20.4651 51.3317C20.1594 51.2552 19.876 51.1076 19.6381 50.9008L14 46" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
  },
  {
    number: "04",
    title: "Security & Compliance",
    description:
      "Built on strict security protocols, ensuring our enterprise and government systems satisfy all regulatory compliance and protection standards.",
    gradient: "from-rose-500/10 to-rose-500/0",
    accent: "#f43f5e",
    icon: (
      <svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg" className="w-full h-full">
        <path opacity="0.15" d="M10 26.6667V12C10 11.4696 10.2107 10.9609 10.5858 10.5858C10.9609 10.2107 11.4696 10 12 10H52C52.5304 10 53.0391 10.2107 53.4142 10.5858C53.7893 10.9609 54 11.4696 54 12V26.6667C54 47.6705 36.1735 54.6292 32.6141 55.8093C32.2161 55.9463 31.7839 55.9463 31.386 55.8093C27.8265 54.6292 10 47.6705 10 26.6667Z" fill="currentColor" />
        <path d="M10 26.6667V12C10 11.4696 10.2107 10.9609 10.5858 10.5858C10.9609 10.2107 11.4696 10 12 10H52C52.5304 10 53.0391 10.2107 53.4142 10.5858C53.7893 10.9609 54 11.4696 54 12V26.6667C54 47.6705 36.1735 54.6292 32.6141 55.8093C32.2161 55.9463 31.7839 55.9463 31.386 55.8093C27.8265 54.6292 10 47.6705 10 26.6667Z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M43 24L28.3333 38L21 31" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
  },
];

// Framer-style stagger container — identical pattern to navbar AnimatePresence
const containerVariants = {
  hidden: {},
  visible: {
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.05,
    },
  },
};

// Clean slide-up + fade — same easing as navbar items
const cardVariants = {
  hidden: { opacity: 0, y: 40, filter: "blur(4px)" },
  visible: {
    opacity: 1,
    y: 0,
    filter: "blur(0px)",
    transition: { duration: 0.6, ease: [0.22, 1, 0.36, 1] },
  },
};

function Card() {
  return (
    <motion.div
      className="card-container w-full grid grid-cols-1 sm:grid-cols-2 gap-4 sm:gap-5"
      variants={containerVariants}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, amount: 0.1 }}
    >
      {items.map((item) => (
        <motion.div
          key={item.number}
          variants={cardVariants}
          whileHover={{ y: -4, scale: 1.01 }}
          transition={{ type: "spring", stiffness: 280, damping: 22 }}
          className="card group relative flex flex-col gap-4 p-5 sm:p-6
            rounded-2xl overflow-hidden cursor-default
            border border-white/[0.08] bg-white/[0.03]
            hover:border-white/[0.18] hover:bg-white/[0.06]
            transition-all duration-300"
          style={{ "--accent": item.accent }}
        >
          {/* subtle gradient wash on hover */}
          <div
            className={`absolute inset-0 bg-gradient-to-br ${item.gradient}
              opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none`}
          />

          {/* top row: number badge + icon */}
          <div className="relative flex items-start justify-between">
            <span
              className="text-[10px] font-[Sansita] font-bold tracking-[0.2em] uppercase
                px-2.5 py-1 rounded-full border"
              style={{ color: item.accent, borderColor: `${item.accent}33`, background: `${item.accent}11` }}
            >
              {item.number}
            </span>

            <div
              className="w-10 h-10 sm:w-11 sm:h-11 flex-shrink-0 transition-transform
                duration-300 group-hover:scale-110"
              style={{ color: item.accent }}
            >
              {item.icon}
            </div>
          </div>

          {/* title */}
          <h3 className="relative font-[SansitaBold] text-[1.05rem] sm:text-[1.1rem]
            leading-snug text-[var(--text-light)]">
            {item.title}
          </h3>

          {/* divider */}
          <div className="relative h-px w-full bg-white/[0.06]" />

          {/* description */}
          <p className="relative font-[Sansita] text-[0.82rem] sm:text-[0.88rem]
            leading-relaxed text-[var(--text-muted)]">
            {item.description}
          </p>

          {/* arrow — appears on hover */}
          <div
            className="relative flex items-center gap-1.5 mt-auto pt-1
              opacity-0 group-hover:opacity-100 translate-x-0 group-hover:translate-x-1
              transition-all duration-300"
            style={{ color: item.accent }}
          >
            <span className="font-[Sansita] text-[0.78rem] font-semibold tracking-wide uppercase">
              Learn more
            </span>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M5 12h14M12 5l7 7-7 7" />
            </svg>
          </div>
        </motion.div>
      ))}
    </motion.div>
  );
}

export default Card;
