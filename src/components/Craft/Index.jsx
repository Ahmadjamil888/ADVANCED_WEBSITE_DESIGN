import Card from "../Card";
import Button from "../Button";
import { useRef, useEffect } from "react";
import { motion } from "framer-motion";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
gsap.registerPlugin(ScrollTrigger);

// Framer navbar-style reveal variants
const fadeUp = {
  hidden: { opacity: 0, y: 28, filter: "blur(4px)" },
  visible: (delay = 0) => ({
    opacity: 1,
    y: 0,
    filter: "blur(0px)",
    transition: { duration: 0.65, ease: [0.22, 1, 0.36, 1], delay },
  }),
};

const lineDraw = {
  hidden: { scaleX: 0, originX: 0 },
  visible: {
    scaleX: 1,
    transition: { duration: 0.8, ease: [0.22, 1, 0.36, 1], delay: 0.2 },
  },
};

function Craft() {
  const headingRef = useRef(null);
  const sectionRef = useRef(null);

  useEffect(() => {
    const para = headingRef.current;
    if (!para) return;

    // character-split GSAP heading animation
    let clutter = "";
    para.textContent.split("").forEach((char) => {
      clutter += char === " "
        ? `<span style="display:inline-block">&nbsp;</span>`
        : `<span style="display:inline-block">${char}</span>`;
    });
    para.innerHTML = clutter;

    const tl = gsap.timeline({
      scrollTrigger: {
        trigger: sectionRef.current,
        start: "top 80%",
        end: "top 30%",
        scrub: 0.5,
      },
    });
    tl.from(para.querySelectorAll("span"), {
      y: 60,
      opacity: 0,
      duration: 0.4,
      stagger: 0.06,
    });

    return () => ScrollTrigger.getAll().forEach((t) => t.kill());
  }, []);

  return (
    <section
      ref={sectionRef}
      data-color="cyan"
      id="solutions"
      className="craft section w-full"
    >
      {/* ── top label bar ── */}
      <motion.div
        className="w-full px-5 sm:px-10 lg:px-16 pt-16 sm:pt-20 lg:pt-28 pb-8"
        variants={fadeUp}
        custom={0}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, amount: 0.2 }}
      >
        <div className="max-w-[1400px] mx-auto flex items-center gap-4">
          <motion.div
            className="h-px flex-1 bg-white/10"
            variants={lineDraw}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
          />
          <span className="font-[Sansita] text-[11px] tracking-[0.25em] uppercase
            text-[var(--accent-cyan)] font-semibold">
            Our Solutions
          </span>
          <motion.div
            className="h-px flex-1 bg-white/10"
            variants={lineDraw}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
          />
        </div>
      </motion.div>

      {/* ── main content wrapper ── */}
      <div className="max-w-[1400px] mx-auto px-5 sm:px-10 lg:px-16
        pb-16 sm:pb-20 lg:pb-28 flex flex-col gap-10 lg:gap-0 lg:grid lg:grid-cols-[1fr_1.1fr] lg:gap-x-20 xl:gap-x-28"
      >
        {/* ── LEFT: text block — NOT sticky, prevents overlap ── */}
        <div className="ltext flex flex-col gap-6 lg:pt-2">
          {/* paragraph */}
          <motion.p
            className="font-[Sansita] text-[0.95rem] sm:text-[1rem]
              font-medium leading-relaxed text-[var(--text-muted)] max-w-[520px]"
            variants={fadeUp}
            custom={0.05}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, amount: 0.3 }}
          >
            Zehanx Technologies is a premier custom software and AI engineering
            agency. We build secure, robust enterprise systems, government
            platforms, and advanced machine learning models. By simplifying
            complexity, we accelerate digital capacity and drive substantial
            real-world outcomes.
          </motion.p>

          {/* heading — character animation via GSAP */}
          <div className="overflow-hidden">
            <h1
              ref={headingRef}
              className="texthead font-[SansitaReg]
                text-[clamp(2.4rem,6vw,5rem)]
                leading-[1.1] text-[var(--text-light)]"
            >
              We Craft Intelligent Enterprise Software
            </h1>
          </div>

          {/* stats row */}
          <motion.div
            className="flex flex-wrap gap-6 sm:gap-10 py-2"
            variants={fadeUp}
            custom={0.2}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, amount: 0.3 }}
          >
            {[
              { value: "50+", label: "Projects Delivered" },
              { value: "99%", label: "Client Satisfaction" },
              { value: "8+", label: "Years Experience" },
            ].map((stat) => (
              <div key={stat.label} className="flex flex-col gap-0.5">
                <span className="font-[SansitaBold] text-[1.8rem] sm:text-[2.2rem]
                  leading-none text-[var(--text-light)]">
                  {stat.value}
                </span>
                <span className="font-[Sansita] text-[0.78rem] text-[var(--text-muted)] tracking-wide">
                  {stat.label}
                </span>
              </div>
            ))}
          </motion.div>

          {/* CTA */}
          <motion.div
            variants={fadeUp}
            custom={0.3}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, amount: 0.3 }}
          >
            <Button bgColor="bg-none" text="OUR SOLUTIONS" />
          </motion.div>
        </div>

        {/* ── RIGHT: cards grid ── */}
        <div className="w-full">
          <Card />
        </div>
      </div>
    </section>
  );
}

export default Craft;
