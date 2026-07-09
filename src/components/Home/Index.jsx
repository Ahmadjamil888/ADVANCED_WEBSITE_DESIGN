import video from '../../assets/video/1ENIoa5sjq.mp4'
import Row from '../Row'
import {useEffect, useState, useRef} from 'react';
import {motion, useScroll, useMotionValueEvent, AnimatePresence} from 'framer-motion';
import { gsap } from "gsap";
import styles from './Style.module.css';
import { Power2, Power4 } from 'gsap/gsap-core';
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { useGSAP } from '@gsap/react';
// import { AiOutlineMenu } from "react-icons/ai";

gsap.registerPlugin(ScrollTrigger);

gsap.set(".slidesm", {scale: 5})

function Home() {

    const container = useRef(null);
    const [isMobileOpen, setIsMobileOpen] = useState(false);

    useEffect(() => {
        var clutter = "";
        const para = document.querySelector(".toptext")
        const characters = para.textContent.split("")
        characters.forEach(function(e) {
            
            clutter += `<span>${e}</span>`
        })
        para.innerHTML = clutter;   
        gsap.set(".toptext span", {opacity: .1})
        gsap.to(".toptext span", {
            scrollTrigger: {
                trigger: ".home",
                start: "top 50%",
                end: "bottom 90%",
                scrub: 1,
            },
            opacity: 1, 
            stagger: .03,
            
        })
    },[]);

    
    useGSAP(() => {
        const tl = gsap.timeline({
            scrollTrigger: {
            trigger: ".home",
            start: "top top",
            end: "bottom bottom",
            scrub: .5,
            }
         });
         tl.to(".vdodiv", {
            clipPath: 'circle(0% at 50% 50%)',
            ease: Power4,
          }, "start")
          tl.to(".slidesm", {
            scale: 1,
            ease: Power2,
         }, 'start');
         tl.to(".lft", {
            xPercent: -10,
            stagger: .03,
            ease: Power4,
            duration: 1,
         }, 'start');
         tl.to(".rgt", {
            xPercent: 10,
            stagger: .03,
            ease: Power4,
            duration: 1,
         }, 'start');           
    }, container )

    const {scrollY} = useScroll();
    const [hidden, setHidden] = useState(false);
  
    useMotionValueEvent(scrollY, "change", (latest) => {

        const previous = scrollY.getPrevious() ?? 0;
        console.log(previous, latest);

        if(latest > previous) {
        setHidden(true);
        }
        else {
        setHidden(false);
        }
    });
   
    
    return (
    <div ref={container} data-color="black" className="home section w-full h-[200vh] relative  ">
        <div className='w-full sticky top-0 left-0 '>
            {/* navbar */}
            <motion.div 
                variants={{
                visible: {y: 0},
                hidden: {y: "-100%"},
                }}
                animate={hidden ? "hidden" : "visible"}
                transition={{duration: 0.35, ease: "easeInOut"}}
                className="section w-[100vw] sm:w-full px-6 fixed top-0 left-0 z-[9]"
            >
                <div className="w-full flex sm:flex items-center justify-between  ">
                    <div className="logo w-[15vh] h-[10vh] cursor-pointer z-[9] flex items-center">
                        {/* logo */}
                        <img src="/z-logo-light.png" alt="Zehanx Logo" className="h-10 w-auto object-contain" />
                    </div>                   
                        <div className="hidden md:flex gap-4 items-center z-[9] cursor-pointer ">
                            {["Solutions", "About", "Insight", "Team"].map((item, index) => (
                                <h4 key={index} className={`${styles.links} h-[3vh] relative py-[2.4vh] px-[2.2vh] text-center flex flex-col
                                font-[Sansita] text-[2.1vh] overflow-hidden font-medium leading-[2.5vh]`}> 
                                    <a href={`#${item.toLowerCase()}`} className={`atag ${styles.atag} relative`}>{item} </a>
                                    <a href={`#${item.toLowerCase()}`} className={`atag ${styles.atag} relative`}>{item} </a>                      
                                </h4>   
                            ))}
                            <a 
                              href="https://cal.com/zehanx-technologies-official" 
                              target="_blank" 
                              rel="noopener noreferrer"
                              className="bg-[var(--accent-violet)] text-white px-5 py-2.5 rounded-full font-[Sansita] text-[1.8vh] font-semibold hover:bg-violet-700 transition-colors ml-4 z-[9]"
                            >
                              Book Meeting
                            </a>
                        </div>

                        <button
                            onClick={() => setIsMobileOpen(true)}
                            className="inline-block sm:hidden z-[9] text-white"
                            aria-label="Open menu"
                        >
                            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <line x1="3" y1="6" x2="21" y2="6"></line>
                                <line x1="3" y1="12" x2="21" y2="12"></line>
                                <line x1="3" y1="18" x2="21" y2="18"></line>
                            </svg>
                        </button>
                 
                    
                </div>
            </motion.div>

            <AnimatePresence>
                {isMobileOpen && (
                    <>
                        <motion.div
                            initial={{opacity: 0}}
                            animate={{opacity: 1}}
                            exit={{opacity: 0}}
                            onClick={() => setIsMobileOpen(false)}
                            className="fixed inset-0 bg-black/60 z-[10] md:hidden"
                        />
                        <motion.nav
                            initial={{x: '100%'}}
                            animate={{x: 0}}
                            exit={{x: '100%'}}
                            transition={{type: 'spring', damping: 25, stiffness: 200}}
                            className="fixed top-0 right-0 h-full w-[80%] max-w-sm bg-[#0b0f19] z-[11] md:hidden flex flex-col justify-center items-start gap-8 p-10 shadow-2xl border-l border-zinc-800"
                        >
                            <button
                                onClick={() => setIsMobileOpen(false)}
                                className="absolute top-6 right-6 text-white"
                                aria-label="Close menu"
                            >
                                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <line x1="18" y1="6" x2="6" y2="18"></line>
                                    <line x1="6" y1="6" x2="18" y2="18"></line>
                                </svg>
                            </button>
                            {["Solutions", "About", "Insight", "Team"].map((item) => (
                                <a
                                    key={item}
                                    href={`#${item.toLowerCase()}`}
                                    onClick={() => setIsMobileOpen(false)}
                                    className="font-[Sansita] text-[2.4vh] text-white font-medium hover:text-cyan-400 transition-colors"
                                >
                                    {item}
                                </a>
                            ))}
                            <a
                                href="https://cal.com/zehanx-technologies-official"
                                target="_blank"
                                rel="noopener noreferrer"
                                onClick={() => setIsMobileOpen(false)}
                                className="bg-[var(--accent-violet)] text-white px-6 py-3 rounded-full font-[Sansita] text-[1.9vh] font-semibold hover:bg-violet-700 transition-colors mt-4"
                            >
                                Book Meeting
                            </a>
                        </motion.nav>
                    </>
                )}
            </AnimatePresence>

            <div className='btmtext absolute z-[4] bottom-[4%] left-[25%] text-center sm:text-start sm:bottom-[7%] sm:left-8 w-80 '>
                <h1 className='sm:text-[2vh] font-semibold leading-relaxed'>
                    We build big ideas.<br/>
                    Government & Enterprise Software.<br/>
                    Advanced ML, DL & AI services.
                </h1>
            </div>
            {/* video div */}
            <div 
                
                className={` vdodiv w-full h-screen absolute z-[3] 
                top-0 left-0 overflow-hidden sm:overflow-visible ${styles.vdodiv}`}
            >   
                <video
                    className="absolute w-full h-screen object-cover top-1/2 left-1/2 
                    -translate-x-1/2 -translate-y-1/2"
                    autoPlay
                    loop
                    muted
                    src={video}
                >     
                </video> 
            </div>

            {/* marquee div */}
            <div 
                className="marqueecontainer w-full h-screen 
                relative overflow-hidden "
            >
                {/* /* top Heading div */ }
                <div 
                    className='heading absolute  top-[12%] sm:top-[7%] left-1/2 
                    -translate-x-1/2 w-[90%] max-w-[35rem]'
                >
                    <h2 className='toptext text-[2.2vh] font-[Sansita] tracking-wide font-medium text-center'>Empowering enterprises with intelligent software solutions that are</h2>
                </div>

                <div 
                    
                    className='slidesm absolute scale-[5]  top-1/2 left-1/2
                    -translate-x-1/2 -translate-y-1/2 w-[90%]'
                >    
                    <div className='row'>
                        <Row 
                            translateClass="-translate-x-1/2"  
                            direction="lft"
                        />
                        <Row 
                            translateClass="-translate-x-2/3"  
                            direction="rgt"
                        />
                        <Row 
                            translateClass="-translate-x-1/4"  
                            direction="lft" 
                        />
                        <Row 
                            translateClass="-translate-x-1/3"  
                            direction="rgt"
                        />
                    </div>            
                    
                </div>
            </div>
        </div>
    </div>
  )
}

export default Home
