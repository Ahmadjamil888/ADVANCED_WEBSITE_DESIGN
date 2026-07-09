// testimonial 1 - no image needed
import { useEffect } from 'react';
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

gsap.registerPlugin(ScrollTrigger);

function Para() {
    useEffect(() => {
        var clutter = "";
        const para = document.querySelector(".textpara")
        const characters = para.textContent.split("")
        characters.forEach(function(e) {
            
            clutter += `<span>${e}</span>`
        })
        para.innerHTML = clutter;   
        gsap.set(".textpara span", {opacity: .1})
        gsap.to(".textpara span", {
            scrollTrigger: {
                trigger: ".para",
                start: "top 70%",
                end: "bottom 90%",
                scrub: 1,
            },
            opacity: 1, 
            stagger: .03,
            
        })
    },[]);

    

  return (
    <div data-color="white" className="para section w-full flex items-center justify-center px-8  ">
        <div className="text sm:w-[80%] flex flex-col items-center sm:items-start justify-between ">
            <div className='hidden w-[50%] sm:flex items-center justify-center mb-12'>
                <hr className='bg-zinc-400 w-[20%] h-[.3vh]' />
            </div>    
            <h3 className='textpara sm:w-[50%] text-cyan-400 font-[Sansita] tracking-wide text-[2.4vh] sm:text-[3.5vh] font-medium text-center leading-[5vh] mb-10'>The team at Zehanx delivered our entire AI-powered LMS — BrainSkills — in record time. Course prep for our medical faculty dropped by 60%, and the AI content generation is genuinely impressive. They didn't just build software; they understood our clinical education needs deeply and shipped a product that our 200+ students actually love using every day.</h3>
            <div className="pers w-full sm:w-[50%] flex flex-col items-center justify-center gap-2">
                <h1 className=' text-[2.8vh] sm:text-[3.8vh] font-medium'>Dr. Rehan Siddiqui</h1>
                <h3 className='text-zinc-400 text-[2.4vh]'>Director of Medical Education @ Usman Hospital</h3>
            </div>
        </div>  
    </div>
  )
}

export default Para
