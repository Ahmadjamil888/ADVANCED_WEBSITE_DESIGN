// testimonial 2 - no image needed
import { useEffect } from 'react';
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

gsap.registerPlugin(ScrollTrigger);



function Para2() {
    useEffect(() => {
        var clutter = "";
        const para = document.querySelector(".textpara2")
        const characters = para.textContent.split("")
        characters.forEach(function(e) {
           
            clutter += `<span>${e}</span>`
        })
        para.innerHTML = clutter;   
        gsap.set(".textpara2 span", {opacity: .1})
        gsap.to(".textpara2 span", {
            scrollTrigger: {
                trigger: ".para2",
                start: "top 60%",
                end: "bottom 90%",
                scrub: 1,
            },
            opacity: 1, 
            stagger: .03,   
        })
    },[]);
  return (
    <div data-color="white" className="para2 section w-full flex items-center mt-32 px-8
    justify-center -translate-y-1/5 ">
        <div className="text sm:w-[80%] flex flex-col items-center sm:items-end justify-center ">
            <div className='hidden w-[40%] sm:flex items-center justify-center mb-12'>
                <hr className='bg-zinc-400 w-[20%] h-[.3vh]' />
            </div>    
            <h3 className='textpara2 sm:w-[40%] text-purple-400 font-[Sansita] text-[2.4vh] sm:text-[3.5vh] font-medium text-center tracking-wide leading-[5vh] mb-10'> Zehanx built us an AI sales agent that handles all inbound queries 24/7. Our lead response time tripled and we eliminated manual follow-up entirely. Honestly the ROI in the first month paid for the entire project. The depth of engineering and how fast they moved was unlike anything we’ve worked with before.</h3>
            <div className="pers w-full sm:w-[40%] flex flex-col items-center justify-center gap-2">
                <h1 className='text-[2.8vh] sm:text-[3.8vh] font-medium'>Fiaz Ahmad</h1>
                <h3 className='text-zinc-400 text-[2.4vh]'>CEO @ Janjua Global</h3>
            </div>
        </div>  
    </div>
  )
}

export default Para2
