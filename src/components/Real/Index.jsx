
import img1 from '../../assets/images/real4.jpg'
import img2 from '../../assets/images/real2.jpg'
import img3 from '../../assets/images/real3.jpg'
import img4 from '../../assets/images/real1.jpg'
import img5 from '../../assets/images/real5.jpg'
import icon1 from '../../assets/images/icon1.png';
import icon2 from '../../assets/images/icon2.png';
import icon3 from '../../assets/images/icon3.png';
import img6 from '../../assets/images/real6.jpg'
import img7 from '../../assets/images/real7.jpg'
import img8 from '../../assets/images/real8.jpg'
import img9 from '../../assets/images/real9.jpg'
import { useRef} from 'react';
import { gsap } from "gsap";
import { Power2,} from 'gsap/gsap-core';
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { useGSAP } from '@gsap/react';

gsap.registerPlugin(ScrollTrigger);

function Real() {

    const container = useRef(null);
    
    useGSAP(() => {
        const tl = gsap.timeline({
            scrollTrigger: {
            trigger: ".real",
            start: "top top",
            end: "bottom bottom",
            scrub: 1,
            }
         });
         tl.to(".slide", {
            xPercent: -300,
            ease: Power2
         })
         tl.to(".image7", {
            opacity: 0,
         })              
    }, container );


  return (
    <div  
        data-color="salmon" 
        ref={container} 
        className="real section sm:w-full  px-8 sm:px-8 mt-32"
        id="about"
    >
      <div 
        className="cont h-[400vh] relative"
    >
        <div  
            className="slides w-full h-[130vh] overflow-hidden
             sticky top-0 left-0 flex"
        >
            {/* 1st slide */}
            <div 
                className="slide w-full flex sm:flex items-center 
                justify-center h-screen flex-shrink-0 "
            >
                
                <div 
                    className="text1 font-[SansitaReg] text-[7vh] leading-[8vh]
                    sm:text-[15vh] sm:leading-[18vh]"
                >
                    <h1 className="">
                        Real Talk,    
                    </h1>
                    <h1 className="">
                        Real Impact
                    </h1>
                </div>
                <div 
                    className="image absolute w-[20vh] h-[20vh] sm:w-[45vh] sm:h-[45vh] -translate-y-2/5
                    top-1/2 sm:-translate-y-1/2 translate-x-1/2
                    right-0 "
                >
                    <img 
                        src={img1}
                        className="w-full h-full object-contain rounded-full"
                    />

                </div>
            </div>
            {/* 2nd slide */}
            <div   
                className="slide w-full h-screen sm:flex flex items-center 
                justify-center flex-shrink-0 relative "
            >
                <div 
                    className="image absolute w-[20vh] h-[20vh] sm:w-[40vh] sm:h-[40vh] 
                    top-10 -translate-y-1/2 right-1/6"
                >
                    <img 
                        src={img2}
                        className="w-full h-full object-contain rounded-full"
                    />
                </div>
                <div className='w-[90%] sm:w-[60%] text-center font-[SansitaReg] relative'>
                    <h3 
                        className='font-[Sansita] w-full text-left font-semibold 
                        tracking-tight text-[2.5vh] sm:w-1/3 font-medium text-white'
                    >
                        We build software that scales to optimize workflows and processes for users globally.
                    </h3>
                    <h1 className=' font-semibold text-[10vh] sm:text-[20vh] leading-none text-white '>
                        10K+
                    </h1>
                    <h3 className=' text-[2vh] sm:text-[5vh] font-semibold sm:leading-[7vh]'>
                        Hand-written logs processed monthly via custom fine-tuned Vision LLMs.
                    </h3>    
                </div>
                <div 
                    className="image absolute w-[12vh] sm:w-[20vh] sm:h-[20vh] top-4/5 bottom-[10vh]
                    sm:top-2/3 sm:left-1/4 -translate-x-1/2"
                >
                    <img 
                        src={img4}
                        className="w-full h-full object-contain rounded-full"
                    />
                </div>
                <div 
                    className="image absolute w-[20vh] sm:w-[45vh] sm:h-[45vh] first-letter: 
                    sm:bottom-0 sm:right-1/6 right-0 sm:-translate-x-2/3 sm:translate-y-1/2
                    translate-y-[20vh]"
                >
                    <img 
                        src={img3}
                        className="w-full h-full object-contain rounded-full"
                    />
                </div>
                <div 
                    className="image absolute w-[20vh] top-10 -right-2/3 sm:w-[45vh] sm:h-[45vh] sm:top-1/2 
                    sm:-translate-y-1/2 sm:translate-x-1/2
                    sm:right-0 "
                >
                    <img 
                        src={img5}
                        className="w-full h-full object-contain rounded-full"
                    />

                </div>
            </div>
            
            {/* 3rd slide */}
            <div 
                className="slide w-full h-screen flex sm:flex items-center justify-center 
                flex-shrink-0  relative"
            >
                <div 
                    className="image absolute w-[20vh] h-[20vh] sm:w-[45vh] sm:h-[45vh]  
                    sm:top-20 sm:-translate-x-1/5 bottom-[20vh] right-[20vh]
                    sm:-translate-y-1/2 sm:right-1/6"
                >
                    <img 
                        src={icon2}
                        className="w-full h-full object-contain rounded-full"
                    />
                </div>
                <div className='w-[90%] sm:w-[60%] text-center font-[SansitaReg] relative'>
                    <h3 
                        className='font-[Sansita] w-full text-left font-semibold tracking-tight 
                        text-[2vh] sm:text-[2.5vh] font-medium text-white'
                    >
                        We build tools that automate admin tasks and reduce operations workload.
                    </h3>
                    <h1 className=' font-semibold text-[10vh] sm:text-[20vh] leading-none text-white'>
                        70%
                    </h1>
                    <h3 className='text-[3.5vh] leading-[4vh] sm:text-[6vh] font-semibold sm:leading-[6vh]'>
                        Workload Reduction in Administration.
                    </h3>    
                </div>
                <div 
                    className="image absolute w-[30vh] sm:w-[60vh] sm:h-[60vh] bottom-10 right-2/4 
                    translate-y-1/3 -translate-x-1/5"
                >
                    <img 
                        src={icon1}
                        className="w-full h-full object-contain rounded-full"
                    />
                </div>
                <div 
                    className="hidden sm:inline-block image absolute w-[20vh] sm:w-[50vh] sm:h-[50vh] top-1/2 -translate-y-1/4 translate-x-1/3
                    right-0 "
                >
                    <img 
                        src={icon3}
                        className="w-full h-full object-contain rounded-full"
                    />

                </div>
            </div>

            {/* 4th slide */}
            <div 
                className="slide w-full h-screen flex  sm:flex items-center justify-center 
                relative flex-shrink-0 "
            >
                <div 
                    className="hidden sm:inline-block image absolute w-[20vh] h-[20vh] sm:w-[40vh] sm:h-[40vh] top-10 translate-x-1/2
                    -translate-y-1/2 right-2/3"
                >
                    <img 
                        src={img6}
                        className="w-full h-full object-contain rounded-full"
                    />
                </div>
                <div className='w-[90%] sm:w-[60%] text-center font-[SansitaReg] relative'>                   
                    <h1 className=' font-semibold text-[10vh] sm:text-[20vh] leading-none text-white'>
                        3x
                    </h1>
                    <h3 className='text-[2.5vh] leading-[3vh] sm:text-[6vh] font-semibold sm:leading-[8vh]'>
                        Faster Lead Response Times via <br/> Autonomous AI Agents.
                    </h3>    
                </div>
                <div 
                    className="image7 absolute sm:w-[45vh] sm:h-[45vh] bottom-0 right-2/4 
                    translate-y-1/2 -translate-x-1/2"
                >
                    <img 
                        src={img7}
                        className="w-full h-full object-contain rounded-full"
                    />
                </div>
                <div 
                    className="image absolute top-0 -right-10 w-[20vh] h-[20vh] sm:w-[45vh] sm:h-[45vh] sm:top-0 sm:translate-y-1/5
                    sm:translate-x-2/3  sm:right-0 "
                >
                    <img 
                        src={img9}
                        className="w-full h-full object-contain rounded-full"
                    />
                </div>
                <div 
                    className="image absolute w-[15vh] h-[15vh] top-2/3 translate-y-1/5
                    -translate-x-1/3  right-1/4 "
                >
                    <img 
                        src={img8}
                        className="w-full h-full object-contain rounded-full"
                    />
                </div>
            </div>   
        </div>
      </div>
    </div>
  )
}

export default Real;
