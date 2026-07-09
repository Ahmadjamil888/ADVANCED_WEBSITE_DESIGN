
import styles from './Style.module.css';
import { LuArrowUpRight } from "react-icons/lu";
import Button from  '../Button'

function Footer() {
  return (
    <div className="section w-full sm:mt-30 ">
      <div className="topfoot bg-[#0f1026] pt-16 pb-16 sm:flex items-center justify-between border-t border-zinc-800">
        <div className="left sm:w-1/2 px-8 sm:px-8">
          <div className="first order-1 font-[SansitaReg] text-[6vh] leading-[7vh] sm:text-[4.2rem] w-full sm:leading-[5rem] sm:mb-10 text-white">
            <h1>Partner with Zehanx.</h1>
          </div>
          <a 
            href="https://cal.com/zehanx-technologies-official" 
            target="_blank" 
            rel="noopener noreferrer" 
            className='middle order-3 sm:order-2 w-[22vh] text-center mt-6 px-4 py-4 sm:w-fit sm:px-8 sm:py-[1.6vh] bg-[#8b5cf6] text-white whitespace-nowrap block z-[9]'
          >
            <div 
              className= {`${styles.masker} flex items-center 
              gap-2 overflow-hidden 
              relative cursor-pointer justify-center`}
            >
              <span 
                className={`${styles.spanMask} font-[Sansita] 
                text-[2.1vh] capitalize tracking-normal 
                font-semibold`}
              >
                BOOK PROJECT
              </span>
              <LuArrowUpRight 
                style={{
                  fontSize: "24px", 
                  color: "#fff", 
                }} 
                className={`${styles.iconMask}`} 
              /> 
            </div>
          </a>
        </div>

        <div className="right order-2 sm:order-3 flex sm:w-1/2 items-center justify-between sm:px-10">
          <div className="rght1 sm:w-2/3 flex flex-col items-end px-8 mt-6">
            <h3 className="text-[2.4vh] sm:text-[1.4rem] font-[Sansita] font-medium text-zinc-300">
              Find out how Zehanx's custom software and AI systems can empower and scale your operations.
            </h3>
          </div>
          <div className="hidden sm:inline-block rght2 relative w-[150px] h-[150px] overflow: hidden;">
            <div className={`${styles.loopFoot}`}></div>
            <div className="icon w-[120px] h-[120px] rounded-full border-zinc-700 border-[1px] px-10 py-10 absolute top-1/2 left-1/2 -translate-y-1/2 -translate-x-1/2 overflow-hidden">
              <div className="w-10 h-10 absolute top-1/2 left-1/2 -translate-y-1/2 -translate-x-1/2 text-white">
                <svg width="100%" height="100%" viewBox="0 0 32 38" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M16 0.703124C16 9.53968 8.83656 16.7031 0 16.7031" stroke="currentColor" strokeWidth="2"></path>
                  <path d="M16 0.703124C16 9.53968 23.1634 16.7031 32 16.7031" stroke="currentColor" strokeWidth="2"></path>
                  <path d="M16 0.703125L16 37.2746" stroke="currentColor" strokeWidth="2"></path>
                </svg>
              </div>
            </div>
          </div>
        </div>
      </div>


      <div className="downfoot w-full h-fit bg-[#030712] px-8 py-10 sm:py-24 border-t border-zinc-900">
        <div className='max-w-7xl mx-auto flex flex-col sm:flex-row sm:items-start justify-between gap-12 sm:gap-6'>
          {/* Quick links */}
          <div className='flex gap-16'>
            <div>
              <h4 className='text-cyan-400 font-[Sansita] text-[2.2vh] font-bold mb-4 uppercase tracking-wider'>Navigation</h4>
              {["Home", "Solutions", "About", "Team"].map((item, index) => (
                <div key={index} className='pb-2'>
                  <a href={`#${item.toLowerCase()}`} className='font-[SansitaReg] text-[2.1vh] text-zinc-300 hover:text-white font-semibold transition-colors'>
                    {item}
                  </a>  
                </div>
              ))}
            </div>
            <div>
              <h4 className='text-cyan-400 font-[Sansita] text-[2.2vh] font-bold mb-4 uppercase tracking-wider'>Contact Us</h4>
              <div className='pb-2 font-[SansitaReg] text-[2.1vh] text-zinc-300 font-semibold'>
                <p>Email: <a href="mailto:zehanxtech@gmail.com" className="hover:text-white transition-colors">zehanxtech@gmail.com</a></p>
              </div>
              <div className='pb-2 font-[SansitaReg] text-[2.1vh] text-zinc-300 font-semibold'>
                <p>Phone/WhatsApp: <a href="https://wa.me/923067060074" target="_blank" rel="noopener noreferrer" className="hover:text-white transition-colors">+92 306 7060074</a></p>
              </div>
            </div>
          </div>

          {/* Subscribe */}
          <div className='flex-1 sm:max-w-md'>
            <h1 className='text-white pb-6 font-[SansitaReg] text-[2.8vh] font-semibold'>Join our mailing list for the latest updates.</h1>
            <div className='flex items-center gap-4 w-full'>
              <input 
                type="text" 
                placeholder="Enter your email address" 
                className="bg-zinc-800 text-white border border-zinc-700 w-full h-[5.9vh] px-4 text-[2vh] focus:outline-none focus:border-cyan-400"
              />
              <Button text="SUBSCRIBE" bgColor="bg-[#8b5cf6]" />
            </div>
          </div>

          {/* Socials */}
          <div className='flex flex-col items-start'>
            <h4 className='text-cyan-400 font-[Sansita] text-[2.2vh] font-bold mb-4 uppercase tracking-wider'>Follow Us</h4>
            <div className='flex gap-6 flex-wrap sm:flex-col sm:gap-2'>
              <a href="https://www.instagram.com/zehanxtech_official/" target="_blank" rel="noopener noreferrer" className='font-[SansitaReg] text-[2.1vh] text-zinc-300 hover:text-white font-semibold transition-colors'>
                Instagram
              </a>
              <a href="https://x.com/zehanxtech" target="_blank" rel="noopener noreferrer" className='font-[SansitaReg] text-[2.1vh] text-zinc-300 hover:text-white font-semibold transition-colors'>
                X (Twitter)
              </a>
              <a href="https://www.youtube.com/channel/UCKKv_31GqzUU8Gk_4IRKWrg" target="_blank" rel="noopener noreferrer" className='font-[SansitaReg] text-[2.1vh] text-zinc-300 hover:text-white font-semibold transition-colors'>
                YouTube
              </a>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Footer
