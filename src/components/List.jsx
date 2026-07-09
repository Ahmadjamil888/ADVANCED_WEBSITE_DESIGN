
import {useEffect} from 'react';
import { gsap } from "gsap";
import { Power4} from 'gsap/gsap-core';

const data = [
    {key: 1, title: "Ahmad Jamil", role: "Founder & CEO"},
    {key: 2, title: "Ahmad Ibrahim", role: "Co-founder & COO"},
    {key: 3, title: "Umair Amin", role: "Co-founder & CMO"},
]


function List() {

  useEffect(() => {
    const list = document.querySelectorAll('.listelem')

    list.forEach((el) => {  
      el.addEventListener('mousemove', function() {
        gsap.to(this.querySelector(".bluelayer"), {
          height: '100%', 
          ease: Power4,
          duration: .15
        })
      })
      el.addEventListener('mouseleave', function() {
        gsap.to(this.querySelector(".bluelayer"), {
          height: '0%',  
          ease: Power4, 
          duration: .15
        })
      })
      el.addEventListener('mouseenter', function() {
        gsap.to(this.querySelectorAll("h1, h3"), {
          color: '#fff',
          ease: Power4,
          duration: .15
        })
      })
      el.addEventListener('mouseleave', function() {
        // reset colours
        gsap.to(this.querySelector("h1"), {
          color: '',
          ease: Power4,
          duration: .15
        })
        gsap.to(this.querySelector("h3"), {
          color: '',
          ease: Power4,
          duration: .15
        })
      })
    })
  })

  return (
    <div className="list-container">
          {data.map((item, index) => {
            return (
              <div 
                key={index} 
                className="listelem w-full py-[3vh] sm:px-[4vh] sm:py-[6vh] 
                flex justify-between items-center
                border-b-2 border-zinc-800 sm:relative cursor-pointer"
              >
            <div 
              className="relative w-full sm:flex 
              sm:items-center justify-between z-[3]"
            >
              <div className="left sm:flex items-center gap-14 sm:text-5xl">
                <h3 className="hidden sm:inline-block opacity-25 text-zinc-400">0{item.key}</h3>
                <h1 className="text-cyan-400 text-3xl sm:text-white sm:text-[6vh] transition-colors duration-200">{item.title}</h1>
              </div>
              <h3 className="font-[Sansita] text-[3vh] sm:text-[2.4vh] font-medium tracking-tight text-zinc-300 transition-colors duration-200">
                {item.role}
              </h3>   
            </div>
            <div className="hidden sm:inline-block bluelayer sm:absolute top-0 left-0 z-[2] w-full h-0 bg-[#8b5cf6] opacity-30 rounded-sm"></div>
          </div>
        )
      })}
    </div>
  )
}

export default List
