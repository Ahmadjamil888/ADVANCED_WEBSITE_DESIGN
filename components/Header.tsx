"use client";
import Link from "next/link";
import Image from "next/image";
import { useState } from "react";
import { ArrowRight } from "@/public";
import { Button } from "@/components";
import { navVariants } from "@/motion";
import { MobileMenu } from "@/animations";
import { navigationItems } from "@/constants";
import { arrowRightWhite, logo } from "@/public";
import { motion, useMotionValueEvent, useScroll } from "framer-motion";

export default function Navbar() {
  const { scrollY } = useScroll();
  const [hidden, setHidden] = useState(false);
  const [productsDropdown, setProductsDropdown] = useState(false);

  useMotionValueEvent(scrollY, "change", (latest) => {
    setHidden(latest > 0);
  });



  return (
    <>
      <motion.div
        initial="initial"
        whileInView="enter"
        variants={navVariants}
        className="fixed w-full top-0 z-50 xm:hidden sm:hidden"
      >
        {/* Top Banner */}
        <motion.div
          className="w-full flex items-center justify-center gap-3 py-3 bg-black"
          variants={navVariants}
          animate={hidden ? "hidden" : "vissible"}
        >
          <div className="xm:hidden sm:hidden">
            <h1 className="text-[#FFFFFF99] text-[18px] font-normal leading-tight">
              Innovative B2B software solutions in AI, ML, Cybersecurity and Enterprise Systems - Zehanx Technologies
            </h1>
          </div>
          <div className="flex gap-2 items-center">
            <a
              href="/contact"
              className="text-white text-[16px] leading-tight font-normal"
            >
              Get in touch
            </a>
            <Image
              src={arrowRightWhite}
              alt="arrowRightWhite"
              width={20}
              height={20}
              className="text-white"
            />
          </div>
        </motion.div>

        {/* Navbar */}
        <motion.div
          className="w-full flex items-center justify-between gap-2 padding-x py-3 backdrop-blur-sm"
          animate={hidden ? { y: -48 } : { y: 0 }}
          transition={{ duration: 0.5, ease: "easeInOut" }}
        >
          <Link href="/">
            <Image src={logo} alt="logo" width={40} height={40} className="cursor-pointer" />
          </Link>

          <div className="flex items-center gap-4 xm:hidden sm:hidden">
            {navigationItems.map((item) =>
              item.title === "Products" ? (
                <div
                  key={item.id}
                  className="relative"
                  onMouseEnter={() => setProductsDropdown(true)}
                  onMouseLeave={() => setProductsDropdown(false)}
                >
                  <div className="flex items-center gap-1 cursor-pointer">
                    <span className="text-[18px] font-normal leading-tight text-[#00000099]">
                      {item.title}
                    </span>
                    <motion.div
                      className="transition-transform duration-200"
                      animate={{ y: productsDropdown ? -2 : 0 }}
                    >
                      <Image
                        src={ArrowRight}
                        alt="arrow-down"
                        width={16}
                        height={16}
                        className="transform rotate-90"
                      />
                    </motion.div>
                  </div>

                  {productsDropdown && (
                    <div className="absolute top-full left-0 mt-2 w-64 bg-white rounded-md shadow-sm border border-gray-100 py-3 z-50">
                      <div className="px-4 py-2 text-[10px] text-gray-400 uppercase font-semibold tracking-wider">
                        Software Products
                      </div>
                      <div className="px-4 py-3 flex flex-col gap-1">
                        <div className="text-black font-medium text-sm">
                          Daak Khana
                        </div>
                        <div className="text-[10px] text-gray-400">
                          Early Development - Beta Launching Soon
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <Link
                  href={item.href}
                  key={item.id}
                  className="text-[18px] font-normal leading-tight text-[#00000099]"
                >
                  {item.title}
                </Link>
              )
            )}

            <Button className="text-white bg-black px-4 py-2" title="Contact Us" href="/contact" />
          </div>
        </motion.div>
      </motion.div>

      {/* Mobile Menu */}
      <div className="fixed w-full top-0 z-50 hidden xm:block sm:block">
        <MobileMenu />
      </div>
    </>
  );
}
