"use client";

import { usePathname } from "next/navigation";
import { Header } from "@/components/Header";
import { Footer } from "@/components/Footer";
import { Flex } from "@/once-ui/components";

export function Chrome({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const hideChrome = pathname?.startsWith("/ai-workspace") || 
                     pathname?.startsWith("/ai-model-generator") ||
                     pathname?.startsWith("/login");

  if (hideChrome) {
    return (
      <Flex zIndex={0} fillWidth paddingY="0" paddingX="0" horizontal="center" flex={1} style={{ minHeight: "100vh" }}>
        <Flex horizontal="center" fillWidth minHeight="0" style={{ width: "100%", height: "100%" }}>
          {children}
        </Flex>
      </Flex>
    );
  }

  return (
    <>
      <Header />
      <Flex zIndex={0} fillWidth paddingY="l" paddingX="l" horizontal="center" flex={1}>
        <Flex horizontal="center" fillWidth minHeight="0">
          {children}
        </Flex>
      </Flex>
      <Footer />
    </>
  );
}

