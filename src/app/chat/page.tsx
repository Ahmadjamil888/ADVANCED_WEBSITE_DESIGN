"use client";

import React, { useEffect } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { useRouter } from "next/navigation";
import Script from "next/script";

export default function ChatRedirect() {
  const { user, loading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!loading) {
      if (user) {
        router.push("/dashboard");
      } else {
        router.push("/login");
      }
    }
  }, [user, loading, router]);

  return (
    <>
      <Script src="https://cdn.tailwindcss.com" />
      <div className="flex h-screen w-full items-center justify-center bg-white">
        <div className="text-lg text-gray-900">Redirecting...</div>
      </div>
    </>
  );
}