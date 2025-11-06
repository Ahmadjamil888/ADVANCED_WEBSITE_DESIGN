import { redirect } from "next/navigation";
import React from "react";
import { Column, Heading, Text, Button } from "@/once-ui/components";

export default function ByteAIEntry() {
  const url = process.env.NEXT_PUBLIC_BYTE_AI_URL;
  if (url && typeof window === "undefined") {
    // On server, perform a redirect if configured
    redirect(url);
  }

  return (
    <Column maxWidth="m" gap="m">
      <Heading variant="display-strong-s">Byte‑AI Studio</Heading>
      <Text onBackground="neutral-weak">
        This route is the entry point to the Byte‑AI (vibe) app. If you have a deployment
        URL configured, you will be redirected automatically.
      </Text>
      {url ? (
        <Button href={url} variant="primary" arrowIcon>
          Open Byte‑AI
        </Button>
      ) : (
        <>
          <Text onBackground="neutral-weak">
            Set NEXT_PUBLIC_BYTE_AI_URL to your Byte‑AI deployment (the vibe app) to enable redirect.
          </Text>
        </>
      )}
    </Column>
  );
}
