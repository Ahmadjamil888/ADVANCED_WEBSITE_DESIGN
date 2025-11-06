"use client";
import React, { useEffect } from "react";
import Link from "next/link";
import { Column, Heading, Text, Flex, Card, Button, RevealFx, Spinner } from "@/once-ui/components";
import { useAuth } from "@/contexts/AuthContext";
import { useRouter } from "next/navigation";

export default function AISelectionPage() {
  const { user, loading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!loading && !user) {
      router.push("/login");
    }
  }, [loading, user, router]);

  if (loading || !user) {
    return (
      <Column maxWidth="m" gap="l" horizontal="center" paddingY="32">
        <Spinner />
      </Column>
    );
  }

  return (
    <>
      <Column maxWidth="m" gap="l" horizontal="center">
        <RevealFx translateY="4">
          <Heading variant="display-strong-s">Choose your experience</Heading>
        </RevealFx>
        <Text onBackground="neutral-weak">Pick an option to get started.</Text>

        <Flex gap="16" wrap horizontal="center">
          <Card>
            <Column gap="m" padding="16" minWidth="280">
              <Heading as="h2" variant="heading-default-m">Byte AI</Heading>
              <Text onBackground="neutral-weak">
                Build AI-powered web apps in a sandbox. This opens the Byte‑AI (vibe) experience.
              </Text>
              <Button href="/byte-ai" variant="primary" size="m" arrowIcon>
                Open Byte AI
              </Button>
            </Column>
          </Card>

          <Card>
            <Column gap="m" padding="16" minWidth="280">
              <Heading as="h2" variant="heading-default-m">Zeh‑AI</Heading>
              <Text onBackground="neutral-weak">
                Create and manage your own AI models in the AI workspace.
              </Text>
              <Button href="/ai-workspace" variant="secondary" size="m" arrowIcon>
                Open Zeh‑AI
              </Button>
            </Column>
          </Card>
        </Flex>
      </Column>
    </>
  );
}
