"use client";
import React from "react";
import { Heading, Flex, Text, Button, Avatar, RevealFx, Column, Badge, Row, Card } from "@/once-ui/components";
import { Mailchimp } from "@/components";
import { baseURL } from "@/app/resources";
import { home, about, person, newsletter } from "@/app/resources/content";
import { Schema } from "@/once-ui/modules";

export default function Home() {
  return (
    <Column maxWidth="m" gap="xl" horizontal="center">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify({
            "@context": "https://schema.org",
            "@type": "WebPage",
            "name": "zehanxtech - AI Systems Development & Web Solutions",
            "description": "Building AI for the betterment of humanity. Custom AI systems, machine learning models, and modern web applications from Gujranwala, Pakistan.",
            "url": `${baseURL}${home.path}`,
            "mainEntity": {
              "@type": "Organization",
              "name": "zehanxtech",
              "description": "AI systems development and web solutions company",
              "url": baseURL,
              "logo": `${baseURL}/logo.jpg`,
              "address": {
                "@type": "PostalAddress",
                "addressLocality": "Gujranwala",
                "addressRegion": "Punjab",
                "addressCountry": "Pakistan"
              },
              "contactPoint": {
                "@type": "ContactPoint",
                "telephone": "+92-344-2693910",
                "email": "zehanxtech@gmail.com",
                "contactType": "customer service"
              }
            },
            "breadcrumb": {
              "@type": "BreadcrumbList",
              "itemListElement": [
                {
                  "@type": "ListItem",
                  "position": 1,
                  "name": "Home",
                  "item": baseURL
                }
              ]
            }
          })
        }}
      />

      {/* Hero Section */}
      <Column fillWidth paddingY="32" gap="m">
        <Column maxWidth="s" gap="m">
          <RevealFx fillWidth horizontal="start" paddingTop="16" paddingBottom="32" paddingLeft="12">
            <Badge background="brand-alpha-weak" paddingX="12" paddingY="4" onBackground="neutral-strong" textVariant="label-default-s" arrow={false}>
              <Row paddingY="2">Welcome to zehanxtech</Row>
            </Badge>
          </RevealFx>
          <RevealFx translateY="4" fillWidth horizontal="start" paddingBottom="16">
            <Heading wrap="balance" variant="display-strong-l">
              {home.headline || "Building AI for Better of Humanity"}
            </Heading>
          </RevealFx>
          <RevealFx translateY="8" delay={0.2} fillWidth horizontal="start" paddingBottom="32">
            <Text wrap="balance" onBackground="neutral-weak" variant="heading-default-xl">
              We're zehanxtech, an AI & Web Development company specializing in cutting-edge artificial intelligence solutions, modern web applications, and innovative digital products that transform businesses.
            </Text>
          </RevealFx>
          <RevealFx paddingTop="12" delay={0.4} horizontal="start" paddingLeft="12">
            <Button
              id="about"
              data-border="rounded"
              href={about.path}
              variant="secondary"
              size="m"
              arrowIcon
            >
              <Flex gap="8" vertical="center">
                {about.avatar?.display && (
                  <Avatar
                    style={{ marginLeft: "-0.75rem", marginRight: "0.25rem" }}
                    src={person.avatar}
                    size="m"
                  />
                )}
                Learn more about us
              </Flex>
            </Button>
          </RevealFx>
        </Column>
      </Column>

      {/* Skills Section */}
      <RevealFx delay={0.2} fillWidth horizontal="center" paddingTop="32" paddingBottom="32">
        <Column maxWidth="s" gap="m">
          <Heading as="h2" variant="display-strong-s">Technologies & Expertise</Heading>
          <Flex gap="12" wrap vertical="center">
            <Badge background="brand-alpha-weak" textVariant="label-default-s">AI & Machine Learning</Badge>
            <Badge background="brand-alpha-weak" textVariant="label-default-s">React & Next.js</Badge>
            <Badge background="brand-alpha-weak" textVariant="label-default-s">TypeScript</Badge>
            <Badge background="brand-alpha-weak" textVariant="label-default-s">Node.js</Badge>
            <Badge background="brand-alpha-weak" textVariant="label-default-s">Python</Badge>
            <Badge background="brand-alpha-weak" textVariant="label-default-s">Neural Networks</Badge>
            <Badge background="brand-alpha-weak" textVariant="label-default-s">Cloud Solutions</Badge>
            <Badge background="brand-alpha-weak" textVariant="label-default-s">UI/UX Design</Badge>
          </Flex>
        </Column>
      </RevealFx>

      {/* Services Section */}
      <RevealFx delay={0.3} fillWidth horizontal="center" paddingTop="32" paddingBottom="32">
        <Column maxWidth="s" gap="m">
          <Heading as="h2" variant="display-strong-s">What We Offer</Heading>
          <Flex gap="16" wrap>
            <Card>
              <Heading variant="heading-default-m">AI Systems Development</Heading>
              <Text onBackground="neutral-weak">Custom AI systems, machine learning models, and intelligent automation solutions.</Text>
            </Card>
            <Card>
              <Heading variant="heading-default-m">Web Development</Heading>
              <Text onBackground="neutral-weak">Modern, scalable web applications using cutting-edge technologies and best practices.</Text>
            </Card>
            <Card>
              <Heading variant="heading-default-m">Digital Innovation</Heading>
              <Text onBackground="neutral-weak">Transformative digital solutions that drive business growth and efficiency.</Text>
            </Card>
          </Flex>
        </Column>
      </RevealFx>

      {/* Timeline Section */}
      <RevealFx delay={0.4} fillWidth horizontal="center" paddingTop="32" paddingBottom="32">
        <Column maxWidth="s" gap="m">
          <Heading as="h2" variant="display-strong-s">Our Journey</Heading>
          <Flex direction="column" gap="12">
            <Card>
              <Heading variant="heading-default-s">2024 - Present</Heading>
              <Text onBackground="neutral-weak">Founded zehanxtech with a mission to build AI for the betterment of humanity.</Text>
            </Card>
            <Card>
              <Heading variant="heading-default-s">2023 - 2024</Heading>
              <Text onBackground="neutral-weak">Research and development in AI technologies and modern web solutions.</Text>
            </Card>
            <Card>
              <Heading variant="heading-default-s">2022 - 2023</Heading>
              <Text onBackground="neutral-weak">Building expertise in machine learning, neural networks, and advanced web development.</Text>
            </Card>
          </Flex>
        </Column>
      </RevealFx>

      {/* Testimonials Section */}
      <RevealFx delay={0.5} fillWidth horizontal="center" paddingTop="32" paddingBottom="32">
        <Column maxWidth="s" gap="m">
          <Heading as="h2" variant="display-strong-s">Client Testimonials</Heading>
          <Flex gap="16" wrap>
            <Card>
              <Text onBackground="neutral-weak">
                "zehanxtech delivered an exceptional AI solution that transformed our business operations. Highly professional and innovative!"
              </Text>
              <Text variant="label-default-s" style={{ marginTop: 8 }}>— Sarah Johnson, Tech Director</Text>
            </Card>
            <Card>
              <Text onBackground="neutral-weak">
                "Their expertise in both AI and web development is outstanding. They created exactly what we envisioned."
              </Text>
              <Text variant="label-default-s" style={{ marginTop: 8 }}>— Michael Chen, Startup Founder</Text>
            </Card>
          </Flex>
        </Column>
      </RevealFx>

      {/* Contact Section */}
      <RevealFx delay={0.6} fillWidth horizontal="center" paddingTop="32" paddingBottom="32">
        <Column maxWidth="s" gap="m">
          <Heading as="h2" variant="display-strong-s">Get In Touch</Heading>
          <Text onBackground="neutral-weak">
            Ready to transform your business with AI? Contact us today.
          </Text>
          <Flex gap="16" wrap vertical="center">
            <Card>
              <Heading variant="heading-default-s">Email</Heading>
              <Text onBackground="neutral-weak">zehanxtech@gmail.com</Text>
            </Card>
            <Card>
              <Heading variant="heading-default-s">Phone</Heading>
              <Text onBackground="neutral-weak">+92 344 2693910</Text>
            </Card>
            <Card>
              <Heading variant="heading-default-s">Location</Heading>
              <Text onBackground="neutral-weak">Gujranwala, Pakistan</Text>
            </Card>
          </Flex>
        </Column>
      </RevealFx>

      {/* Newsletter Section */}
      {newsletter.display && (
        <RevealFx delay={0.7} fillWidth horizontal="center" paddingTop="32" paddingBottom="32">
          <Column maxWidth="s" gap="m">
            <Heading variant="display-strong-s">Stay Updated</Heading>
            <Text onBackground="neutral-weak">
              Subscribe to our newsletter for AI insights, tech updates, and project showcases.
            </Text>
            <Mailchimp newsletter={newsletter} />
          </Column>
        </RevealFx>
      )}
    </Column>
  );
}