"use client";
import React from "react";
import { Heading, Flex, Text, Button, Avatar, RevealFx, Column, Badge, Row, Card } from "@/once-ui/components";
import { Mailchimp } from "@/components";
import { baseURL } from "@/app/resources";
import { home, about, person, newsletter } from "@/app/resources/content";
import { Schema } from "@/once-ui/modules";


export default function Home() {
  return (
    <>
      <style jsx global>{`
        /* Mobile Responsive Styles */
        @media (max-width: 768px) {
          .mobile-responsive {
            padding: 16px !important;
          }
          .mobile-text-center {
            text-align: center !important;
          }
          .mobile-small-gap {
            gap: 12px !important;
          }
          .mobile-card-stack {
            flex-direction: column !important;
          }
          .mobile-full-width {
            width: 100% !important;
          }
          .mobile-padding-sm {
            padding: 12px !important;
          }
          .mobile-margin-sm {
            margin: 8px 0 !important;
          }
          .mobile-font-sm {
            font-size: 14px !important;
            line-height: 1.4 !important;
          }
          .mobile-heading-sm {
            font-size: 24px !important;
            line-height: 1.2 !important;
          }
          .mobile-display-sm {
            font-size: 32px !important;
            line-height: 1.1 !important;
          }
        }
        
        @media (max-width: 480px) {
          .mobile-responsive {
            padding: 12px !important;
          }
          .mobile-display-sm {
            font-size: 28px !important;
          }
          .mobile-heading-sm {
            font-size: 20px !important;
          }
          .mobile-font-sm {
            font-size: 13px !important;
          }
        }
      `}</style>
      
      <Column maxWidth="m" gap="xl" horizontal="center" className="mobile-responsive">
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
      <Column fillWidth paddingY="32" gap="m" className="mobile-responsive">
        <Column maxWidth="s" gap="m" className="mobile-text-center">
          <RevealFx fillWidth horizontal="start" paddingTop="16" paddingBottom="32" paddingLeft="12" className="mobile-text-center">
            <Badge background="brand-alpha-weak" paddingX="12" paddingY="4" onBackground="neutral-strong" textVariant="label-default-s" arrow={false}>
              <Row paddingY="2">Welcome to zehanxtech</Row>
            </Badge>
          </RevealFx>
          <RevealFx translateY="4" fillWidth horizontal="start" paddingBottom="16" className="mobile-text-center">
            <Heading wrap="balance" variant="display-strong-l" className="mobile-display-sm">
              World's first multi agent platform
            </Heading>
          </RevealFx>
          <RevealFx translateY="8" delay={0.2} fillWidth horizontal="start" paddingBottom="32" className="mobile-text-center">
            <Text wrap="balance" onBackground="neutral-weak" variant="heading-default-xl" className="mobile-font-sm">
              Revolutionary AI technology that autonomously creates, trains, and deploys custom AI models. Simply describe what you want, and our AI builds it for you - from sentiment analysis to computer vision models.
            </Text>
          </RevealFx>
          <RevealFx paddingTop="12" delay={0.4} horizontal="start" paddingLeft="12" className="mobile-text-center">
            <Button
              id="about"
              data-border="rounded"
              href={about.path}
              variant="secondary"
              size="m"
              arrowIcon
              className="mobile-full-width"
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
      <RevealFx delay={0.2} fillWidth horizontal="center" paddingTop="32" paddingBottom="32" className="mobile-responsive">
        <Column maxWidth="s" gap="m" className="mobile-text-center">
          <Heading as="h2" variant="display-strong-s" className="mobile-heading-sm">Technologies & Expertise</Heading>
          <Flex gap="12" wrap vertical="center" className="mobile-small-gap">
            <Badge background="brand-alpha-weak" textVariant="label-default-s" className="mobile-padding-sm">AI & Machine Learning</Badge>
            <Badge background="brand-alpha-weak" textVariant="label-default-s" className="mobile-padding-sm">React & Next.js</Badge>
            <Badge background="brand-alpha-weak" textVariant="label-default-s" className="mobile-padding-sm">TypeScript</Badge>
            <Badge background="brand-alpha-weak" textVariant="label-default-s" className="mobile-padding-sm">Node.js</Badge>
            <Badge background="brand-alpha-weak" textVariant="label-default-s" className="mobile-padding-sm">Python</Badge>
            <Badge background="brand-alpha-weak" textVariant="label-default-s" className="mobile-padding-sm">Neural Networks</Badge>
            <Badge background="brand-alpha-weak" textVariant="label-default-s" className="mobile-padding-sm">Cloud Solutions</Badge>
            <Badge background="brand-alpha-weak" textVariant="label-default-s" className="mobile-padding-sm">UI/UX Design</Badge>
          </Flex>
        </Column>
      </RevealFx>

      {/* Services Section */}
      <RevealFx delay={0.3} fillWidth horizontal="center" paddingTop="32" paddingBottom="32" className="mobile-responsive">
        <Column maxWidth="s" gap="m" className="mobile-text-center">
          <Heading as="h2" variant="display-strong-s" className="mobile-heading-sm">Revolutionary AI Technology</Heading>
          <Flex gap="16" wrap className="mobile-card-stack mobile-small-gap">
            <Card className="mobile-full-width mobile-padding-sm">
              <Heading variant="heading-default-m" className="mobile-font-sm">🤖 AI Model Generation</Heading>
              <Text onBackground="neutral-weak" className="mobile-font-sm">Describe any AI model you need, and our AI will generate, train, and deploy it automatically.</Text>
            </Card>
            <Card className="mobile-full-width mobile-padding-sm">
              <Heading variant="heading-default-m" className="mobile-font-sm">🚀 Instant Deployment</Heading>
              <Text onBackground="neutral-weak" className="mobile-font-sm">Deploy your custom AI models to Hugging Face Hub with one click - no coding required.</Text>
            </Card>
            <Card className="mobile-full-width mobile-padding-sm">
              <Heading variant="heading-default-m" className="mobile-font-sm">🎯 Any Use Case</Heading>
              <Text onBackground="neutral-weak" className="mobile-font-sm">From sentiment analysis to computer vision - create AI models for any business need.</Text>
            </Card>
          </Flex>
        </Column>
      </RevealFx>

      {/* Timeline Section */}
      <RevealFx delay={0.4} fillWidth horizontal="center" paddingTop="32" paddingBottom="32" className="mobile-responsive">
        <Column maxWidth="s" gap="m" className="mobile-text-center">
          <Heading as="h2" variant="display-strong-s" className="mobile-heading-sm">Our Journey</Heading>
          <Flex direction="column" gap="12" className="mobile-small-gap">
            <Card className="mobile-full-width mobile-padding-sm">
              <Heading variant="heading-default-s" className="mobile-font-sm">2024 - Present</Heading>
              <Text onBackground="neutral-weak" className="mobile-font-sm">Founded zehanxtech with a mission to build AI for the betterment of humanity.</Text>
            </Card>
            <Card className="mobile-full-width mobile-padding-sm">
              <Heading variant="heading-default-s" className="mobile-font-sm">2023 - 2024</Heading>
              <Text onBackground="neutral-weak" className="mobile-font-sm">Research and development in AI technologies and modern web solutions.</Text>
            </Card>
            <Card className="mobile-full-width mobile-padding-sm">
              <Heading variant="heading-default-s" className="mobile-font-sm">2022 - 2023</Heading>
              <Text onBackground="neutral-weak" className="mobile-font-sm">Building expertise in machine learning, neural networks, and advanced web development.</Text>
            </Card>
          </Flex>
        </Column>
      </RevealFx>

      {/* Testimonials Section */}
      <RevealFx delay={0.5} fillWidth horizontal="center" paddingTop="32" paddingBottom="32" className="mobile-responsive">
        <Column maxWidth="s" gap="m" className="mobile-text-center">
          <Heading as="h2" variant="display-strong-s" className="mobile-heading-sm">Client Testimonials</Heading>
          <Flex gap="16" wrap className="mobile-card-stack mobile-small-gap">
            <Card className="mobile-full-width mobile-padding-sm">
              <Text onBackground="neutral-weak" className="mobile-font-sm">
                "zehanxtech delivered an exceptional AI solution that transformed our business operations. Highly professional and innovative!"
              </Text>
              <Text variant="label-default-s" style={{ marginTop: 8 }} className="mobile-font-sm">— Sarah Johnson, Tech Director</Text>
            </Card>
            <Card className="mobile-full-width mobile-padding-sm">
              <Text onBackground="neutral-weak" className="mobile-font-sm">
                "Their expertise in both AI and web development is outstanding. They created exactly what we envisioned."
              </Text>
              <Text variant="label-default-s" style={{ marginTop: 8 }} className="mobile-font-sm">— Michael Chen, Startup Founder</Text>
            </Card>
          </Flex>
        </Column>
      </RevealFx>

      {/* Contact Section */}
      <RevealFx delay={0.6} fillWidth horizontal="center" paddingTop="32" paddingBottom="32" className="mobile-responsive">
        <Column maxWidth="s" gap="m" className="mobile-text-center">
          <Heading as="h2" variant="display-strong-s" className="mobile-heading-sm">Get In Touch</Heading>
          <Text onBackground="neutral-weak" className="mobile-font-sm">
            Ready to transform your business with AI? Contact us today.
          </Text>
          <Flex gap="16" wrap vertical="center" className="mobile-card-stack mobile-small-gap">
            <Card className="mobile-full-width mobile-padding-sm">
              <Heading variant="heading-default-s" className="mobile-font-sm">Email</Heading>
              <Text onBackground="neutral-weak" className="mobile-font-sm">zehanxtech@gmail.com</Text>
            </Card>
            <Card className="mobile-full-width mobile-padding-sm">
              <Heading variant="heading-default-s" className="mobile-font-sm">Phone</Heading>
              <Text onBackground="neutral-weak" className="mobile-font-sm">+92 344 2693910</Text>
            </Card>
            <Card className="mobile-full-width mobile-padding-sm">
              <Heading variant="heading-default-s" className="mobile-font-sm">Location</Heading>
              <Text onBackground="neutral-weak" className="mobile-font-sm">Gujranwala, Pakistan</Text>
            </Card>
          </Flex>
        </Column>
      </RevealFx>

      {/* Newsletter Section */}
      {newsletter.display && (
        <RevealFx delay={0.7} fillWidth horizontal="center" paddingTop="32" paddingBottom="32" className="mobile-responsive">
          <Column maxWidth="s" gap="m" className="mobile-text-center">
            <Heading variant="display-strong-s" className="mobile-heading-sm">Stay Updated</Heading>
            <Text onBackground="neutral-weak" className="mobile-font-sm">
              Subscribe to our newsletter for AI insights, tech updates, and project showcases.
            </Text>
            <Mailchimp newsletter={newsletter} />
          </Column>
        </RevealFx>
      )}
    </Column>
    </>
  );
}