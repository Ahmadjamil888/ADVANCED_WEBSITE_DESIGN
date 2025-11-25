import React from "react";
import { Heading, Flex, Text, Column, Badge, Row, Card } from "@/once-ui/components";
import { baseURL } from "@/app/resources";
import { about, person } from "@/app/resources/content";

export const metadata = {
  title: 'Contact zehanxtech - AI Systems Development Company',
  description: 'Get in touch with zehanxtech for AI systems development, machine learning solutions, and web applications. Located in Gujranwala, Pakistan. Email: zehanxtech@gmail.com, Phone: +92 344 2693910',
  keywords: [
    'contact zehanxtech',
    'AI development contact',
    'Gujranwala AI company',
    'Pakistan AI services',
    'machine learning consultation',
    'AI systems development contact',
    'zehanxtech email',
    'zehanxtech phone'
  ],
  openGraph: {
    title: 'Contact zehanxtech - AI Systems Development Company',
    description: 'Ready to transform your business with AI? Contact zehanxtech for custom AI solutions and web development services.',
    url: `${baseURL}/contact`,
    images: [
      {
        url: '/og-contact.jpg',
        width: 1200,
        height: 630,
        alt: 'Contact zehanxtech - AI Development Services',
      },
    ],
  },
  alternates: {
    canonical: `${baseURL}/contact`,
  },
};

export default function Contact() {
  return (
    <Column maxWidth="m" gap="xl" horizontal="center">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify({
            "@context": "https://schema.org",
            "@type": "ContactPage",
            "name": "Contact zehanxtech - AI Systems Development Company",
            "description": "Get in touch with zehanxtech for AI systems development, machine learning solutions, and web applications.",
            "url": `${baseURL}/contact`,
            "mainEntity": {
              "@type": "Organization",
              "name": "zehanxtech",
              "address": {
                "@type": "PostalAddress",
                "addressLocality": "Gujranwala",
                "addressRegion": "Punjab",
                "addressCountry": "Pakistan"
              },
              "contactPoint": [
                {
                  "@type": "ContactPoint",
                  "telephone": "+92-344-2693910",
                  "contactType": "customer service",
                  "availableLanguage": ["English", "Urdu"],
                  "hoursAvailable": {
                    "@type": "OpeningHoursSpecification",
                    "dayOfWeek": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                    "opens": "09:00",
                    "closes": "18:00"
                  }
                },
                {
                  "@type": "ContactPoint",
                  "email": "zehanxtech@gmail.com",
                  "contactType": "customer service"
                }
              ]
            },
            "breadcrumb": {
              "@type": "BreadcrumbList",
              "itemListElement": [
                {
                  "@type": "ListItem",
                  "position": 1,
                  "name": "Home",
                  "item": baseURL
                },
                {
                  "@type": "ListItem",
                  "position": 2,
                  "name": "Contact",
                  "item": `${baseURL}/contact`
                }
              ]
            }
          })
        }}
      />

      {/* Hero Section */}
      <Column fillWidth paddingY="32" gap="m">
        <Column maxWidth="s" gap="m" horizontal="center">
          <Badge background="brand-alpha-weak" paddingX="12" paddingY="4" onBackground="neutral-strong" textVariant="label-default-s" arrow={false}>
            <Row paddingY="2">Contact Us</Row>
          </Badge>
          
          <Heading wrap="balance" variant="display-strong-l">
            Get In Touch
          </Heading>
          
          <Text wrap="balance" onBackground="neutral-weak" variant="heading-default-xl">
            Ready to transform your business with AI? We'd love to hear from you.
          </Text>
        </Column>
      </Column>

      {/* Contact Information */}
      <Column fillWidth paddingY="32" gap="l" horizontal="center">
        <Column maxWidth="s" gap="l">
          <Heading variant="display-strong-s">
            Contact Information
          </Heading>
          
          <Flex gap="24" wrap horizontal="center">
            <Card style={{ minWidth: '300px', padding: '24px' }}>
              <Flex gap="16" vertical="start">
                <div style={{ 
                  width: '56px', 
                  height: '56px', 
                  backgroundColor: '#2563eb', 
                  borderRadius: '12px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M4 4H20C21.1 4 22 4.9 22 6V18C22 19.1 21.1 20 20 20H4C2.9 20 2 19.1 2 18V6C2 4.9 2.9 4 4 4Z" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    <polyline points="22,6 12,13 2,6" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </div>
                <div>
                  <Heading variant="heading-default-l" paddingBottom="8">Email Address</Heading>
                  <Text onBackground="neutral-weak" variant="body-default-l" paddingBottom="4">
                    zehanxtech@gmail.com
                  </Text>
                  <Text onBackground="neutral-weak" variant="body-default-s">
                    Professional inquiries and project discussions
                  </Text>
                </div>
              </Flex>
            </Card>

            <Card style={{ minWidth: '300px', padding: '24px' }}>
              <Flex gap="16" vertical="start">
                <div style={{ 
                  width: '56px', 
                  height: '56px', 
                  backgroundColor: '#2563eb', 
                  borderRadius: '12px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M22 16.92V19.92C22 20.52 21.39 21 20.92 21C9.36 21 0 11.64 0 0.08C0 -0.39 0.48 -1 1.08 -1H4.08C4.68 -1 5.08 -0.39 5.08 0.08C5.08 2.25 5.42 4.35 6.05 6.31C6.18 6.65 6.1 7.02 5.82 7.3L4.05 9.07C6.05 13.04 9.96 16.95 13.93 18.95L15.7 17.18C15.98 16.9 16.35 16.82 16.69 16.95C18.65 17.58 20.75 17.92 22.92 17.92C23.39 17.92 24 18.32 24 18.92V21.92Z" fill="white"/>
                  </svg>
                </div>
                <div>
                  <Heading variant="heading-default-l" paddingBottom="8">Phone Number</Heading>
                  <Text onBackground="neutral-weak" variant="body-default-l" paddingBottom="4">
                    +92 344 2693910
                  </Text>
                  <Text onBackground="neutral-weak" variant="body-default-s">
                    Available Mon-Fri, 9 AM - 6 PM (PKT)
                  </Text>
                </div>
              </Flex>
            </Card>

            <Card style={{ minWidth: '300px', padding: '24px' }}>
              <Flex gap="16" vertical="start">
                <div style={{ 
                  width: '56px', 
                  height: '56px', 
                  backgroundColor: '#2563eb', 
                  borderRadius: '12px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M21 10C21 17 12 23 12 23S3 17 3 10C3 5.03 7.03 1 12 1S21 5.03 21 10Z" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    <circle cx="12" cy="10" r="3" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </div>
                <div>
                  <Heading variant="heading-default-l" paddingBottom="8">Location</Heading>
                  <Text onBackground="neutral-weak" variant="body-default-l" paddingBottom="4">
                    Gujranwala, Pakistan
                  </Text>
                  <Text onBackground="neutral-weak" variant="body-default-s">
                    Serving clients globally with remote collaboration
                  </Text>
                </div>
              </Flex>
            </Card>
          </Flex>
        </Column>
      </Column>

      {/* Services Overview */}
      <Column fillWidth paddingY="32" gap="l" horizontal="center">
        <Column maxWidth="s" gap="l">
          <Heading variant="display-strong-s">
            How We Can Help
          </Heading>
          
          <Flex gap="16" wrap horizontal="center">
            <Card style={{ maxWidth: '300px' }}>
              <Heading variant="heading-default-m" paddingBottom="8">AI Systems Development</Heading>
              <Text onBackground="neutral-weak" variant="body-default-s">
                Custom AI systems, machine learning models, intelligent chatbots, and advanced automation solutions.
              </Text>
            </Card>

            <Card style={{ maxWidth: '300px' }}>
              <Heading variant="heading-default-m" paddingBottom="8">Web Development</Heading>
              <Text onBackground="neutral-weak" variant="body-default-s">
                Modern web applications, responsive design, full-stack development, and cloud deployment.
              </Text>
            </Card>

            <Card style={{ maxWidth: '300px' }}>
              <Heading variant="heading-default-m" paddingBottom="8">Consulting</Heading>
              <Text onBackground="neutral-weak" variant="body-default-s">
                Technology consulting, AI strategy, digital transformation, and technical guidance.
              </Text>
            </Card>
          </Flex>
        </Column>
      </Column>

      {/* Social Links */}
      <Column fillWidth paddingY="32" gap="l" horizontal="center">
        <Column maxWidth="s" gap="l">
          <Heading variant="display-strong-s">
            Connect With Us
          </Heading>
          
          <Flex gap="16" wrap horizontal="center">
            <Card style={{ minWidth: '200px', textAlign: 'center' }}>
              <a 
                href="https://github.com/Ahmadjamil888" 
                target="_blank" 
                rel="noopener noreferrer"
                style={{ textDecoration: 'none', color: 'inherit' }}
              >
                <Heading variant="heading-default-s" paddingBottom="4">GitHub</Heading>
                <Text onBackground="neutral-weak" variant="body-default-s">
                  View our open source projects
                </Text>
              </a>
            </Card>

            <Card style={{ minWidth: '200px', textAlign: 'center' }}>
              <a 
                href="https://www.linkedin.com/company/zehanxtech" 
                target="_blank" 
                rel="noopener noreferrer"
                style={{ textDecoration: 'none', color: 'inherit' }}
              >
                <Heading variant="heading-default-s" paddingBottom="4">LinkedIn</Heading>
                <Text onBackground="neutral-weak" variant="body-default-s">
                  Connect professionally
                </Text>
              </a>
            </Card>

            <Card style={{ minWidth: '200px', textAlign: 'center' }}>
              <a 
                href="https://www.youtube.com/@zehanxtech" 
                target="_blank" 
                rel="noopener noreferrer"
                style={{ textDecoration: 'none', color: 'inherit' }}
              >
                <Heading variant="heading-default-s" paddingBottom="4">YouTube</Heading>
                <Text onBackground="neutral-weak" variant="body-default-s">
                  Watch our tech content
                </Text>
              </a>
            </Card>
          </Flex>
        </Column>
      </Column>

      {/* Call to Action */}
      <Column fillWidth paddingY="32" gap="l" horizontal="center">
        <Column maxWidth="s" gap="m" horizontal="center">
          <Heading variant="display-strong-s">
            Ready to Get Started?
          </Heading>
          <Text onBackground="neutral-weak" variant="body-default-l">
            Drop us an email or give us a call. We're excited to discuss your project and explore how we can help bring your ideas to life with cutting-edge AI and web technologies.
          </Text>
          
          <Flex gap="16" wrap horizontal="center" paddingTop="16">
            <a 
              href="mailto:zehanxtech@gmail.com"
              style={{
                display: 'inline-block',
                padding: '12px 24px',
                backgroundColor: '#2563eb',
                color: 'white',
                textDecoration: 'none',
                borderRadius: '8px',
                fontWeight: '600'
              }}
            >
              Send Email
            </a>
            <a 
              href="tel:+923442693910"
              style={{
                display: 'inline-block',
                padding: '12px 24px',
                border: '2px solid #2563eb',
                color: '#2563eb',
                textDecoration: 'none',
                borderRadius: '8px',
                fontWeight: '600'
              }}
            >
              Call Now
            </a>
          </Flex>
        </Column>
      </Column>
    </Column>
  );
}