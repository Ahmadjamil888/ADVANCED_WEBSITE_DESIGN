"use client";
import React from "react";
import { Heading, Flex, Text, Column, Badge, Row, Card, Icon } from "@/once-ui/components";
import { baseURL } from "@/app/resources";
import { about, person } from "@/app/resources/content";
import { Schema } from "@/once-ui/modules";

export default function Contact() {
  return (
    <Column maxWidth="m" gap="xl" horizontal="center">
      <Schema
        as="webPage"
        baseURL={baseURL}
        path="/contact"
        title="Contact - zehanxtech"
        description="Get in touch with zehanxtech for AI development and web solutions"
        image={`${baseURL}/og?title=${encodeURIComponent("Contact zehanxtech")}`}
        author={{
          name: "zehanxtech",
          url: `${baseURL}${about.path}`,
          image: `${baseURL}${person.avatar}`,
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
            <Card style={{ minWidth: '280px', textAlign: 'center' }}>
              <Flex vertical="center" horizontal="center" gap="12" paddingBottom="16">
                <div style={{ 
                  width: '48px', 
                  height: '48px', 
                  backgroundColor: '#2563eb', 
                  borderRadius: '12px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  <span style={{ fontSize: '24px' }}>📧</span>
                </div>
              </Flex>
              <Heading variant="heading-default-m" paddingBottom="8">Email</Heading>
              <Text onBackground="neutral-weak" variant="body-default-m">
                zehanxtech@gmail.com
              </Text>
              <Text onBackground="neutral-weak" variant="body-default-s" paddingTop="4">
                We respond within 24 hours
              </Text>
            </Card>

            <Card style={{ minWidth: '280px', textAlign: 'center' }}>
              <Flex vertical="center" horizontal="center" gap="12" paddingBottom="16">
                <div style={{ 
                  width: '48px', 
                  height: '48px', 
                  backgroundColor: '#2563eb', 
                  borderRadius: '12px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  <span style={{ fontSize: '24px' }}>📱</span>
                </div>
              </Flex>
              <Heading variant="heading-default-m" paddingBottom="8">Phone</Heading>
              <Text onBackground="neutral-weak" variant="body-default-m">
                +92 344 2693910
              </Text>
              <Text onBackground="neutral-weak" variant="body-default-s" paddingTop="4">
                Available Mon-Fri, 9 AM - 6 PM
              </Text>
            </Card>

            <Card style={{ minWidth: '280px', textAlign: 'center' }}>
              <Flex vertical="center" horizontal="center" gap="12" paddingBottom="16">
                <div style={{ 
                  width: '48px', 
                  height: '48px', 
                  backgroundColor: '#2563eb', 
                  borderRadius: '12px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  <span style={{ fontSize: '24px' }}>🌍</span>
                </div>
              </Flex>
              <Heading variant="heading-default-m" paddingBottom="8">Location</Heading>
              <Text onBackground="neutral-weak" variant="body-default-m">
                Karachi, Pakistan
              </Text>
              <Text onBackground="neutral-weak" variant="body-default-s" paddingTop="4">
                Serving clients globally
              </Text>
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
              <Heading variant="heading-default-m" paddingBottom="8">AI Development</Heading>
              <Text onBackground="neutral-weak" variant="body-default-s">
                Custom AI solutions, machine learning models, chatbots, and intelligent automation systems.
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
                fontWeight: '600',
                transition: 'background-color 0.3s'
              }}
              onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#1d4ed8'}
              onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#2563eb'}
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
                fontWeight: '600',
                transition: 'all 0.3s'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = '#2563eb';
                e.currentTarget.style.color = 'white';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = 'transparent';
                e.currentTarget.style.color = '#2563eb';
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