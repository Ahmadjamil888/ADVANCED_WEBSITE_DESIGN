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

        /* Glowing Cards Styles */
        .glowing-cards-container {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
          gap: 2rem;
          width: 100%;
          max-width: 1200px;
        }

        .glowing-card {
          position: relative;
          width: 100%;
          padding: 2rem;
          border-radius: 12px;
          border: 1px rgba(255, 255, 255, 0.1);
          background: rgba(40, 40, 40, 0.7);
          box-shadow: 2px 4px 16px 0px rgba(248, 248, 248, 0.06) inset;
          overflow: hidden;
          transition: all 0.3s ease;
          backdrop-filter: blur(10px);
        }

        .glowing-card:hover {
          border-color: rgba(207, 48, 170, 0.5);
          background: rgba(64, 47, 181, 0.1);
          box-shadow: 0 0 30px rgba(207, 48, 170, 0.2), 2px 4px 16px 0px rgba(248, 248, 248, 0.06) inset;
        }

        .card-image-container {
          height: 200px;
          border-radius: 12px;
          background: linear-gradient(135deg, rgba(64, 47, 181, 0.2), rgba(207, 48, 170, 0.2));
          display: flex;
          align-items: center;
          justify-content: center;
        }

        /* Testimonial Card Styles */
        .testimonial-card {
          position: relative;
          background: linear-gradient(135deg, #30344c 0%, #2a2d3f 100%);
          padding: 1.5rem;
          border-radius: 10px;
          box-shadow: 4px 4px 20px rgba(0, 0, 0, 0.3);
          max-width: 300px;
          transition: all 200ms ease-in-out;
          border: 1px solid rgba(207, 48, 170, 0.1);
        }

        .testimonial-card:hover {
          transform: translateY(-4px);
          box-shadow: 4px 8px 30px rgba(207, 48, 170, 0.2);
          border-color: rgba(207, 48, 170, 0.3);
        }

        .testimonial-body {
          display: flex;
          flex-direction: column;
        }

        .testimonial-text {
          color: #c0c3d7;
          font-weight: 400;
          line-height: 1.6;
          margin-bottom: 1rem;
          font-size: 0.95rem;
        }

        .testimonial-username {
          color: #C6E1ED;
          font-size: 0.85rem;
          font-weight: 600;
          margin-bottom: 1rem;
        }

        .testimonial-footer {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding-top: 1rem;
          border-top: 1px solid rgba(255, 255, 255, 0.1);
        }

        .testimonial-stats {
          display: flex;
          gap: 1.5rem;
          flex: 1;
        }

        .testimonial-stat {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          color: #9fa4aa;
          font-size: 0.85rem;
          cursor: pointer;
          transition: color 200ms ease;
        }

        .testimonial-stat:hover {
          color: #cf30aa;
        }

        .testimonial-stat svg {
          width: 18px;
          height: 18px;
          stroke: #9fa4aa;
          transition: stroke 200ms ease;
        }

        .testimonial-stat:hover svg {
          stroke: #cf30aa;
        }

        .testimonial-viewers {
          display: flex;
          align-items: center;
          gap: -6px;
        }

        .testimonial-viewer-avatar {
          width: 24px;
          height: 24px;
          background: linear-gradient(135deg, #402fb5, #cf30aa);
          border-radius: 50%;
          border: 2px solid #ffffff;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 10px;
          color: #ffffff;
          font-weight: bold;
          margin-left: -8px;
          transition: all 200ms ease;
        }

        .testimonial-viewer-avatar:first-child {
          margin-left: 0;
        }

        .testimonial-viewer-avatar:hover {
          transform: scale(1.1);
          z-index: 10;
        }

        .testimonial-viewer-more {
          margin-left: 0.5rem;
          color: #9fa4aa;
          font-size: 0.85rem;
          font-weight: 600;
        }

        .testimonials-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
          gap: 2rem;
          width: 100%;
          max-width: 1200px;
        }

        @media (max-width: 768px) {
          .testimonials-grid {
            grid-template-columns: 1fr;
            gap: 1.5rem;
          }

          .testimonial-card {
            max-width: 100%;
          }
        }

        .card-content {
          margin-bottom: 1.5rem;
          position: relative;
          overflow: hidden;
          mask-image: radial-gradient(50% 50% at 50% 50%, white 0%, transparent 100%);
        }

        .card-icons {
          display: flex;
          flex-direction: row;
          justify-content: center;
          align-items: center;
          gap: 1rem;
          flex-wrap: wrap;
        }

        .icon-circle {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 48px;
          height: 48px;
          border-radius: 50%;
          background: rgba(248, 248, 248, 0.01);
          box-shadow: 0px 0px 8px 0px rgba(248, 248, 248, 0.25) inset, 0px 32px 24px -16px rgba(0, 0, 0, 0.4);
          font-size: 1.5rem;
          transition: all 0.3s ease;
        }

        .icon-circle:hover {
          transform: scale(1.1);
          box-shadow: 0px 0px 16px 0px rgba(207, 48, 170, 0.4) inset, 0px 32px 24px -16px rgba(0, 0, 0, 0.4);
        }

        .card-content h3 {
          font-size: 1.25rem;
          font-weight: 600;
          color: #ffffff;
          margin-bottom: 0.75rem;
        }

        .card-content p {
          font-size: 0.95rem;
          color: rgba(255, 255, 255, 0.7);
          line-height: 1.6;
        }

        .glow-line {
          position: absolute;
          height: 40px;
          width: 1px;
          top: 20%;
          left: 50%;
          background: linear-gradient(to bottom, transparent, #cf30aa, transparent);
          opacity: 0;
          transition: opacity 0.3s ease;
          animation: moveGlow 3s ease-in-out infinite;
        }

        .glowing-card:hover .glow-line {
          opacity: 1;
        }

        @keyframes moveGlow {
          0%, 100% { top: 0%; }
          50% { top: 80%; }
        }

        @media (max-width: 768px) {
          .glowing-cards-container {
            grid-template-columns: 1fr;
            gap: 1.5rem;
          }

          .glowing-card {
            padding: 1.5rem;
          }

          .card-image-container {
            height: 150px;
          }

          .icon-circle {
            width: 40px;
            height: 40px;
            font-size: 1.25rem;
          }

          .card-content h3 {
            font-size: 1.1rem;
          }

          .card-content p {
            font-size: 0.9rem;
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

      {/* Services Section with Glowing Cards */}
      <RevealFx delay={0.3} fillWidth horizontal="center" paddingTop="32" paddingBottom="32" className="mobile-responsive">
        <Column maxWidth="l" gap="m" fillWidth className="mobile-text-center">
          <Heading as="h2" variant="display-strong-s" className="mobile-heading-sm">Revolutionary AI Technology</Heading>
          
          <div className="glowing-cards-container">
            {/* Card 1 */}
            <div className="glowing-card">
              <div className="card-image-container">
                <div className="card-icons">
                  <div className="icon-circle">🤖</div>
                  <div className="icon-circle">⚙️</div>
                  <div className="icon-circle">🧠</div>
                </div>
              </div>
              <div className="glow-line"></div>
              <div className="card-content">
                <h3>AI Model Generation</h3>
                <p>Describe any AI model you need, and our AI will generate, train, and deploy it automatically. From sentiment analysis to computer vision.</p>
              </div>
            </div>

            {/* Card 2 */}
            <div className="glowing-card">
              <div className="card-image-container">
                <div className="card-icons">
                  <div className="icon-circle">🚀</div>
                  <div className="icon-circle">☁️</div>
                  <div className="icon-circle">⚡</div>
                </div>
              </div>
              <div className="glow-line"></div>
              <div className="card-content">
                <h3>E2B Deployment</h3>
                <p>Deploy your trained PyTorch models to E2B sandboxes with REST API endpoints. Production-ready in seconds with zero configuration.</p>
              </div>
            </div>

            {/* Card 3 */}
            <div className="glowing-card">
              <div className="card-image-container">
                <div className="card-icons">
                  <div className="icon-circle">🎯</div>
                  <div className="icon-circle">📊</div>
                  <div className="icon-circle">🔧</div>
                </div>
              </div>
              <div className="glow-line"></div>
              <div className="card-content">
                <h3>Any Use Case</h3>
                <p>Build AI models for any business need. Text classification, image recognition, time series forecasting, and much more.</p>
              </div>
            </div>
          </div>

          <Button
            href="/ai-model-generator"
            variant="primary"
            size="m"
            arrowIcon
            className="mobile-full-width"
          >
            Try AI Model Generator
          </Button>
        </Column>
      </RevealFx>

      {/* Timeline Section */}
      <RevealFx delay={0.4} fillWidth horizontal="center" paddingTop="32" paddingBottom="32" className="mobile-responsive">
        <Column maxWidth="s" gap="m" fillWidth className="mobile-text-center">
          <Heading as="h2" variant="display-strong-s" className="mobile-heading-sm">Our Journey</Heading>
          <Column gap="16" fillWidth className="mobile-small-gap">
            <Card fillWidth className="mobile-padding-sm">
              <Column gap="8">
                <Heading variant="heading-default-s" className="mobile-font-sm">2024 - Present</Heading>
                <Text onBackground="neutral-weak" className="mobile-font-sm">Founded zehanxtech with a mission to build AI for the betterment of humanity.</Text>
              </Column>
            </Card>
            <Card fillWidth className="mobile-padding-sm">
              <Column gap="8">
                <Heading variant="heading-default-s" className="mobile-font-sm">2023 - 2024</Heading>
                <Text onBackground="neutral-weak" className="mobile-font-sm">Research and development in AI technologies and modern web solutions.</Text>
              </Column>
            </Card>
            <Card fillWidth className="mobile-padding-sm">
              <Column gap="8">
                <Heading variant="heading-default-s" className="mobile-font-sm">2022 - 2023</Heading>
                <Text onBackground="neutral-weak" className="mobile-font-sm">Building expertise in machine learning, neural networks, and advanced web development.</Text>
              </Column>
            </Card>
          </Column>
        </Column>
      </RevealFx>

      {/* Testimonials Section */}
      <RevealFx delay={0.5} fillWidth horizontal="center" paddingTop="32" paddingBottom="32" className="mobile-responsive">
        <Column maxWidth="l" gap="m" fillWidth className="mobile-text-center">
          <Heading as="h2" variant="display-strong-s" className="mobile-heading-sm">Client Testimonials</Heading>
          
          <div className="testimonials-grid">
            {/* Testimonial 1 */}
            <div className="testimonial-card">
              <div className="testimonial-body">
                <p className="testimonial-text">
                  "zehanxtech delivered an exceptional AI solution that transformed our business operations. Highly professional and innovative!"
                </p>
                <span className="testimonial-username">from: Sarah Johnson, Tech Director</span>
                
                <div className="testimonial-footer">
                  <div className="testimonial-stats">
                    <div className="testimonial-stat">
                      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <path strokeWidth="1.5" d="M16 10H16.01M12 10H12.01M8 10H8.01M3 10C3 4.64706 5.11765 3 12 3C18.8824 3 21 4.64706 21 10C21 15.3529 18.8824 17 12 17C11.6592 17 11.3301 16.996 11.0124 16.9876L7 21V16.4939C4.0328 15.6692 3 13.7383 3 10Z"></path>
                      </svg>
                      24
                    </div>
                    <div className="testimonial-stat">
                      <svg fill="#000000" xmlns="http://www.w3.org/2000/svg" viewBox="-2.5 0 32 32">
                        <path fill="#9fa4aa" d="M0 10.284l0.505 0.36c0.089 0.064 0.92 0.621 2.604 0.621 0.27 0 0.55-0.015 0.836-0.044 3.752 4.346 6.411 7.472 7.060 8.299-1.227 2.735-1.42 5.808-0.537 8.686l0.256 0.834 7.63-7.631 8.309 8.309 0.742-0.742-8.309-8.309 7.631-7.631-0.834-0.255c-2.829-0.868-5.986-0.672-8.686 0.537-0.825-0.648-3.942-3.3-8.28-7.044 0.11-0.669 0.23-2.183-0.575-3.441l-0.352-0.549-8.001 8.001z"></path>
                      </svg>
                      12
                    </div>
                    <div className="testimonial-stat">
                      <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path opacity="0.1" d="M21 6C21 7.65685 19.6569 9 18 9C16.3431 9 15 7.65685 15 6C15 4.34315 16.3431 3 18 3C19.6569 3 21 4.34315 21 6Z" fill="#323232"></path>
                        <path d="M21 6C21 7.65685 19.6569 9 18 9C16.3431 9 15 7.65685 15 6C15 4.34315 16.3431 3 18 3C19.6569 3 21 4.34315 21 6Z" strokeWidth="2"></path>
                      </svg>
                      8
                    </div>
                  </div>
                  <div className="testimonial-viewers">
                    <div className="testimonial-viewer-avatar">SJ</div>
                    <div className="testimonial-viewer-avatar">MC</div>
                    <div className="testimonial-viewer-avatar">+18</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Testimonial 2 */}
            <div className="testimonial-card">
              <div className="testimonial-body">
                <p className="testimonial-text">
                  "Their expertise in both AI and web development is outstanding. They created exactly what we envisioned and delivered on time."
                </p>
                <span className="testimonial-username">from: Michael Chen, Startup Founder</span>
                
                <div className="testimonial-footer">
                  <div className="testimonial-stats">
                    <div className="testimonial-stat">
                      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <path strokeWidth="1.5" d="M16 10H16.01M12 10H12.01M8 10H8.01M3 10C3 4.64706 5.11765 3 12 3C18.8824 3 21 4.64706 21 10C21 15.3529 18.8824 17 12 17C11.6592 17 11.3301 16.996 11.0124 16.9876L7 21V16.4939C4.0328 15.6692 3 13.7383 3 10Z"></path>
                      </svg>
                      31
                    </div>
                    <div className="testimonial-stat">
                      <svg fill="#000000" xmlns="http://www.w3.org/2000/svg" viewBox="-2.5 0 32 32">
                        <path fill="#9fa4aa" d="M0 10.284l0.505 0.36c0.089 0.064 0.92 0.621 2.604 0.621 0.27 0 0.55-0.015 0.836-0.044 3.752 4.346 6.411 7.472 7.060 8.299-1.227 2.735-1.42 5.808-0.537 8.686l0.256 0.834 7.63-7.631 8.309 8.309 0.742-0.742-8.309-8.309 7.631-7.631-0.834-0.255c-2.829-0.868-5.986-0.672-8.686 0.537-0.825-0.648-3.942-3.3-8.28-7.044 0.11-0.669 0.23-2.183-0.575-3.441l-0.352-0.549-8.001 8.001z"></path>
                      </svg>
                      19
                    </div>
                    <div className="testimonial-stat">
                      <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path opacity="0.1" d="M21 6C21 7.65685 19.6569 9 18 9C16.3431 9 15 7.65685 15 6C15 4.34315 16.3431 3 18 3C19.6569 3 21 4.34315 21 6Z" fill="#323232"></path>
                        <path d="M21 6C21 7.65685 19.6569 9 18 9C16.3431 9 15 7.65685 15 6C15 4.34315 16.3431 3 18 3C19.6569 3 21 4.34315 21 6Z" strokeWidth="2"></path>
                      </svg>
                      15
                    </div>
                  </div>
                  <div className="testimonial-viewers">
                    <div className="testimonial-viewer-avatar">MC</div>
                    <div className="testimonial-viewer-avatar">AK</div>
                    <div className="testimonial-viewer-avatar">+22</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Testimonial 3 */}
            <div className="testimonial-card">
              <div className="testimonial-body">
                <p className="testimonial-text">
                  "The AI model generation system is revolutionary. We saved months of development time and got production-ready models instantly!"
                </p>
                <span className="testimonial-username">from: Alex Kumar, CTO</span>
                
                <div className="testimonial-footer">
                  <div className="testimonial-stats">
                    <div className="testimonial-stat">
                      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <path strokeWidth="1.5" d="M16 10H16.01M12 10H12.01M8 10H8.01M3 10C3 4.64706 5.11765 3 12 3C18.8824 3 21 4.64706 21 10C21 15.3529 18.8824 17 12 17C11.6592 17 11.3301 16.996 11.0124 16.9876L7 21V16.4939C4.0328 15.6692 3 13.7383 3 10Z"></path>
                      </svg>
                      42
                    </div>
                    <div className="testimonial-stat">
                      <svg fill="#000000" xmlns="http://www.w3.org/2000/svg" viewBox="-2.5 0 32 32">
                        <path fill="#9fa4aa" d="M0 10.284l0.505 0.36c0.089 0.064 0.92 0.621 2.604 0.621 0.27 0 0.55-0.015 0.836-0.044 3.752 4.346 6.411 7.472 7.060 8.299-1.227 2.735-1.42 5.808-0.537 8.686l0.256 0.834 7.63-7.631 8.309 8.309 0.742-0.742-8.309-8.309 7.631-7.631-0.834-0.255c-2.829-0.868-5.986-0.672-8.686 0.537-0.825-0.648-3.942-3.3-8.28-7.044 0.11-0.669 0.23-2.183-0.575-3.441l-0.352-0.549-8.001 8.001z"></path>
                      </svg>
                      28
                    </div>
                    <div className="testimonial-stat">
                      <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path opacity="0.1" d="M21 6C21 7.65685 19.6569 9 18 9C16.3431 9 15 7.65685 15 6C15 4.34315 16.3431 3 18 3C19.6569 3 21 4.34315 21 6Z" fill="#323232"></path>
                        <path d="M21 6C21 7.65685 19.6569 9 18 9C16.3431 9 15 7.65685 15 6C15 4.34315 16.3431 3 18 3C19.6569 3 21 4.34315 21 6Z" strokeWidth="2"></path>
                      </svg>
                      21
                    </div>
                  </div>
                  <div className="testimonial-viewers">
                    <div className="testimonial-viewer-avatar">AK</div>
                    <div className="testimonial-viewer-avatar">RJ</div>
                    <div className="testimonial-viewer-avatar">+25</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </Column>
      </RevealFx>

      {/* Contact Section */}
      <RevealFx delay={0.6} fillWidth horizontal="center" paddingTop="32" paddingBottom="32" className="mobile-responsive">
        <Column maxWidth="s" gap="m" fillWidth className="mobile-text-center">
          <Heading as="h2" variant="display-strong-s" className="mobile-heading-sm">Get In Touch</Heading>
          <Text onBackground="neutral-weak" className="mobile-font-sm">
            Ready to transform your business with AI? Contact us today.
          </Text>
          <Column gap="16" fillWidth className="mobile-small-gap">
            <Card fillWidth className="mobile-padding-sm">
              <Column gap="8">
                <Heading variant="heading-default-s" className="mobile-font-sm">Email</Heading>
                <Text onBackground="neutral-weak" className="mobile-font-sm">zehanxtech@gmail.com</Text>
              </Column>
            </Card>
            <Card fillWidth className="mobile-padding-sm">
              <Column gap="8">
                <Heading variant="heading-default-s" className="mobile-font-sm">Phone</Heading>
                <Text onBackground="neutral-weak" className="mobile-font-sm">+92 344 2693910</Text>
              </Column>
            </Card>
            <Card fillWidth className="mobile-padding-sm">
              <Column gap="8">
                <Heading variant="heading-default-s" className="mobile-font-sm">Location</Heading>
                <Text onBackground="neutral-weak" className="mobile-font-sm">Gujranwala, Pakistan</Text>
              </Column>
            </Card>
          </Column>
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