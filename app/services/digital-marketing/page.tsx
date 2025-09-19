import { Badge } from "@/components/ui/badge";
import { Section } from "@/components/ui/section";
import { 
  BarChart3, 
  Target, 
  TrendingUp, 
  Mail, 
  Search, 
  Share2, 
  PenTool, 
  Globe,
  Zap,
  CheckCircle,
  ArrowRight
} from "lucide-react";
import Link from "next/link";
import Navbar from "@/components/sections/navbar/default";
import Footer from "@/components/sections/footer/default";

export const metadata = {
  title: 'Digital Marketing Services | Zehan X Technologies',
  description: 'Comprehensive digital marketing solutions including SEO, social media, content marketing, PPC, and analytics. Drive growth and engagement for your business.',
};

const services = [
  {
    title: "Search Engine Optimization (SEO)",
    description: "Boost your website's visibility in search results with our proven SEO strategies",
    icon: <Search className="size-8" />,
    features: ["Keyword Research", "On-Page Optimization", "Technical SEO", "Link Building", "Local SEO"]
  },
  {
    title: "Social Media Marketing",
    description: "Build your brand presence across all major social media platforms",
    icon: <Share2 className="size-8" />,
    features: ["Content Strategy", "Community Management", "Paid Advertising", "Influencer Partnerships", "Analytics & Reporting"]
  },
  {
    title: "Pay-Per-Click (PPC) Advertising",
    description: "Drive immediate traffic and conversions with targeted ad campaigns",
    icon: <Target className="size-8" />,
    features: ["Google Ads", "Facebook Ads", "LinkedIn Ads", "Campaign Optimization", "ROI Tracking"]
  },
  {
    title: "Content Marketing",
    description: "Create compelling content that engages your audience and drives conversions",
    icon: <PenTool className="size-8" />,
    features: ["Blog Writing", "Video Content", "Infographics", "Email Marketing", "Content Strategy"]
  },
  {
    title: "Email Marketing",
    description: "Nurture leads and retain customers with personalized email campaigns",
    icon: <Mail className="size-8" />,
    features: ["Email Automation", "Newsletter Design", "Segmentation", "A/B Testing", "Performance Analytics"]
  },
  {
    title: "Analytics & Reporting",
    description: "Track performance and optimize your marketing efforts with data-driven insights",
    icon: <BarChart3 className="size-8" />,
    features: ["Google Analytics", "Custom Dashboards", "Conversion Tracking", "ROI Analysis", "Monthly Reports"]
  }
];

const strategies = [
  {
    step: "1",
    title: "Strategy Development",
    description: "We analyze your business, competitors, and target audience to create a customized marketing strategy",
    icon: <Target className="size-6" />
  },
  {
    step: "2",
    title: "Campaign Implementation",
    description: "We execute multi-channel campaigns across SEO, social media, PPC, and content marketing",
    icon: <Zap className="size-6" />
  },
  {
    step: "3",
    title: "Performance Monitoring",
    description: "We continuously monitor and optimize campaigns for maximum ROI and engagement",
    icon: <TrendingUp className="size-6" />
  },
  {
    step: "4",
    title: "Growth & Scaling",
    description: "We scale successful campaigns and expand into new channels to accelerate growth",
    icon: <Globe className="size-6" />
  }
];

const benefits = [
  {
    title: "Increased Brand Visibility",
    description: "Boost your online presence and reach more potential customers",
    icon: <Globe className="size-8" />
  },
  {
    title: "Higher Conversion Rates",
    description: "Turn more visitors into customers with optimized marketing funnels",
    icon: <TrendingUp className="size-8" />
  },
  {
    title: "Better ROI",
    description: "Maximize your marketing budget with data-driven strategies and optimization",
    icon: <BarChart3 className="size-8" />
  },
  {
    title: "Competitive Advantage",
    description: "Stay ahead of competitors with cutting-edge digital marketing tactics",
    icon: <Target className="size-8" />
  }
];

export default function DigitalMarketing() {
  return (
    <main className="min-h-screen w-full overflow-hidden bg-black text-white">
      <Navbar />
      
      {/* Hero Section */}
      <Section className="pt-24 pb-16">
        <div className="max-w-container mx-auto text-center">
          <Badge variant="outline" className="mb-6 border-blue-500/30 bg-blue-500/10">
            <BarChart3 className="mr-2 size-4 text-blue-400" />
            <span className="text-blue-400 font-medium">Digital Marketing</span>
          </Badge>
          <h1 className="text-4xl font-bold mb-6 sm:text-6xl animate-fade-in-up">
            Drive Growth with <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-400">Strategic Digital Marketing</span>
          </h1>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto animate-fade-in-up" style={{animationDelay: '0.2s'}}>
            Transform your business with comprehensive digital marketing solutions that increase visibility, 
            drive traffic, and convert leads into loyal customers.
          </p>
        </div>
      </Section>

      {/* Services Grid */}
      <Section className="py-20">
        <div className="max-w-container mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold mb-4 text-white animate-fade-in-up">
              Our <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-400">Digital Marketing Services</span>
            </h2>
            <p className="text-gray-300 text-lg max-w-2xl mx-auto animate-fade-in-up" style={{animationDelay: '0.2s'}}>
              Comprehensive digital marketing solutions tailored to your business goals and target audience
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {services.map((service, index) => (
              <div 
                key={index} 
                className="group bg-black rounded-2xl p-8 border border-blue-500/30 hover:border-blue-500 transition-all duration-300 hover:transform hover:scale-105 animate-fade-in-up"
                style={{animationDelay: `${index * 0.1}s`}}
              >
                <div className="w-16 h-16 bg-blue-500/20 rounded-xl flex items-center justify-center mb-6 group-hover:bg-blue-500/30 transition-colors">
                  {service.icon}
                </div>
                <h3 className="text-xl font-semibold mb-4 text-white">{service.title}</h3>
                <p className="text-gray-300 mb-6 text-sm">{service.description}</p>
                <ul className="space-y-2">
                  {service.features.map((feature, i) => (
                    <li key={i} className="flex items-center text-sm text-gray-300">
                      <CheckCircle className="size-4 text-blue-400 mr-2 flex-shrink-0" />
                      {feature}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      </Section>

      {/* Process Section */}
      <Section className="py-20 bg-gradient-to-br from-gray-900 via-black to-gray-900">
        <div className="max-w-container mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold mb-4 text-white animate-fade-in-up">
              Our <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-400">4-Step Process</span>
            </h2>
            <p className="text-gray-300 text-lg max-w-2xl mx-auto animate-fade-in-up" style={{animationDelay: '0.2s'}}>
              We follow a proven methodology to deliver exceptional digital marketing results
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {strategies.map((strategy, index) => (
              <div key={index} className="text-center animate-fade-in-up" style={{animationDelay: `${index * 0.2}s`}}>
                <div className="w-16 h-16 bg-blue-500/20 rounded-full flex items-center justify-center mx-auto mb-4 text-blue-400">
                  {strategy.icon}
                </div>
                <div className="w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center mx-auto mb-4 text-sm font-bold">
                  {strategy.step}
                </div>
                <h3 className="text-lg font-semibold mb-2 text-white">{strategy.title}</h3>
                <p className="text-gray-300 text-sm">{strategy.description}</p>
              </div>
            ))}
          </div>
        </div>
      </Section>

      {/* Benefits Section */}
      <Section className="py-20">
        <div className="max-w-container mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold mb-4 text-white animate-fade-in-up">
              Why Choose Our <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-400">Digital Marketing?</span>
            </h2>
            <p className="text-gray-300 text-lg max-w-2xl mx-auto animate-fade-in-up" style={{animationDelay: '0.2s'}}>
              We deliver measurable results that drive business growth and success
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {benefits.map((benefit, index) => (
              <div 
                key={index} 
                className="text-center p-6 bg-black rounded-2xl border border-blue-500/30 hover:border-blue-500 transition-all duration-300 animate-fade-in-up"
                style={{animationDelay: `${index * 0.1}s`}}
              >
                <div className="w-16 h-16 bg-blue-500/20 rounded-full flex items-center justify-center mx-auto mb-4 text-blue-400">
                  {benefit.icon}
                </div>
                <h3 className="text-xl font-semibold mb-4 text-white">{benefit.title}</h3>
                <p className="text-gray-300 text-sm">{benefit.description}</p>
              </div>
            ))}
          </div>
        </div>
      </Section>

      {/* CTA Section */}
      <Section className="py-20 bg-gradient-to-br from-gray-900 via-black to-gray-900">
        <div className="max-w-container mx-auto text-center">
          <h2 className="text-3xl font-bold mb-6 text-white animate-fade-in-up">
            Ready to <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-400">Transform Your Digital Presence?</span>
          </h2>
          <p className="text-gray-300 mb-8 max-w-2xl mx-auto animate-fade-in-up" style={{animationDelay: '0.2s'}}>
            Let's discuss your digital marketing goals and create a strategy that drives real results for your business.
          </p>
          <div className="flex justify-center gap-4 animate-fade-in-up" style={{animationDelay: '0.4s'}}>
            <Link 
              href="/contact" 
              className="inline-flex items-center justify-center rounded-md bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 text-sm font-medium shadow-lg hover:shadow-xl transition-all duration-300 transform hover:scale-105"
            >
              <Mail className="w-5 h-5 mr-2" />
              Get Custom Quote
            </Link>
            <Link 
              href="/contact" 
              className="inline-flex items-center justify-center rounded-md border border-blue-500/50 hover:border-blue-500 bg-transparent text-blue-400 hover:text-blue-300 px-8 py-3 text-sm font-medium transition-all duration-300 hover:scale-105"
            >
              Contact Us
              <ArrowRight className="w-4 h-4 ml-2" />
            </Link>
          </div>
        </div>
      </Section>
      
      <Footer />
    </main>
  );
}
