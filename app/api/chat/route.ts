import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const { messages } = await request.json();
    
    if (!messages || !Array.isArray(messages)) {
      return NextResponse.json(
        { error: 'Invalid messages format' },
        { status: 400 }
      );
    }

    const userMessage = messages[messages.length - 1]?.content || '';
    
    // Generate intelligent response based on user input
    const aiResponse = generateIntelligentResponse(userMessage);
    
    return NextResponse.json({
      message: aiResponse,
      timestamp: new Date().toISOString()
    });
    
  } catch (error) {
    console.error('Chat API Error:', error);
    return NextResponse.json(
      { error: 'Failed to generate response' },
      { status: 500 }
    );
  }
}

function generateIntelligentResponse(userInput: string): string {
  const input = userInput.toLowerCase();
  
  // Advanced pattern matching for more intelligent responses
  
  // Company and About queries
  if (input.match(/(zehan|company|about|who are you|what is|tell me about)/)) {
    return "I'm Zehan AI, created by Zehan X Technologies - a cutting-edge AI and web development company. We've evolved from a small web development agency into industry leaders in artificial intelligence and machine learning. Our expertise spans custom AI models, Next.js development, deep learning systems, and enterprise solutions that transform businesses worldwide.";
  }
  
  // AI and Machine Learning queries
  if (input.match(/(ai|artificial intelligence|machine learning|ml|deep learning|neural network|model)/)) {
    return "Our AI capabilities are extensive! We develop custom machine learning models, implement predictive analytics, create intelligent automation systems, and build sophisticated neural networks. Our specialties include natural language processing, computer vision, recommendation systems, and automated decision-making platforms. Each AI solution is tailored to solve specific business challenges and deliver measurable ROI.";
  }
  
  // Web Development queries
  if (input.match(/(web|website|development|next\.?js|react|typescript|frontend|backend|fullstack)/)) {
    return "We excel in modern web development using cutting-edge technologies. Our stack includes Next.js, React, TypeScript, and advanced backend systems. We build everything from responsive business websites to complex enterprise applications with seamless user experiences. Our development approach focuses on performance, scalability, security, and SEO optimization to ensure your web presence drives business growth.";
  }
  
  // Services and Capabilities
  if (input.match(/(service|help|offer|can you|what do|capabilities|solutions)/)) {
    return "We offer comprehensive technology solutions: 🤖 Custom AI & Machine Learning models, ⚡ Next.js & React development, 🧠 Deep learning systems, 🌐 Full-stack web applications, 💬 Intelligent chatbots, 📊 Data analytics & insights, 🔒 Enterprise security solutions, and ⚡ Performance optimization. Each service is customized to your specific business needs and goals.";
  }
  
  // Business and ROI focused
  if (input.match(/(business|roi|return|investment|profit|revenue|growth|transform)/)) {
    return "Our AI and web solutions are designed to drive real business value. We help companies increase efficiency by 40-60% through intelligent automation, boost revenue with predictive analytics and personalized experiences, reduce operational costs through smart process optimization, and gain competitive advantages with cutting-edge technology. Every project focuses on measurable business outcomes and ROI.";
  }
  
  // Technical Implementation
  if (input.match(/(how|implement|integrate|build|create|develop|technical)/)) {
    return "Our implementation process is thorough and collaborative. We start with detailed requirements analysis, design custom architectures using best practices, develop with agile methodologies, conduct extensive testing and optimization, and provide ongoing support and maintenance. We use modern DevOps practices, cloud-native solutions, and ensure seamless integration with your existing systems.";
  }
  
  // Pricing and Investment
  if (input.match(/(price|cost|pricing|budget|investment|quote|estimate)/)) {
    return "Our pricing is competitive and value-focused, customized based on project scope, complexity, and requirements. We offer flexible engagement models: project-based pricing for defined deliverables, retainer agreements for ongoing development, and equity partnerships for startups. Contact us for a detailed quote - we'll provide transparent pricing that aligns with your budget and delivers exceptional ROI.";
  }
  
  // Contact and Next Steps
  if (input.match(/(contact|reach|email|phone|meeting|consultation|get started|next step)/)) {
    return "Ready to transform your business with AI? Let's connect! You can reach us through our contact page, schedule a free consultation to discuss your specific needs, or email us directly. We offer complimentary project assessments where we analyze your requirements and provide strategic recommendations. Our team is excited to help you leverage AI and modern web technologies for business growth.";
  }
  
  // Greetings and Conversational
  if (input.match(/(hello|hi|hey|good morning|good afternoon|good evening)/)) {
    return "Hello! I'm Zehan AI, your intelligent assistant from Zehan X Technologies. I'm here to help you discover how our AI and web development expertise can transform your business. Whether you're interested in machine learning solutions, modern web applications, or digital transformation strategies, I'm ready to provide detailed insights. What would you like to explore?";
  }
  
  // Gratitude
  if (input.match(/(thank|thanks|appreciate|grateful)/)) {
    return "You're very welcome! I'm delighted to help you learn about our AI and web development capabilities. If you have more questions about specific technologies, implementation strategies, or how we can address your business challenges, please don't hesitate to ask. I'm here to provide detailed, helpful information!";
  }
  
  // Industry and Use Cases
  if (input.match(/(industry|use case|example|application|sector|vertical)/)) {
    return "We serve diverse industries with tailored AI solutions: 🏥 Healthcare (diagnostic AI, patient management), 💰 Finance (fraud detection, algorithmic trading), 🛒 E-commerce (recommendation engines, inventory optimization), 🏭 Manufacturing (predictive maintenance, quality control), 📚 Education (personalized learning, assessment automation), and 🚗 Automotive (autonomous systems, predictive analytics). Each solution addresses specific industry challenges and compliance requirements.";
  }
  
  // Default intelligent response with context awareness
  const topics = extractTopics(input);
  if (topics.length > 0) {
    return `I understand you're interested in ${topics.join(', ')}. At Zehan X Technologies, we have deep expertise in these areas. Our AI-powered solutions and modern web development approaches can help you achieve your goals. Would you like me to elaborate on how we can specifically address your needs in ${topics[0]}? I can provide detailed information about our methodologies, case studies, and implementation strategies.`;
  }
  
  return `That's a thoughtful question! As Zehan AI, I'm designed to help you understand how artificial intelligence and modern web development can solve real business problems. While I'd love to provide more specific insights, could you help me understand what aspect interests you most? Are you looking for information about AI implementation, web development solutions, business transformation strategies, or something else? The more specific you are, the better I can assist you!`;
}

function extractTopics(input: string): string[] {
  const topics = [];
  const topicMap = {
    'automation': ['automat', 'workflow', 'process'],
    'analytics': ['analytic', 'data', 'insight', 'report'],
    'security': ['secur', 'protect', 'safe', 'privacy'],
    'performance': ['perform', 'speed', 'fast', 'optim'],
    'scalability': ['scal', 'grow', 'expand'],
    'integration': ['integrat', 'connect', 'api'],
    'mobile': ['mobile', 'app', 'ios', 'android'],
    'cloud': ['cloud', 'aws', 'azure', 'deploy']
  };
  
  for (const [topic, keywords] of Object.entries(topicMap)) {
    if (keywords.some(keyword => input.includes(keyword))) {
      topics.push(topic);
    }
  }
  
  return topics;
}