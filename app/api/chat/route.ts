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
      model: 'zehan-ai-intelligent-system',
      timestamp: new Date().toISOString()
    });
    
  } catch (err) {
    console.error('Chat API Error:', err);
    return NextResponse.json(
      { error: 'Failed to generate response' },
      { status: 500 }
    );
  }
}

function generateIntelligentResponse(userInput: string): string {
  const input = userInput.toLowerCase();
  
  // Advanced pattern matching for more intelligent responses
  
  // Mathematical calculations
  if (input.match(/^\s*\d+\s*[-+*/]\s*\d+\s*$/) || input.match(/what\s+is\s+\d+\s*[-+*/]\s*\d+/)) {
    try {
      const mathExpression = input.replace(/what\s+is\s+/, '').trim();
      const result = evaluateSimpleMath(mathExpression);
      return `The answer is ${result}. As Zehan AI, I can help with calculations and much more! I'm also here to discuss our AI and web development services. What else would you like to know?`;
    } catch {
      return "I can help with basic math calculations! Try asking me something like '5 + 3' or 'what is 10 - 4'. I'm also here to discuss our AI and web development expertise.";
    }
  }
  
  // Technical AI/ML concepts
  if (input.match(/supervised\s+learning/)) {
    return "Supervised learning is a fundamental machine learning approach where algorithms learn from labeled training data to make predictions on new, unseen data. The model learns the relationship between input features and known outputs. At Zehan X Technologies, we use supervised learning for classification (predicting categories) and regression (predicting continuous values) tasks like fraud detection, price prediction, and customer segmentation.";
  }
  
  if (input.match(/unsupervised\s+learning/)) {
    return "Unsupervised learning discovers hidden patterns in data without labeled examples. It includes clustering (grouping similar data), dimensionality reduction, and anomaly detection. We use unsupervised learning for customer segmentation, recommendation systems, and identifying unusual patterns in business data.";
  }
  
  if (input.match(/neural\s+network/)) {
    return "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information through weighted connections. We build custom neural networks for image recognition, natural language processing, and complex pattern recognition tasks.";
  }
  
  if (input.match(/deep\s+learning/)) {
    return "Deep learning uses multi-layered neural networks to automatically learn complex patterns from data. It excels at tasks like image recognition, speech processing, and natural language understanding. Our deep learning solutions include computer vision systems, chatbots, and predictive analytics platforms.";
  }
  
  if (input.match(/machine\s+learning|what.*ml/)) {
    return "Machine Learning is a subset of AI that enables computers to learn and improve from data without being explicitly programmed. It includes supervised learning (with labeled data), unsupervised learning (finding patterns), and reinforcement learning (learning through rewards). We implement ML for predictive analytics, automation, and intelligent decision-making systems.";
  }
  
  if (input.match(/artificial\s+intelligence|what.*ai/)) {
    return "Artificial Intelligence is the simulation of human intelligence in machines. It encompasses machine learning, natural language processing, computer vision, and robotics. At Zehan X Technologies, we develop AI systems that can understand, learn, and make decisions to solve complex business problems and automate processes.";
  }
  
  if (input.match(/nlp|natural\s+language/)) {
    return "Natural Language Processing (NLP) enables computers to understand, interpret, and generate human language. It powers chatbots, sentiment analysis, language translation, and text summarization. We build NLP solutions for customer service automation, content analysis, and intelligent document processing.";
  }
  
  if (input.match(/computer\s+vision/)) {
    return "Computer Vision enables machines to interpret and understand visual information from images and videos. It's used for object detection, facial recognition, medical imaging, and quality control. Our computer vision solutions help businesses automate visual inspection, enhance security, and extract insights from visual data.";
  }
  
  // Company and About queries (more specific)
  if (input.match(/(who are you|what are you|introduce yourself)/) && !input.includes('zehan') && !input.includes('company')) {
    return "I'm Zehan AI, an intelligent assistant created by Zehan X Technologies. I can help with technical questions, mathematical calculations, and provide information about AI, machine learning, and web development. I'm also here to discuss how our technology solutions can benefit your business. What would you like to explore?";
  }
  
  if (input.match(/(zehan|company|about.*zehan|tell me about.*company)/)) {
    return "Zehan X Technologies is a cutting-edge AI and web development company. We've evolved from a small web development agency into industry leaders in artificial intelligence and machine learning. Our expertise spans custom AI models, Next.js development, deep learning systems, and enterprise solutions that transform businesses worldwide.";
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

function evaluateSimpleMath(expression: string): number {
  // Simple and safe math evaluation for basic operations
  const cleanExpression = expression.replace(/\s+/g, '');
  
  // Only allow numbers and basic operators
  if (!/^[\d+\-*/().]+$/.test(cleanExpression)) {
    throw new Error('Invalid expression');
  }
  
  // Basic operations
  if (cleanExpression.includes('+')) {
    const parts = cleanExpression.split('+');
    return parseFloat(parts[0]) + parseFloat(parts[1]);
  }
  if (cleanExpression.includes('-')) {
    const parts = cleanExpression.split('-');
    return parseFloat(parts[0]) - parseFloat(parts[1]);
  }
  if (cleanExpression.includes('*')) {
    const parts = cleanExpression.split('*');
    return parseFloat(parts[0]) * parseFloat(parts[1]);
  }
  if (cleanExpression.includes('/')) {
    const parts = cleanExpression.split('/');
    const divisor = parseFloat(parts[1]);
    if (divisor === 0) throw new Error('Division by zero');
    return parseFloat(parts[0]) / divisor;
  }
  
  throw new Error('Unsupported operation');
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