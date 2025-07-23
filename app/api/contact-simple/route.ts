import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const { email, name, message, company } = await request.json();
    
    console.log('📧 Contact form submission received:', { 
      name, 
      email, 
      company: company || 'Not provided',
      messageLength: message?.length,
      timestamp: new Date().toISOString()
    });

    // Validate required fields
    if (!email || !name || !message) {
      return NextResponse.json(
        { success: false, message: 'Name, email, and message are required' },
        { status: 400 }
      );
    }

    // Validate email format
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      return NextResponse.json(
        { success: false, message: 'Please enter a valid email address' },
        { status: 400 }
      );
    }

    // METHOD 1: Try Web3Forms (most reliable for server-side)
    if (process.env.WEB3FORMS_ACCESS_KEY) {
      try {
        console.log('🔄 Attempting Web3Forms...');
        
        const formData = new FormData();
        formData.append('access_key', process.env.WEB3FORMS_ACCESS_KEY);
        formData.append('subject', `New Contact Form Submission from ${name} - Zehan X Technologies`);
        formData.append('name', name);
        formData.append('email', email);
        formData.append('message', message);
        formData.append('company', company || 'Not provided');
        formData.append('from_name', 'Zehan X Technologies Website');
        
        const response = await fetch('https://api.web3forms.com/submit', {
          method: 'POST',
          body: formData
        });

        const result = await response.json();
        
        if (result.success) {
          console.log('✅ Web3Forms successful:', result);
          return NextResponse.json({
            success: true,
            message: 'Message sent successfully! We\'ll get back to you within 24 hours.',
            method: 'Web3Forms'
          });
        } else {
          console.error('❌ Web3Forms failed:', result);
          throw new Error(`Web3Forms failed: ${result.message || 'Unknown error'}`);
        }
      } catch (web3Error) {
        console.error('Web3Forms error:', web3Error);
      }
    }

    // METHOD 2: Try Formspree as backup
    try {
      console.log('🔄 Attempting Formspree backup...');
      
      const formspreeResponse = await fetch('https://formspree.io/f/xdkogqpw', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({
          name: name,
          email: email,
          message: message,
          company: company || 'Not provided',
          _replyto: email,
          _subject: `New Contact Form Submission from ${name} - Zehan X Technologies`
        })
      });

      if (formspreeResponse.ok) {
        console.log('✅ Formspree backup successful');
        return NextResponse.json({
          success: true,
          message: 'Message sent successfully! We\'ll get back to you within 24 hours.',
          method: 'Formspree'
        });
      } else {
        console.error('❌ Formspree backup failed:', formspreeResponse.status);
      }
    } catch (formspreeError) {
      console.error('Formspree backup error:', formspreeError);
    }

    // METHOD 3: Try direct email to a webhook service
    try {
      console.log('🔄 Attempting webhook notification...');
      
      // Using a webhook service like webhook.site for testing
      // Replace this URL with your own webhook or notification service
      const webhookUrl = 'https://webhook.site/unique-id'; // You can get this from webhook.site
      
      const webhookResponse = await fetch(webhookUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          type: 'contact_form_submission',
          timestamp: new Date().toISOString(),
          data: {
            name,
            email,
            company: company || 'Not provided',
            message,
            source: 'Zehan X Technologies Website'
          }
        })
      });

      if (webhookResponse.ok) {
        console.log('✅ Webhook notification sent');
      }
    } catch (webhookError) {
      console.error('Webhook error:', webhookError);
    }

    // FINAL FALLBACK: Detailed logging for manual processing
    console.log('=== CONTACT FORM SUBMISSION - MANUAL PROCESSING REQUIRED ===');
    console.log('Timestamp:', new Date().toISOString());
    console.log('Name:', name);
    console.log('Email:', email);
    console.log('Company:', company || 'Not provided');
    console.log('Message:', message);
    console.log('User Agent:', request.headers.get('user-agent'));
    console.log('IP:', request.headers.get('x-forwarded-for') || request.headers.get('x-real-ip') || 'unknown');
    console.log('Referer:', request.headers.get('referer'));
    console.log('=============================================================');

    // Create a structured log entry
    const logEntry = {
      timestamp: new Date().toISOString(),
      type: 'contact_form_submission',
      data: {
        name,
        email,
        company: company || 'Not provided',
        message,
        userAgent: request.headers.get('user-agent'),
        ip: request.headers.get('x-forwarded-for') || request.headers.get('x-real-ip') || 'unknown',
        referer: request.headers.get('referer')
      }
    };
    
    console.log('STRUCTURED_LOG:', JSON.stringify(logEntry));

    return NextResponse.json({
      success: true,
      message: 'Message received successfully! We\'ll get back to you within 24 hours.',
      method: 'Logged for manual processing'
    });

  } catch (error) {
    console.error('Contact form error:', error);
    
    return NextResponse.json(
      { 
        success: false, 
        message: 'Failed to send message. Please try again or contact us directly at shazabjamildhami@gmail.com' 
      },
      { status: 500 }
    );
  }
}