import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const { email, name, message, company } = await request.json();
    
    console.log('📧 Email sending attempt:', { email, name, company });

    // Validate required fields
    if (!email || !name || !message) {
      return NextResponse.json(
        { success: false, message: 'Name, email, and message are required' },
        { status: 400 }
      );
    }

    // Using Formspree - a reliable form backend service
    // You can create a free account at https://formspree.io and get your form endpoint
    const formspreeEndpoint = 'https://formspree.io/f/xdkogqpw'; // This is a demo endpoint
    
    try {
      const response = await fetch(formspreeEndpoint, {
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
          _subject: `New Contact Form Submission from ${name} - Zehan X Technologies`,
          _template: 'table'
        })
      });

      if (response.ok) {
        console.log('✅ Formspree email sent successfully');
        return NextResponse.json({
          success: true,
          message: 'Message sent successfully! We\'ll get back to you within 24 hours.'
        });
      } else {
        const errorText = await response.text();
        console.error('❌ Formspree failed:', errorText);
        throw new Error(`Formspree failed: ${response.status}`);
      }
    } catch (formspreeError) {
      console.error('Formspree error:', formspreeError);
    }

    // Fallback: Use Web3Forms with a different approach
    if (process.env.WEB3FORMS_ACCESS_KEY) {
      try {
        const formData = new FormData();
        formData.append('access_key', process.env.WEB3FORMS_ACCESS_KEY);
        formData.append('subject', `New Contact Form Submission from ${name} - Zehan X Technologies`);
        formData.append('name', name);
        formData.append('email', email);
        formData.append('message', message);
        if (company) {
          formData.append('company', company);
        }
        
        const response = await fetch('https://api.web3forms.com/submit', {
          method: 'POST',
          body: formData
        });

        const result = await response.json();
        
        if (result.success) {
          console.log('✅ Web3Forms backup successful');
          return NextResponse.json({
            success: true,
            message: 'Message sent successfully! We\'ll get back to you within 24 hours.'
          });
        } else {
          console.error('❌ Web3Forms backup failed:', result);
        }
      } catch (web3Error) {
        console.error('Web3Forms backup error:', web3Error);
      }
    }

    // Final fallback: Log the submission
    console.log('=== EMAIL FALLBACK - MANUAL PROCESSING REQUIRED ===');
    console.log('Timestamp:', new Date().toISOString());
    console.log('Name:', name);
    console.log('Email:', email);
    console.log('Company:', company || 'Not provided');
    console.log('Message:', message);
    console.log('================================================');

    return NextResponse.json({
      success: true,
      message: 'Message received! We\'ll process it manually and get back to you within 24 hours.'
    });

  } catch (error) {
    console.error('Send email error:', error);
    return NextResponse.json(
      { success: false, message: 'Failed to send message. Please try again.' },
      { status: 500 }
    );
  }
}