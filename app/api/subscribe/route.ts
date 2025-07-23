import { NextRequest, NextResponse } from 'next/server'

export async function POST(request: NextRequest) {
  try {
    const { email, name, message, company } = await request.json()
    
    console.log('📧 Contact form submission received:', { email, name, company, messageLength: message?.length })
    console.log('📧 Environment variables check:', { 
      hasWeb3FormsKey: !!process.env.WEB3FORMS_ACCESS_KEY,
      keyFirstChars: process.env.WEB3FORMS_ACCESS_KEY ? process.env.WEB3FORMS_ACCESS_KEY.substring(0, 8) + '...' : 'missing'
    })

    // Validate required fields
    if (!email || !name || !message) {
      return NextResponse.json(
        { success: false, message: 'Name, email, and message are required' },
        { status: 400 }
      )
    }

    // Validate email format
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
    if (!emailRegex.test(email)) {
      return NextResponse.json(
        { success: false, message: 'Please enter a valid email address' },
        { status: 400 }
      )
    }

    // OPTION 1: Using Formspree (Free tier: 50 submissions/month)
    // Go to https://formspree.io, create account, get your form endpoint
    if (process.env.FORMSPREE_ENDPOINT) {
      try {
        const response = await fetch(process.env.FORMSPREE_ENDPOINT, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            email: email,
            name: name,
            company: company || 'Not provided',
            message: message,
            _replyto: email,
            _subject: `New Contact Form Submission from ${name} - Zehan X Technologies`,
          }),
        })

        if (response.ok) {
          return NextResponse.json({ 
            success: true, 
            message: 'Message sent successfully! We\'ll get back to you within 24 hours.' 
          })
        } else {
          throw new Error('Formspree failed')
        }
      } catch (formspreeError) {
        console.error('Formspree error:', formspreeError)
        // Fall through to logging
      }
    }

    // OPTION 2: Using Web3Forms (Free tier: 250 submissions/month)
    // Using your Web3Forms access key
    if (process.env.WEB3FORMS_ACCESS_KEY) {
      try {
        // Using JSON format for more reliable processing with Web3Forms
        // Web3Forms accepts both FormData and JSON, but JSON is more reliable for server-side API calls
        const web3formsData = {
          access_key: process.env.WEB3FORMS_ACCESS_KEY,
          subject: `New Contact Form Submission from ${name} - Zehan X Technologies`,
          from_name: 'Zehan X Technologies Website',
          name: name,
          email: email,
          message: message,
          company: company || 'Not provided',
          reply_to: email,
          to_email: 'shazabjamildhami@gmail.com'
        };
        
        console.log('📧 Sending to Web3Forms with access key:', process.env.WEB3FORMS_ACCESS_KEY?.substring(0, 8) + '...');
        console.log('📧 Sending to recipient email:', 'shazabjamildhami@gmail.com');
        
        const response = await fetch('https://api.web3forms.com/submit', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          },
          body: JSON.stringify(web3formsData)
        })

        const responseText = await response.text();
        let result;
        
        try {
          // Try to parse as JSON
          result = JSON.parse(responseText);
        } catch (e) {
          console.error('Failed to parse Web3Forms response as JSON:', responseText);
          throw new Error(`Invalid response from Web3Forms: ${responseText.substring(0, 100)}...`);
        }
        
        if (result.success) {
          console.log('✅ Web3Forms submission successful:', {
            name,
            email,
            company: company || 'Not provided',
            timestamp: new Date().toISOString(),
            responseData: result
          });
          
          return NextResponse.json({ 
            success: true, 
            message: 'Message sent successfully! We\'ll get back to you within 24 hours.' 
          });
        } else {
          console.error('❌ Web3Forms submission failed:', {
            error: result.message || 'Unknown error',
            statusCode: response.status,
            responseData: result
          });
          throw new Error(`Web3Forms failed: ${result.message || 'Unknown error'} (Status: ${response.status})`);
        }
      } catch (web3formsError) {
        console.error('Web3Forms error:', web3formsError)
        // Fall through to logging
      }
    }

    // FALLBACK: Simple logging to console (always works)
    // This logs the message and you can check server logs or Vercel function logs
    console.log('=== NEW CONTACT FORM SUBMISSION ===')
    console.log('Timestamp:', new Date().toISOString())
    console.log('Name:', name)
    console.log('Email:', email)
    console.log('Company:', company || 'Not provided')
    console.log('Message:', message)
    console.log('User Agent:', request.headers.get('user-agent'))
    console.log('IP:', request.headers.get('x-forwarded-for') || request.headers.get('x-real-ip') || 'unknown')
    console.log('=====================================')

    // Store in a simple JSON format for easy parsing
    const submissionData = {
      timestamp: new Date().toISOString(),
      name,
      email,
      company: company || 'Not provided',
      message,
      userAgent: request.headers.get('user-agent'),
      ip: request.headers.get('x-forwarded-for') || request.headers.get('x-real-ip') || 'unknown'
    }
    
    console.log('JSON_SUBMISSION:', JSON.stringify(submissionData))
    
    return NextResponse.json({ 
      success: true, 
      message: 'Message received successfully! We\'ll get back to you within 24 hours. Your message has been logged and we will contact you soon.' 
    })

  } catch (error) {
    console.error('Contact form error:', error)
    
    return NextResponse.json(
      { success: false, message: 'Failed to send message. Please try again or contact us directly at shazabjamildhami@gmail.com' },
      { status: 500 }
    )
  }
}