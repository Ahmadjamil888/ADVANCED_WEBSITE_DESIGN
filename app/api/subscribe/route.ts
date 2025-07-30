import { NextRequest, NextResponse } from 'next/server'
import nodemailer from 'nodemailer'

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

    // OPTION 1: Using Gmail SMTP (Direct email sending) - PRIMARY METHOD
    // This requires Gmail App Password to be set up
    if (process.env.GMAIL_USER && process.env.GMAIL_APP_PASSWORD) {
      try {
        const transporter = nodemailer.createTransport({
          service: 'gmail',
          auth: {
            user: process.env.GMAIL_USER,
            pass: process.env.GMAIL_APP_PASSWORD,
          },
        });

        const mailOptions = {
          from: `"Zehan X Technologies Website" <${process.env.GMAIL_USER}>`,
          to: 'shazabjamildhami@gmail.com',
          replyTo: email,
          subject: `New Contact Form Submission from ${name} - Zehan X Technologies`,
          html: `
            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
              <h2 style="color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px;">
                New Contact Form Submission
              </h2>
              
              <div style="background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0;">
                <h3 style="color: #007bff; margin-top: 0;">Contact Details:</h3>
                <p><strong>Name:</strong> ${name}</p>
                <p><strong>Email:</strong> ${email}</p>
                <p><strong>Company:</strong> ${company || 'Not provided'}</p>
              </div>
              
              <div style="background-color: #fff; padding: 20px; border: 1px solid #dee2e6; border-radius: 5px;">
                <h3 style="color: #007bff; margin-top: 0;">Message:</h3>
                <p style="line-height: 1.6; white-space: pre-wrap;">${message}</p>
              </div>
              
              <div style="margin-top: 20px; padding: 15px; background-color: #e9ecef; border-radius: 5px;">
                <p style="margin: 0; font-size: 14px; color: #6c757d;">
                  <strong>Submitted:</strong> ${new Date().toLocaleString()}<br>
                  <strong>Reply directly to:</strong> ${email}
                </p>
              </div>
            </div>
          `,
          text: `
New Contact Form Submission - Zehan X Technologies

Name: ${name}
Email: ${email}
Company: ${company || 'Not provided'}

Message:
${message}

Submitted: ${new Date().toLocaleString()}
Reply directly to: ${email}
          `
        };

        await transporter.sendMail(mailOptions);
        
        console.log('✅ Gmail SMTP email sent successfully:', {
          name,
          email,
          company: company || 'Not provided',
          timestamp: new Date().toISOString()
        });
        
        return NextResponse.json({ 
          success: true, 
          message: 'Message sent successfully! We\'ll get back to you within 24 hours.' 
        });
      } catch (gmailError) {
        console.error('Gmail SMTP error:', gmailError);
        // Fall through to next option
      }
    }

    // OPTION 2: Using EmailJS (Free tier: 200 emails/month) - BACKUP
    // This is a client-side email service that works reliably
    if (process.env.EMAILJS_SERVICE_ID && process.env.EMAILJS_TEMPLATE_ID && process.env.EMAILJS_PUBLIC_KEY) {
      try {
        const emailData = {
          service_id: process.env.EMAILJS_SERVICE_ID,
          template_id: process.env.EMAILJS_TEMPLATE_ID,
          user_id: process.env.EMAILJS_PUBLIC_KEY,
          template_params: {
            from_name: name,
            from_email: email,
            to_email: 'shazabjamildhami@gmail.com',
            subject: `New Contact Form Submission from ${name} - Zehan X Technologies`,
            message: message,
            company: company || 'Not provided',
            reply_to: email
          }
        };

        const response = await fetch('https://api.emailjs.com/api/v1.0/email/send', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(emailData)
        });

        if (response.ok) {
          console.log('✅ EmailJS submission successful');
          return NextResponse.json({ 
            success: true, 
            message: 'Message sent successfully! We\'ll get back to you within 24 hours.' 
          });
        } else {
          throw new Error('EmailJS failed');
        }
      } catch (emailjsError) {
        console.error('EmailJS error:', emailjsError);
        // Fall through to next option
      }
    }

    // OPTION 3: Using Web3Forms (Free tier: 250 submissions/month) - BACKUP
    // Using your Web3Forms access key
    if (process.env.WEB3FORMS_ACCESS_KEY) {
      try {
        // IMPORTANT: Web3Forms requires the access_key to be sent in the payload
        // The email will be sent to the email associated with this access_key in your Web3Forms account
        const payload = {
          access_key: process.env.WEB3FORMS_ACCESS_KEY,
          subject: `New Contact Form Submission from ${name} - Zehan X Technologies`,
          from_name: 'Zehan X Technologies Website',
          name: name,
          email: email,
          message: message,
          company: company || 'Not provided',
        };
        
        console.log('📧 Sending to Web3Forms with access key:', process.env.WEB3FORMS_ACCESS_KEY);
        
        const response = await fetch('https://api.web3forms.com/submit', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          },
          body: JSON.stringify(payload)
        })

        const responseText = await response.text();
        let result;
        
        try {
          // Try to parse as JSON
          result = JSON.parse(responseText);
        } catch {
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