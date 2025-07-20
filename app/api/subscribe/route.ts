import { NextRequest, NextResponse } from 'next/server'

export async function POST(request: NextRequest) {
  try {
    const { email, name, message, company } = await request.json()

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

    // OPTION 1: Using EmailJS (Free service - No server setup needed)
    // This will send emails directly from the frontend using EmailJS service
    
    // OPTION 2: Using Resend (Free tier: 3000 emails/month)
    if (process.env.RESEND_API_KEY) {
      try {
        const response = await fetch('https://api.resend.com/emails', {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${process.env.RESEND_API_KEY}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            from: 'contact@zehanx.com', // You'll need to verify this domain with Resend
            to: ['shazabjamildhami@gmail.com'],
            reply_to: email,
            subject: `New Contact Form Submission from ${name} - Zehan X Technologies`,
            html: `
              <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 8px;">
                <h2 style="color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px;">
                  New Contact Form Submission - Zehan X Technologies
                </h2>
                
                <div style="margin: 20px 0;">
                  <h3 style="color: #555; margin-bottom: 15px;">Contact Details:</h3>
                  <table style="width: 100%; border-collapse: collapse;">
                    <tr style="background-color: #f8f9fa;">
                      <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold; width: 30%;">Name:</td>
                      <td style="padding: 10px; border: 1px solid #ddd;">${name}</td>
                    </tr>
                    <tr>
                      <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">Email:</td>
                      <td style="padding: 10px; border: 1px solid #ddd;"><a href="mailto:${email}">${email}</a></td>
                    </tr>
                    ${company ? `
                    <tr style="background-color: #f8f9fa;">
                      <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">Company:</td>
                      <td style="padding: 10px; border: 1px solid #ddd;">${company}</td>
                    </tr>
                    ` : ''}
                    <tr>
                      <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">Submitted:</td>
                      <td style="padding: 10px; border: 1px solid #ddd;">${new Date().toLocaleString('en-US', { 
                        timeZone: 'UTC',
                        year: 'numeric',
                        month: 'long',
                        day: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit'
                      })} UTC</td>
                    </tr>
                  </table>
                </div>

                <div style="margin: 20px 0;">
                  <h3 style="color: #555; margin-bottom: 15px;">Message:</h3>
                  <div style="background-color: #f8f9fa; padding: 15px; border-left: 4px solid #007bff; border-radius: 4px;">
                    ${message.replace(/\n/g, '<br>')}
                  </div>
                </div>

                <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 12px;">
                  <p>This message was sent from the Zehan X Technologies contact form.</p>
                  <p>Reply directly to this email to respond to ${name}.</p>
                </div>
              </div>
            `,
          }),
        })

        if (response.ok) {
          return NextResponse.json({ 
            success: true, 
            message: 'Message sent successfully! We\'ll get back to you within 24 hours.' 
          })
        } else {
          throw new Error('Resend API failed')
        }
      } catch (resendError) {
        console.error('Resend error:', resendError)
        // Fall through to other options
      }
    }

    // OPTION 3: Using Formspree (Free tier: 50 submissions/month)
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
            company: company,
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
        // Fall through to simple logging
      }
    }

    // OPTION 4: Simple logging to console and database (fallback)
    // This logs the message and you can check server logs
    console.log('=== NEW CONTACT FORM SUBMISSION ===')
    console.log('Name:', name)
    console.log('Email:', email)
    console.log('Company:', company || 'Not provided')
    console.log('Message:', message)
    console.log('Timestamp:', new Date().toISOString())
    console.log('=====================================')

    // You can also save to a database here if needed
    // For now, we'll return success and you can check the server logs
    
    return NextResponse.json({ 
      success: true, 
      message: 'Message received! We\'ll get back to you within 24 hours. (Check server logs for details)' 
    })

  } catch (error) {
    console.error('Contact form error:', error)
    
    return NextResponse.json(
      { success: false, message: 'Failed to send message. Please try again or contact us directly at shazabjamildhami@gmail.com' },
      { status: 500 }
    )
  }
}