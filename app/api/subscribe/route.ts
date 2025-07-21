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
        const formData = new FormData()
        formData.append('access_key', process.env.WEB3FORMS_ACCESS_KEY)
        formData.append('name', name)
        formData.append('email', email)
        formData.append('message', message)
        formData.append('subject', `New Contact Form Submission from ${name} - Zehan X Technologies`)
        
        if (company) {
          formData.append('company', company)
        }
        
        // Add custom fields for better organization
        formData.append('from_name', 'Zehan X Technologies Contact Form')
        formData.append('replyto', email)

        const response = await fetch('https://api.web3forms.com/submit', {
          method: 'POST',
          body: formData
        })

        const result = await response.json()
        
        if (result.success) {
          console.log('✅ Web3Forms submission successful:', {
            name,
            email,
            company: company || 'Not provided',
            timestamp: new Date().toISOString()
          })
          
          return NextResponse.json({ 
            success: true, 
            message: 'Message sent successfully! We\'ll get back to you within 24 hours.' 
          })
        } else {
          console.error('Web3Forms failed:', result)
          throw new Error(`Web3Forms failed: ${result.message || 'Unknown error'}`)
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