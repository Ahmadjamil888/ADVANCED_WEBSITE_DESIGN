import { NextResponse } from 'next/server';

// This is a test endpoint to verify Web3Forms is working correctly
// Access this endpoint at /api/test-email to send a test email
export async function GET() {
  try {
    // Check if Web3Forms access key is available
    if (!process.env.WEB3FORMS_ACCESS_KEY) {
      return NextResponse.json(
        { success: false, message: 'Web3Forms access key is not configured' },
        { status: 500 }
      );
    }

    // IMPORTANT: Web3Forms requires the access_key to be sent in the payload
    // The email will be sent to the email associated with this access_key in your Web3Forms account
    const payload = {
      access_key: process.env.WEB3FORMS_ACCESS_KEY,
      subject: `TEST EMAIL - Contact Form Verification`,
      from_name: 'Zehan X Technologies Website',
      name: 'Test User',
      email: 'test@example.com',
      message: 'This is a test message to verify the contact form is working correctly.',
      company: 'Test Company',
    };
    
    console.log('📧 TEST: Sending to Web3Forms with access key:', process.env.WEB3FORMS_ACCESS_KEY);
    
    const response = await fetch('https://api.web3forms.com/submit', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify(payload)
    });

    const responseText = await response.text();
    let result;
    
    try {
      // Try to parse as JSON
      result = JSON.parse(responseText);
    } catch {
      console.error('Failed to parse Web3Forms response as JSON:', responseText);
      return NextResponse.json(
        { success: false, message: `Invalid response from Web3Forms: ${responseText.substring(0, 100)}...` },
        { status: 500 }
      );
    }
    
    if (result.success) {
      console.log('✅ TEST: Web3Forms test email sent successfully:', {
        timestamp: new Date().toISOString(),
        responseData: result
      });
      
      return NextResponse.json({ 
        success: true, 
        message: 'Test email sent successfully! Check your inbox at shazabjamildhami@gmail.com',
        details: result
      });
    } else {
      console.error('❌ TEST: Web3Forms test email failed:', {
        error: result.message || 'Unknown error',
        statusCode: response.status,
        responseData: result
      });
      
      return NextResponse.json(
        { 
          success: false, 
          message: `Web3Forms test failed: ${result.message || 'Unknown error'} (Status: ${response.status})`,
          details: result
        },
        { status: 500 }
      );
    }
  } catch (error) {
    console.error('Test email error:', error);
    
    return NextResponse.json(
      { success: false, message: 'Failed to send test email. See server logs for details.' },
      { status: 500 }
    );
  }
}