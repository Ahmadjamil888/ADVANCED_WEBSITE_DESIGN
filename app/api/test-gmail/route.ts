import { NextRequest, NextResponse } from 'next/server';
import nodemailer from 'nodemailer';

export async function GET(request: NextRequest) {
  try {
    console.log('🧪 Testing Gmail SMTP configuration...');
    
    // Check if Gmail credentials are available
    if (!process.env.GMAIL_USER || !process.env.GMAIL_APP_PASSWORD) {
      return NextResponse.json(
        { 
          success: false, 
          message: 'Gmail credentials not configured',
          details: {
            hasUser: !!process.env.GMAIL_USER,
            hasPassword: !!process.env.GMAIL_APP_PASSWORD
          }
        },
        { status: 500 }
      );
    }

    console.log('📧 Gmail User:', process.env.GMAIL_USER);
    console.log('📧 Gmail Password length:', process.env.GMAIL_APP_PASSWORD.length);

    // Create transporter
    const transporter = nodemailer.createTransport({
      service: 'gmail',
      auth: {
        user: process.env.GMAIL_USER,
        pass: process.env.GMAIL_APP_PASSWORD,
      },
    });

    // Verify connection
    await transporter.verify();
    console.log('✅ Gmail SMTP connection verified');

    // Send test email
    const mailOptions = {
      from: `"Zehan X Technologies Website" <${process.env.GMAIL_USER}>`,
      to: 'shazabjamildhami@gmail.com',
      subject: 'TEST EMAIL - Gmail SMTP Verification',
      html: `
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
          <h2 style="color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px;">
            Gmail SMTP Test Email
          </h2>
          
          <div style="background-color: #d4edda; padding: 20px; border-radius: 5px; margin: 20px 0; border: 1px solid #c3e6cb;">
            <h3 style="color: #155724; margin-top: 0;">✅ Success!</h3>
            <p style="color: #155724; margin: 0;">
              Your Gmail SMTP configuration is working correctly. The contact form should now be able to send emails directly to your Gmail account.
            </p>
          </div>
          
          <div style="margin-top: 20px; padding: 15px; background-color: #e9ecef; border-radius: 5px;">
            <p style="margin: 0; font-size: 14px; color: #6c757d;">
              <strong>Test sent:</strong> ${new Date().toLocaleString()}<br>
              <strong>From:</strong> Zehan X Technologies Contact Form System
            </p>
          </div>
        </div>
      `,
      text: `
Gmail SMTP Test Email

✅ Success! Your Gmail SMTP configuration is working correctly.

The contact form should now be able to send emails directly to your Gmail account.

Test sent: ${new Date().toLocaleString()}
From: Zehan X Technologies Contact Form System
      `
    };

    const info = await transporter.sendMail(mailOptions);
    
    console.log('✅ Test email sent successfully:', info.messageId);
    
    return NextResponse.json({
      success: true,
      message: 'Test email sent successfully! Check your inbox at shazabjamildhami@gmail.com',
      details: {
        messageId: info.messageId,
        timestamp: new Date().toISOString()
      }
    });

  } catch (error) {
    console.error('❌ Gmail SMTP test failed:', error);
    
    return NextResponse.json(
      { 
        success: false, 
        message: 'Gmail SMTP test failed',
        error: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}