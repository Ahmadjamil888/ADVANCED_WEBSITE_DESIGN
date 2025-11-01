'use client'

import React, { useState, useEffect } from 'react'
import { supabase } from '@/lib/supabase'
import { useAuth } from '@/contexts/AuthContext'
import { useRouter } from 'next/navigation'

export default function LoginPage() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [isSignUp, setIsSignUp] = useState(false)
  const { user } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (user) {
      router.push('/ai-workspace')
    }
  }, [user, router])

  const handleEmailAuth = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!supabase) {
      alert('Authentication service is not available')
      return
    }
    
    setLoading(true)

    try {
      if (isSignUp) {
        const { error } = await supabase.auth.signUp({
          email,
          password,
        })
        if (error) throw error
        alert('Check your email for the confirmation link!')
      } else {
        const { error } = await supabase.auth.signInWithPassword({
          email,
          password,
        })
        if (error) throw error
      }
    } catch (error: any) {
      alert(error.message)
    } finally {
      setLoading(false)
    }
  }

  const handleGoogleAuth = async () => {
    if (!supabase) {
      alert('Authentication service is not available')
      return
    }
    
    const { error } = await supabase.auth.signInWithOAuth({
      provider: 'google',
      options: {
        redirectTo: `${window.location.origin}/ai-workspace`
      }
    })
    if (error) alert(error.message)
  }

  const handleGitHubAuth = async () => {
    if (!supabase) {
      alert('Authentication service is not available')
      return
    }
    
    const { error } = await supabase.auth.signInWithOAuth({
      provider: 'github',
      options: {
        redirectTo: `${window.location.origin}/ai-workspace`
      }
    })
    if (error) alert(error.message)
  }

  return (
    <>
      <style jsx global>{`
        * {
          margin: 0;
          padding: 0;
          box-sizing: border-box;
        }
        html, body {
          height: 100%;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
        }
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
        .login-container {
          min-height: 100vh;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 20px;
          position: relative;
          overflow: hidden;
        }
        .login-container::before {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="white" opacity="0.1"/><circle cx="75" cy="75" r="1" fill="white" opacity="0.1"/><circle cx="50" cy="10" r="0.5" fill="white" opacity="0.1"/><circle cx="10" cy="60" r="0.5" fill="white" opacity="0.1"/><circle cx="90" cy="40" r="0.5" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
          pointer-events: none;
        }
        .login-card {
          background: rgba(255, 255, 255, 0.95);
          backdrop-filter: blur(20px);
          border-radius: 24px;
          padding: 48px;
          box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
          width: 100%;
          max-width: 440px;
          animation: fadeIn 0.6s ease-out;
          border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .logo-section {
          text-align: center;
          margin-bottom: 40px;
        }
        .logo {
          width: 80px;
          height: 80px;
          border-radius: 20px;
          margin: 0 auto 20px;
          display: block;
          box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        }
        .brand-title {
          font-size: 28px;
          font-weight: 700;
          color: #1a1a1a;
          margin-bottom: 8px;
          letter-spacing: -0.5px;
        }
        .brand-subtitle {
          font-size: 16px;
          color: #6b7280;
          margin-bottom: 8px;
        }
        .welcome-title {
          font-size: 32px;
          font-weight: 800;
          color: #1a1a1a;
          margin-bottom: 32px;
          text-align: center;
          letter-spacing: -1px;
        }
        .oauth-section {
          margin-bottom: 32px;
        }
        .oauth-button {
          width: 100%;
          padding: 16px 24px;
          border: 2px solid #e5e7eb;
          border-radius: 12px;
          background: white;
          color: #374151;
          font-size: 16px;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s ease;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 12px;
          margin-bottom: 12px;
        }
        .oauth-button:hover {
          border-color: #d1d5db;
          background: #f9fafb;
          transform: translateY(-1px);
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }
        .oauth-button:active {
          transform: translateY(0);
        }
        .google-button {
          border-color: #ea4335;
          color: #ea4335;
        }
        .google-button:hover {
          background: #fef2f2;
          border-color: #dc2626;
        }
        .github-button {
          border-color: #24292e;
          color: #24292e;
        }
        .github-button:hover {
          background: #f6f8fa;
          border-color: #1b1f23;
        }
        .divider {
          text-align: center;
          margin: 32px 0;
          position: relative;
        }
        .divider::before {
          content: '';
          position: absolute;
          top: 50%;
          left: 0;
          right: 0;
          height: 1px;
          background: #e5e7eb;
        }
        .divider-text {
          background: rgba(255, 255, 255, 0.95);
          padding: 0 20px;
          color: #6b7280;
          font-size: 14px;
          font-weight: 500;
        }
        .form-section {
          margin-bottom: 32px;
        }
        .form-group {
          margin-bottom: 20px;
        }
        .form-label {
          display: block;
          font-size: 14px;
          font-weight: 600;
          color: #374151;
          margin-bottom: 8px;
        }
        .form-input {
          width: 100%;
          padding: 16px;
          border: 2px solid #e5e7eb;
          border-radius: 12px;
          font-size: 16px;
          color: #1f2937;
          background: white;
          transition: all 0.2s ease;
        }
        .form-input:focus {
          outline: none;
          border-color: #667eea;
          box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        .form-input::placeholder {
          color: #9ca3af;
        }
        .primary-button {
          width: 100%;
          padding: 16px 24px;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          border: none;
          border-radius: 12px;
          font-size: 16px;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s ease;
          position: relative;
          overflow: hidden;
        }
        .primary-button:hover {
          transform: translateY(-2px);
          box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        }
        .primary-button:active {
          transform: translateY(0);
        }
        .primary-button:disabled {
          opacity: 0.7;
          cursor: not-allowed;
          transform: none;
        }
        .loading-spinner {
          animation: pulse 1.5s ease-in-out infinite;
        }
        .toggle-button {
          background: none;
          border: none;
          color: #667eea;
          font-size: 14px;
          font-weight: 600;
          cursor: pointer;
          text-decoration: underline;
          transition: color 0.2s ease;
        }
        .toggle-button:hover {
          color: #5a67d8;
        }
        .footer-text {
          text-align: center;
          font-size: 12px;
          color: #9ca3af;
          line-height: 1.5;
        }
        .footer-text a {
          color: #667eea;
          text-decoration: none;
        }
        .footer-text a:hover {
          text-decoration: underline;
        }
        
        /* Mobile Responsive */
        @media (max-width: 768px) {
          .login-container {
            padding: 16px;
          }
          .login-card {
            padding: 32px 24px;
            border-radius: 20px;
          }
          .logo {
            width: 64px;
            height: 64px;
          }
          .brand-title {
            font-size: 24px;
          }
          .welcome-title {
            font-size: 28px;
            margin-bottom: 24px;
          }
          .oauth-button {
            padding: 14px 20px;
            font-size: 15px;
          }
          .form-input {
            padding: 14px;
            font-size: 16px; /* Prevent zoom on iOS */
          }
          .primary-button {
            padding: 14px 20px;
            font-size: 15px;
          }
        }
        
        @media (max-width: 480px) {
          .login-container {
            padding: 12px;
          }
          .login-card {
            padding: 24px 20px;
          }
          .brand-title {
            font-size: 22px;
          }
          .welcome-title {
            font-size: 24px;
          }
        }
      `}</style>

      <div className="login-container">
        <div className="login-card">
          {/* Logo Section */}
          <div className="logo-section">
            <img src="/logo.jpg" alt="zehanxtech" className="logo" />
            <h1 className="brand-title">zehanxtech</h1>
            <p className="brand-subtitle">AI for Better Humanity</p>
          </div>

          <h2 className="welcome-title">
            {isSignUp ? 'Join the Future' : 'Welcome Back'}
          </h2>

          {/* OAuth Buttons */}
          <div className="oauth-section">
            <button
              onClick={handleGoogleAuth}
              className="oauth-button google-button"
              disabled={loading}
            >
              <svg width="20" height="20" viewBox="0 0 24 24">
                <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
              </svg>
              Continue with Google
            </button>

            <button
              onClick={handleGitHubAuth}
              className="oauth-button github-button"
              disabled={loading}
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
              </svg>
              Continue with GitHub
            </button>
          </div>

          {/* Divider */}
          <div className="divider">
            <span className="divider-text">or continue with email</span>
          </div>

          {/* Email Form */}
          <form onSubmit={handleEmailAuth} className="form-section">
            <div className="form-group">
              <label htmlFor="email" className="form-label">Email Address</label>
              <input
                id="email"
                type="email"
                placeholder="Enter your email address"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="form-input"
                required
                disabled={loading}
              />
            </div>

            <div className="form-group">
              <label htmlFor="password" className="form-label">Password</label>
              <input
                id="password"
                type="password"
                placeholder="Enter your password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="form-input"
                required
                disabled={loading}
                minLength={6}
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="primary-button"
            >
              {loading ? (
                <span className="loading-spinner">
                  {isSignUp ? 'Creating Account...' : 'Signing In...'}
                </span>
              ) : (
                isSignUp ? 'Create Account' : 'Sign In'
              )}
            </button>
          </form>

          {/* Toggle Sign Up/Sign In */}
          <div style={{ textAlign: 'center', marginBottom: '24px' }}>
            <button
              onClick={() => setIsSignUp(!isSignUp)}
              className="toggle-button"
              disabled={loading}
            >
              {isSignUp ? 'Already have an account? Sign in' : "Don't have an account? Join us"}
            </button>
          </div>

          {/* Footer */}
          <div className="footer-text">
            By continuing, you agree to our{' '}
            <a href="/terms" target="_blank">Terms of Service</a> and{' '}
            <a href="/privacy" target="_blank">Privacy Policy</a>
          </div>
        </div>
      </div>
    </>
  )
}