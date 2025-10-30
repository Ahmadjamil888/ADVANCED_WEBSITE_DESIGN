'use client'

import React, { useState, useEffect } from 'react'
import { supabase } from '@/lib/supabase'
import { useAuth } from '@/contexts/AuthContext'
import { useRouter } from 'next/navigation'
import Image from 'next/image'

export default function LoginPage() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [isSignUp, setIsSignUp] = useState(false)
  const { user } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (user) {
      router.push('/dashboard')
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
        redirectTo: `${window.location.origin}/dashboard`
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
        redirectTo: `${window.location.origin}/dashboard`
      }
    })
    if (error) alert(error.message)
  }

  return (
    <div style={{
      margin: 0,
      padding: 0,
      height: '100vh',
      width: '100vw',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      position: 'fixed',
      top: 0,
      left: 0,
      zIndex: 9999
    }}>
      <style jsx>{`
        * {
          margin: 0;
          padding: 0;
          box-sizing: border-box;
        }
        
        .login-card {
          background: rgba(255, 255, 255, 1);
          backdrop-filter: blur(20px);
          border-radius: 24px;
          padding: 48px 40px;
          width: 100%;
          max-width: 420px;
          box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
          border: 1px solid rgba(255, 255, 255, 0.2);
          position: relative;
        }
        
        .login-card::before {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          height: 5px;
          background: linear-gradient(90deg, #667eea, #764ba2);
          border-radius: 24px 24px 0 0;
        }
        
        .logo-section {
          text-align: center;
          margin-bottom: 32px;
        }
        
        .logo {
          width: 80px;
          height: 80px;
          border-radius: 20px;
          margin: 0 auto 16px;
          display: block;
          box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
          object-fit: cover;
        }
        
        .brand-name {
          font-size: 24px;
          font-weight: 700;
          color: #1a202c;
          margin-bottom: 4px;
          letter-spacing: -0.5px;
        }
        
        .brand-tagline {
          font-size: 14px;
          color: #718096;
          font-weight: 500;
        }
        
        .form-title {
          font-size: 32px;
          font-weight: 800;
          color: #1a202c;
          text-align: center;
          margin-bottom: 32px;
          letter-spacing: -1px;
        }
        
        .oauth-button {
          width: 100%;
          padding: 16px 20px;
          border: 2px solid rgba(226, 232, 240, 0.8);
          border-radius: 16px;
          background: rgba(255, 255, 255, 1);
          color: #2d3748;
          font-size: 15px;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.3s ease;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 12px;
          margin-bottom: 12px;
          backdrop-filter: blur(5px);
        }
        
        .oauth-button:hover {
          border-color: rgba(102, 126, 234, 0.3);
          transform: translateY(-2px);
          box-shadow: 0 12px 24px rgba(0, 0, 0, 0.1);
          background: rgba(255, 255, 255, 1);
        }
        
        .oauth-icon {
          width: 20px;
          height: 20px;
          flex-shrink: 0;
        }
        
        .divider {
          position: relative;
          text-align: center;
          margin: 28px 0;
        }
        
        .divider::before {
          content: '';
          position: absolute;
          top: 50%;
          left: 0;
          right: 0;
          height: 1px;
          background: rgba(226, 232, 240, 0.8);
        }
        
        .divider-text {
          background: rgba(255, 255, 255, 0.95);
          padding: 0 16px;
          color: #718096;
          font-size: 13px;
          font-weight: 500;
        }
        
        .form-input {
          width: 100%;
          padding: 16px 18px;
          border: 2px solid rgba(226, 232, 240, 0.8);
          border-radius: 16px;
          font-size: 15px;
          color: #1a202c;
          background: rgba(255, 255, 255, 1);
          transition: all 0.3s ease;
          backdrop-filter: blur(5px);
          margin-bottom: 16px;
        }
        
        .form-input:focus {
          outline: none;
          border-color: #667eea;
          background: rgba(255, 255, 255, 0.95);
          box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
          transform: translateY(-1px);
        }
        
        .form-input::placeholder {
          color: #a0aec0;
        }
        
        .submit-button {
          width: 100%;
          padding: 18px;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          border: none;
          border-radius: 16px;
          font-size: 16px;
          font-weight: 700;
          cursor: pointer;
          transition: all 0.3s ease;
          margin-bottom: 24px;
          position: relative;
          overflow: hidden;
        }
        
        .submit-button:hover {
          transform: translateY(-2px);
          box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35);
        }
        
        .submit-button:disabled {
          opacity: 0.7;
          cursor: not-allowed;
          transform: none;
        }
        
        .toggle-button {
          background: none;
          border: none;
          color: #667eea;
          font-size: 14px;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s ease;
          padding: 8px 12px;
          border-radius: 8px;
        }
        
        .toggle-button:hover {
          color: #764ba2;
          background: rgba(102, 126, 234, 0.05);
        }
        
        .terms-text {
          text-align: center;
          font-size: 12px;
          color: #718096;
          line-height: 1.6;
          margin-top: 16px;
        }
        
        .terms-link {
          color: #667eea;
          text-decoration: none;
          font-weight: 500;
        }
        
        .terms-link:hover {
          text-decoration: underline;
        }
        
        .loading-spinner {
          display: inline-block;
          width: 18px;
          height: 18px;
          border: 2px solid rgba(255, 255, 255, 0.3);
          border-radius: 50%;
          border-top-color: #ffffff;
          animation: spin 0.8s ease-in-out infinite;
          margin-right: 8px;
        }
        
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        
        @media (max-width: 480px) {
          .login-card {
            margin: 0;
            border-radius: 0;
            width: 100vw;
            height: 100vh;
            padding: 40px 24px;
            display: flex;
            flex-direction: column;
            justify-content: center;
          }
        }
      `}</style>

      <div className="login-card">
        <div className="logo-section">
          <Image
            src="/logo.jpg"
            alt="zehanxtech"
            width={80}
            height={80}
            className="logo"
          />
          <div className="brand-name">zehanxtech</div>
          <div className="brand-tagline">AI for Better Humanity</div>
        </div>

        <h1 className="form-title">
          {isSignUp ? 'Join Us' : 'Welcome'}
        </h1>

        <div style={{ marginBottom: '28px' }}>
          <button onClick={handleGoogleAuth} className="oauth-button">
            <svg className="oauth-icon" viewBox="0 0 24 24">
              <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
              <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
              <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
              <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
            </svg>
            Continue with Google
          </button>

          <button onClick={handleGitHubAuth} className="oauth-button">
            <svg className="oauth-icon" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
            </svg>
            Continue with GitHub
          </button>
        </div>

        <div className="divider">
          <span className="divider-text">or continue with email</span>
        </div>

        <form onSubmit={handleEmailAuth}>
          <input
            className="form-input"
            type="email"
            placeholder="Enter your email address"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />

          <input
            className="form-input"
            type="password"
            placeholder="Enter your password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />

          <button
            type="submit"
            disabled={loading}
            className="submit-button"
          >
            {loading && <span className="loading-spinner"></span>}
            {loading ? 'Authenticating...' : isSignUp ? 'Create Account' : 'Sign In'}
          </button>
        </form>

        <div style={{ textAlign: 'center', marginBottom: '24px' }}>
          <button
            type="button"
            onClick={() => setIsSignUp(!isSignUp)}
            className="toggle-button"
          >
            {isSignUp ? 'Already have an account? Sign in' : "Don't have an account? Join us"}
          </button>
        </div>

        <div className="terms-text">
          By continuing, you agree to our{' '}
          <a href="#" className="terms-link">Terms of Service</a>{' '}
          and{' '}
          <a href="#" className="terms-link">Privacy Policy</a>
        </div>
      </div>
    </div>
  )
}