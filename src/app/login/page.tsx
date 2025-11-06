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
        redirectTo: `${window.location.origin}/ai`
      }
    })
    if (error) alert(error.message)
  }

  return (
    <>
      <style jsx global>{`
        :root {
          --bg: #f6f8fb;
          --card: #ffffff;
          --muted: #6b7280;
          --primary: #0b5cff;
          --accent: #0ea5e9;
          --border: #e6e9ef;
          --radius: 12px;
          --shadow: 0 8px 30px rgba(12,18,27,0.06);
          --glass: rgba(255,255,255,0.6);
          font-family: "Inter", system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
        }

        html, body {
          height: 100vh;
          margin: 0;
          background: linear-gradient(180deg, var(--bg), #eef3fb 120%);
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

        /* Layout */
        .split {
          min-height: 100vh;
          display: flex;
          align-items: stretch;
          justify-content: stretch;
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
        }

        .left, .right {
          flex: 1 1 50%;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        /* Left (form) */
        .panel {
          width: 100%;
          max-width: 520px;
          padding: 44px;
          background: var(--card);
          border-radius: 16px;
          box-shadow: var(--shadow);
          border: 1px solid var(--border);
        }

        .brand {
          display: flex;
          align-items: center;
          gap: 12px;
          margin-bottom: 18px;
        }

        .logo {
          width: 44px;
          height: 44px;
          border-radius: 10px;
          background: linear-gradient(135deg, var(--primary), var(--accent));
          display: flex;
          align-items: center;
          justify-content: center;
          color: white;
          font-weight: 700;
          box-shadow: 0 6px 18px rgba(11,92,255,0.12);
        }

        h1 {
          margin: 0;
          font-size: 22px;
          font-weight: 700;
          color: #0f1724;
        }

        p.lead {
          margin: 8px 0 26px 0;
          color: var(--muted);
          font-size: 14px;
        }

        /* OAuth button */
        .oauth {
          display: flex;
          gap: 12px;
          align-items: center;
          padding: 12px 14px;
          border-radius: 10px;
          border: 1px solid var(--border);
          background: white;
          cursor: pointer;
          width: 100%;
          box-shadow: 0 1px 0 rgba(12,18,27,0.02);
          transition: transform .12s ease, box-shadow .12s ease;
          text-decoration: none;
        }

        .oauth:hover {
          transform: translateY(-2px);
          box-shadow: 0 12px 30px rgba(12,18,27,0.06);
        }

        .oauth img {
          width: 18px;
          height: 18px;
        }

        .oauth span {
          flex: 1;
          text-align: center;
          font-weight: 600;
          color: #111827;
          font-size: 14px;
        }

        .divider {
          display: flex;
          gap: 12px;
          align-items: center;
          margin: 20px 0;
          color: var(--muted);
          font-size: 13px;
        }

        .divider:before, .divider:after {
          content: "";
          flex: 1;
          height: 1px;
          background: var(--border);
          border-radius: 2px;
        }

        form .field {
          margin-bottom: 14px;
        }

        label {
          display: block;
          font-size: 13px;
          color: #374151;
          margin-bottom: 8px;
          font-weight: 600;
        }

        input[type="email"], input[type="password"] {
          width: 100%;
          padding: 12px 14px;
          border-radius: 10px;
          border: 1px solid var(--border);
          background: #fbfdff;
          font-size: 14px;
          outline: none;
          transition: box-shadow .12s ease, border-color .12s ease;
          box-sizing: border-box;
        }

        input:focus {
          box-shadow: 0 6px 20px rgba(14,165,233,0.12);
          border-color: var(--accent);
        }

        .row {
          display: flex;
          align-items: center;
          justify-content: space-between;
          font-size: 13px;
          color: var(--muted);
          margin-bottom: 18px;
        }

        .remember input {
          width: 16px;
          height: 16px;
          accent-color: var(--primary);
        }

        button.cta {
          width: 100%;
          padding: 12px 14px;
          border-radius: 10px;
          background: linear-gradient(90deg, var(--primary), #3d7bff);
          border: 0;
          color: white;
          font-weight: 700;
          font-size: 15px;
          cursor: pointer;
          box-shadow: 0 10px 30px rgba(11,92,255,0.12);
          transition: transform .12s ease, opacity .12s ease;
        }

        button.cta:active {
          transform: translateY(1px);
        }

        button.cta:disabled {
          opacity: 0.7;
          cursor: not-allowed;
        }

        .muted-note {
          margin-top: 14px;
          font-size: 13px;
          color: var(--muted);
          text-align: center;
        }

        .muted-note a {
          color: var(--primary);
          font-weight: 700;
          text-decoration: none;
        }

        /* Right (image) */
        .right {
          position: relative;
          overflow: hidden;
        }

        .hero {
          width: 100%;
          height: 100%;
          min-height: 420px;
          background-image: url('https://images.unsplash.com/photo-1531379410503-4a3b6a6f6a4b?auto=format&fit=crop&w=1400&q=80');
          background-size: cover;
          background-position: center;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .overlay {
          width: 100%;
          height: 100%;
          background: linear-gradient(180deg, rgba(3,7,18,0.18), rgba(2,6,23,0.46));
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 32px;
        }

        .hero-card {
          max-width: 520px;
          color: white;
          text-align: left;
          border-radius: 12px;
          padding: 28px;
          backdrop-filter: blur(6px);
          background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.02));
          box-shadow: 0 10px 40px rgba(2,6,23,0.35);
          border: 1px solid rgba(255,255,255,0.06);
        }

        .hero h2 {
          margin: 0;
          font-size: 28px;
          line-height: 1.05;
          font-weight: 700;
        }

        .hero p {
          margin-top: 12px;
          color: rgba(255,255,255,0.85);
          font-size: 15px;
        }

        .hero-tags {
          margin-top: 18px;
          display: flex;
          gap: 10px;
          flex-wrap: wrap;
        }

        .hero-tag {
          background: rgba(255,255,255,0.06);
          padding: 8px 12px;
          border-radius: 8px;
          font-size: 13px;
        }

        /* Small screens */
        @media (max-width: 900px) {
          .right {
            display: none;
          }
          .left {
            flex: 1 1 100%;
            padding: 40px 18px;
            align-items: flex-start;
            justify-content: flex-start;
          }
          .panel {
            margin-top: 28px;
            width: 100%;
            box-shadow: none;
            border-radius: 10px;
          }
        }

        @media (max-width: 420px) {
          .panel {
            padding: 20px;
          }
          .brand h1 {
            font-size: 18px;
          }
          .hero h2 {
            font-size: 20px;
          }
        }
      `}</style>

      <div className="split">
        {/* LEFT: form */}
        <div className="left">
          <div className="panel" role="main">
            <div className="brand">
              <div>
                <h1>{isSignUp ? 'Create Account' : 'Welcome back'}</h1>
                <p className="lead">
                  {isSignUp 
                    ? 'Join zehanxtech — secure, fast, enterprise-ready AI platform.'
                    : 'Sign in to continue to zehanxtech — secure, fast, enterprise-ready.'
                  }
                </p>
              </div>
            </div>

            {/* GOOGLE OAUTH BUTTON */}
            <button className="oauth" onClick={handleGoogleAuth} disabled={loading}>
              <img 
                src="data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 533.5 544.3'><path fill='%23EA4335' d='M533.5 278.4c0-18.4-1.6-36-4.6-53.1H272v100.6h146.9c-6.3 33.8-25.6 62.5-54.6 81.5l88.1 68.3c51.4-47.5 81.1-117 81.1-197.3z'/><path fill='%234285F4' d='M272 544.3c73.8 0 135.8-24.5 181.1-66.5l-88.1-68.3c-24.5 16.5-56 26.3-93 26.3-71.5 0-132.1-48.2-153.8-113.1l-90.3 69.6c41.7 82.6 128 141.9 244.1 141.9z'/><path fill='%2340C853' d='M118.2 327.9c-11.7-34.9-11.7-72.4 0-107.3l-90.3-69.6C7.5 219 0 244.6 0 272c0 27.4 7.5 53 27.9 121.1l90.3-65.2z'/><path fill='%23FBBC05' d='M272 107.7c39.9 0 75.7 13.7 104 40.6l78-78C401.4 25.6 347.1 0 272 0 155.9 0 69.6 59.3 27.9 141.9l90.3 69.6C139.9 156 200.5 107.7 272 107.7z'/></svg>" 
                alt="Google" 
              />
              <span>Continue with Google</span>
            </button>

            <div className="divider">or sign in with email</div>

            {/* EMAIL / PASSWORD FORM */}
            <form onSubmit={handleEmailAuth}>
              <div className="field">
                <label htmlFor="email">Email address</label>
                <input 
                  id="email" 
                  name="email" 
                  type="email" 
                  autoComplete="email" 
                  required 
                  placeholder="you@company.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  disabled={loading}
                />
              </div>

              <div className="field">
                <label htmlFor="password">Password</label>
                <input 
                  id="password" 
                  name="password" 
                  type="password" 
                  autoComplete="current-password" 
                  required 
                  placeholder="••••••••"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  disabled={loading}
                  minLength={6}
                />
              </div>

              <div className="row">
                <label className="remember">
                  <input type="checkbox" name="remember" /> Remember me
                </label>
                <a href="/forgot" style={{ color: 'var(--primary)', textDecoration: 'none', fontWeight: '600' }}>
                  Forgot password?
                </a>
              </div>

              <button className="cta" type="submit" disabled={loading}>
                {loading ? (isSignUp ? 'Creating Account...' : 'Signing In...') : (isSignUp ? 'Create Account' : 'Sign in')}
              </button>

              <p className="muted-note">
                {isSignUp ? 'Already have an account? ' : "Don't have an account? "}
                <button 
                  type="button"
                  onClick={() => setIsSignUp(!isSignUp)}
                  style={{ 
                    background: 'none', 
                    border: 'none', 
                    color: 'var(--primary)', 
                    fontWeight: '700', 
                    textDecoration: 'none',
                    cursor: 'pointer',
                    fontSize: 'inherit'
                  }}
                  disabled={loading}
                >
                  {isSignUp ? 'Sign in' : 'Create one'}
                </button>
              </p>
            </form>
          </div>
        </div>

        {/* RIGHT: image/marketing */}
        <div className="right">
          <div className="hero">
            <div className="overlay">
              <div className="hero-card">
                <h2>Build intelligent systems<br />for modern enterprises</h2>
                <p>Create custom AI models with zero coding — from sentiment analysis to computer vision, deployed instantly to production.</p>
                
                <div className="hero-tags">
                  <div className="hero-tag">Auto ML</div>
                  <div className="hero-tag">Zero Code</div>
                  <div className="hero-tag">Instant Deploy</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}