'use client'

import React, { useState, useEffect } from 'react'
import { supabase } from '@/lib/supabase'
import { useAuth } from '@/contexts/AuthContext'
import { useRouter } from 'next/navigation'
import { 
  Flex, 
  Button, 
  Text, 
  Heading, 
  Input, 
  Card, 
  Avatar,
  PasswordInput
} from "@/once-ui/components"

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
    <Flex 
      fillWidth 
      fillHeight 
      center 
      background="brand-medium"
      style={{ 
        position: 'fixed',
        top: 0,
        left: 0,
        zIndex: 9999
      }}
    >
      <Card 
        padding="xl" 
        background="neutral-weak" 
        radius="l"
        shadow="l"
        maxWidth={26}
        fillWidth
      >
        <Flex direction="column" gap="l" center>
          {/* Logo Section */}
          <Flex direction="column" gap="s" center>
            <Avatar 
              size="xl" 
              src="/logo.jpg"
            />
            <Heading variant="heading-strong-l" onBackground="neutral-weak">
              zehanxtech
            </Heading>
            <Text variant="body-default-s" onBackground="neutral-medium">
              AI for Better Humanity
            </Text>
          </Flex>

          <Heading variant="heading-strong-xl" onBackground="neutral-weak">
            {isSignUp ? 'Join Us' : 'Welcome'}
          </Heading>

          {/* OAuth Buttons */}
          <Flex direction="column" gap="s" fillWidth>
            <Button
              onClick={handleGoogleAuth}
              variant="secondary"
              size="l"
              fillWidth
            >
              Continue with Google
            </Button>

            <Button
              onClick={handleGitHubAuth}
              variant="secondary"
              size="l"
              fillWidth
            >
              Continue with GitHub
            </Button>
          </Flex>

          {/* Divider */}
          <Flex center fillWidth>
            <Text variant="body-default-s" onBackground="neutral-medium">
              or continue with email
            </Text>
          </Flex>

          {/* Email Form */}
          <form onSubmit={handleEmailAuth} style={{ width: '100%' }}>
            <Flex direction="column" gap="m" fillWidth>
              <Input
                id="email"
                label="Email"
                type="email"
                placeholder="Enter your email address"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />

              <PasswordInput
                id="password"
                label="Password"
                placeholder="Enter your password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />

              <Button
                type="submit"
                disabled={loading}
                variant="primary"
                size="l"
                fillWidth
              >
                {loading ? 'Authenticating...' : isSignUp ? 'Create Account' : 'Sign In'}
              </Button>
            </Flex>
          </form>

          <Button
            onClick={() => setIsSignUp(!isSignUp)}
            variant="tertiary"
            size="m"
          >
            {isSignUp ? 'Already have an account? Sign in' : "Don't have an account? Join us"}
          </Button>

          <Text variant="body-default-xs" onBackground="neutral-medium" align="center">
            By continuing, you agree to our Terms of Service and Privacy Policy
          </Text>
        </Flex>
      </Card>
    </Flex>
  )
}