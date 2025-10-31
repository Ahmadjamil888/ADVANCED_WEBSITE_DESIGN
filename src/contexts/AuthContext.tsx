'use client'

import React, { createContext, useContext, useEffect, useState } from 'react'
import { User, Session } from '@supabase/supabase-js'
import { supabase } from '@/lib/supabase'

interface AuthContextType {
  user: User | null
  session: Session | null
  loading: boolean
  signOut: () => Promise<void>
}

const AuthContext = createContext<AuthContextType>({
  user: null,
  session: null,
  loading: true,
  signOut: async () => {}
})

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null)
  const [session, setSession] = useState<Session | null>(null)
  const [loading, setLoading] = useState(true)
  const [initialized, setInitialized] = useState(false)



  // Fallback timeout to prevent infinite loading
  useEffect(() => {
    const timeout = setTimeout(() => {
      console.log('AuthContext: Timeout reached, forcing loading to false')
      setLoading(false)
      setInitialized(true)
    }, 1000) // Even faster timeout - 1 second

    return () => clearTimeout(timeout)
  }, [])

  useEffect(() => {
    console.log('AuthContext: Supabase client:', supabase ? 'initialized' : 'null')
    console.log('AuthContext: Environment check:', {
      url: !!process.env.NEXT_PUBLIC_SUPABASE_URL,
      key: !!process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY
    })
    
    if (!supabase) {
      console.log('AuthContext: No Supabase client, setting loading to false')
      setLoading(false)
      return
    }

    // Get initial session immediately
    const initializeAuth = () => {
      if (!supabase) {
        setLoading(false)
        setInitialized(true)
        return
      }

      // Try to get session synchronously first
      supabase.auth.getSession()
        .then(({ data: { session }, error }) => {
          console.log('AuthContext: Initial session check:', { session: !!session, error })
          
          if (error) {
            console.error('AuthContext: Session error:', error)
          }
          
          setSession(session)
          setUser(session?.user ?? null)
          setLoading(false)
          setInitialized(true)
        })
        .catch((error) => {
          console.error('AuthContext: Error getting session:', error)
          setLoading(false)
          setInitialized(true)
        })
    }

    initializeAuth()

    // Listen for auth changes
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange(async (event, session) => {
      console.log('AuthContext: Auth state changed:', event, !!session)
      setSession(session)
      setUser(session?.user ?? null)
      setLoading(false)

      // Create user record in our users table if it doesn't exist (async, don't block)
      if (event === 'SIGNED_IN' && session?.user && supabase) {
        // Don't await this - let it run in background
        const createUserIfNeeded = async () => {
          try {
            if (!supabase) return
            
            const { data: existingUser } = await supabase
              .from('users')
              .select('id')
              .eq('id', session.user.id)
              .single()

            if (!existingUser) {
              await supabase.from('users').insert({
                id: session.user.id,
                email: session.user.email!,
                username: session.user.user_metadata?.full_name || session.user.email!.split('@')[0],
              })
            }
          } catch (error) {
            console.log('Background user creation error:', error)
          }
        }
        createUserIfNeeded()
      }
    })

    return () => subscription.unsubscribe()
  }, [])

  const signOut = async () => {
    if (supabase) {
      await supabase.auth.signOut()
    }
  }

  const value = {
    user,
    session,
    loading: loading && !initialized,
    signOut,
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}