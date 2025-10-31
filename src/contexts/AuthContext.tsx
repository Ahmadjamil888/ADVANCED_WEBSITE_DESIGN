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

  // Fallback timeout to prevent infinite loading
  useEffect(() => {
    const timeout = setTimeout(() => {
      console.log('AuthContext: Timeout reached, checking if we should force loading to false')
      // Only force loading to false if we haven't already resolved the auth state
      if (loading) {
        console.log('AuthContext: Forcing loading to false due to timeout')
        setLoading(false)
      }
    }, 3000) // Reduced to 3 seconds for faster response

    return () => clearTimeout(timeout)
  }, [loading])

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

    // Get initial session with faster timeout
    const getInitialSession = async () => {
      try {
        if (!supabase) {
          setLoading(false)
          return
        }
        
        const { data: { session }, error } = await supabase.auth.getSession()
        console.log('AuthContext: Initial session check:', { session: !!session, error })
        
        if (error) {
          console.error('AuthContext: Session error:', error)
        }
        
        setSession(session)
        setUser(session?.user ?? null)
        setLoading(false)
      } catch (error) {
        console.error('AuthContext: Error getting session:', error)
        setLoading(false)
      }
    }

    getInitialSession()

    // Listen for auth changes
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange(async (event, session) => {
      setSession(session)
      setUser(session?.user ?? null)
      setLoading(false)

      // Create user record in our users table if it doesn't exist
      if (event === 'SIGNED_IN' && session?.user && supabase) {
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
    loading,
    signOut,
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}