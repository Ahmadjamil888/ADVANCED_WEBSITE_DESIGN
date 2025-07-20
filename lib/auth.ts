import NextAuth from "next-auth"
import CredentialsProvider from "next-auth/providers/credentials"
import bcrypt from "bcryptjs"

export const authOptions = {
  providers: [
    CredentialsProvider({
      name: "credentials",
      credentials: {
        email: { label: "Email", type: "email" },
        password: { label: "Password", type: "password" }
      },
      async authorize(credentials) {
        if (!credentials?.email || !credentials?.password) {
          return null
        }

        // For demo purposes, using a simple check
        // In production, you'd check against a database
        const user = {
          id: "1",
          email: credentials.email,
          name: "User"
        }

        return user
      }
    })
  ],
  pages: {
    signIn: "/auth/signin",
    signUp: "/auth/signup"
  },
  session: {
    strategy: "jwt" as const
  }
}

export default NextAuth(authOptions)