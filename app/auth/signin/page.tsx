import { SignIn } from '@clerk/nextjs';
import { Section } from "@/components/ui/section";
import { Badge } from "@/components/ui/badge";
import { Lock } from "lucide-react";
import Link from "next/link";

export default function SignInPage() {
  return (
    <Section className="min-h-screen flex items-center justify-center py-12">
      <div className="max-w-md w-full mx-auto">
        <div className="text-center mb-8">
          <Badge variant="outline" className="mb-4 border-blue-500/20 bg-blue-500/10">
            <Lock className="mr-2 size-4 text-blue-400" />
            <span className="text-blue-400">Sign In</span>
          </Badge>
          <h1 className="text-3xl font-bold mb-2 bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
            Welcome Back
          </h1>
          <p className="text-muted-foreground">
            Sign in to access your Zehan X Technologies account
          </p>
        </div>

        <div className="border border-border/50 rounded-lg p-6 bg-card/50 backdrop-blur-sm">
          <SignIn 
            appearance={{
              elements: {
                formButtonPrimary: 
                  "bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white border-0",
                card: "bg-transparent shadow-none",
                headerTitle: "text-foreground",
                headerSubtitle: "text-muted-foreground",
                socialButtonsBlockButton: 
                  "border-border/50 hover:bg-accent/50 text-foreground",
                formFieldInput: 
                  "bg-background border-border/50 focus:border-blue-500/50 text-foreground",
                formFieldLabel: "text-foreground",
                identityPreviewText: "text-muted-foreground",
                formResendCodeLink: "text-blue-400 hover:text-blue-300",
                footerActionLink: "text-blue-400 hover:text-blue-300"
              }
            }}
          />
        </div>

        <div className="mt-8 text-center">
          <Link 
            href="/" 
            className="text-muted-foreground hover:text-blue-400 text-sm transition-colors duration-200"
          >
            ← Back to Home
          </Link>
        </div>
      </div>
    </Section>
  );
}