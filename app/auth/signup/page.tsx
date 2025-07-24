import { SignUp } from '@clerk/nextjs';
import { Section } from "@/components/ui/section";
import { Badge } from "@/components/ui/badge";
import { UserPlus } from "lucide-react";
import Link from "next/link";

export default function SignUpPage() {
  return (
    <Section className="min-h-screen flex items-center justify-center py-12">
      <div className="max-w-md w-full mx-auto">
        <div className="text-center mb-8">
          <Badge variant="outline" className="mb-4 border-green-500/20 bg-green-500/10">
            <UserPlus className="mr-2 size-4 text-green-400" />
            <span className="text-green-400">Sign Up</span>
          </Badge>
          <h1 className="text-3xl font-bold mb-2 bg-gradient-to-r from-green-400 to-blue-400 bg-clip-text text-transparent">
            Join Zehan X
          </h1>
          <p className="text-muted-foreground">
            Create your account to get started with our AI solutions
          </p>
        </div>

        <div className="border border-border/50 rounded-lg p-6 bg-card/50 backdrop-blur-sm">
          <SignUp 
            appearance={{
              elements: {
                formButtonPrimary: 
                  "bg-gradient-to-r from-green-600 to-blue-600 hover:from-green-700 hover:to-blue-700 text-white border-0",
                card: "bg-transparent shadow-none",
                headerTitle: "text-foreground",
                headerSubtitle: "text-muted-foreground",
                socialButtonsBlockButton: 
                  "border-border/50 hover:bg-accent/50 text-foreground",
                formFieldInput: 
                  "bg-background border-border/50 focus:border-green-500/50 text-foreground",
                formFieldLabel: "text-foreground",
                identityPreviewText: "text-muted-foreground",
                formResendCodeLink: "text-green-400 hover:text-green-300",
                footerActionLink: "text-green-400 hover:text-green-300"
              }
            }}
          />
        </div>

        <div className="mt-8 text-center">
          <Link 
            href="/" 
            className="text-muted-foreground hover:text-green-400 text-sm transition-colors duration-200"
          >
            ← Back to Home
          </Link>
        </div>
      </div>
    </Section>
  );
}