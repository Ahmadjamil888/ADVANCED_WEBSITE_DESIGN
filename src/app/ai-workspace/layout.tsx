import { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'AI Workspace - zehanxtech',
  description: 'Generate, train, and deploy custom AI models',
}

export default function AIWorkspaceLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div 
      className="fixed inset-0 bg-white overflow-hidden"
      style={{ 
        zIndex: 9999,
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
      }}
    >
      <style dangerouslySetInnerHTML={{
        __html: `
          /* Override parent layout styles for AI workspace */
          body {
            margin: 0 !important;
            padding: 0 !important;
            overflow: hidden !important;
          }
          
          /* Hide navbar and footer from main layout */
          header, nav, footer, .header, .footer, .navbar {
            display: none !important;
            visibility: hidden !important;
          }
        `
      }} />
      {children}
    </div>
  )
}