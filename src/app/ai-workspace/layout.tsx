export default function AIWorkspaceLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <>
      <style jsx global>{`
        /* Override parent layout styles */
        body {
          margin: 0 !important;
          padding: 0 !important;
          overflow: hidden !important;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
        }
        
        /* Hide navbar and footer from main layout */
        header, nav, footer, .header, .footer, .navbar {
          display: none !important;
          visibility: hidden !important;
        }
        
        /* Ensure AI workspace takes full screen */
        .ai-workspace-container {
          position: fixed;
          top: 0;
          left: 0;
          width: 100vw;
          height: 100vh;
          background: #ffffff;
          z-index: 9999;
          overflow: hidden;
        }
      `}</style>
      
      <div className="ai-workspace-container">
        {children}
      </div>
    </>
  )
}