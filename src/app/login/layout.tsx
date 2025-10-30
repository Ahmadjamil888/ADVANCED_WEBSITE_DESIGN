export default function LoginLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div style={{ 
      position: 'fixed', 
      top: 0, 
      left: 0, 
      width: '100vw', 
      height: '100vh', 
      zIndex: 9999,
      background: 'white'
    }}>
      {children}
    </div>
  )
}