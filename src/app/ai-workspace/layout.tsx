export default function AIWorkspaceLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" style={{ height: '100%', margin: 0, padding: 0 }}>
      <body style={{ height: '100%', margin: 0, padding: 0, overflow: 'hidden' }}>
        <div style={{ height: '100%', width: '100%' }}>
          {children}
        </div>
      </body>
    </html>
  );
}