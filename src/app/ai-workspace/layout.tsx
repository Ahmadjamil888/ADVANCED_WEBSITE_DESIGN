import './animations.css';

export default function AIWorkspaceLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div style={{
      minHeight: '100vh',
      backgroundColor: '#030712',
      overflowY: 'auto'
    }}>
      {children}
    </div>
  );
}