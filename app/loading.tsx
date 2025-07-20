export default function Loading() {
  return (
    <div className="min-h-screen w-full flex items-center justify-center bg-background">
      <div className="flex flex-col items-center gap-4">
        <div className="relative">
          <div className="size-12 border-4 border-muted rounded-full animate-spin border-t-primary"></div>
        </div>
        <div className="text-lg font-bold tracking-wider text-white" style={{ 
          fontFamily: 'monospace, "Courier New", Courier',
          textShadow: '0 0 10px rgba(255,255,255,0.5)',
          letterSpacing: '0.2em'
        }}>
          ZEHAN X
        </div>
        <p className="text-muted-foreground text-sm">Loading...</p>
      </div>
    </div>
  );
}