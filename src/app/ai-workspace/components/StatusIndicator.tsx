'use client';

interface StatusIndicatorProps {
  message: string;
  step?: number;
  total?: number;
  type?: 'info' | 'success' | 'error' | 'warning';
}

export function StatusIndicator({ message, step, total, type = 'info' }: StatusIndicatorProps) {
  const colors = {
    info: 'bg-blue-500',
    success: 'bg-green-500',
    error: 'bg-red-500',
    warning: 'bg-yellow-500',
  };

  const icons = {
    info: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
    success: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
      </svg>
    ),
    error: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
      </svg>
    ),
    warning: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
      </svg>
    ),
  };

  return (
    <div className="flex gap-3 mb-4 animate-fade-in">
      <div className={`w-8 h-8 rounded-full ${colors[type]} flex items-center justify-center flex-shrink-0 text-white`}>
        {icons[type]}
      </div>
      <div className="flex-1">
        <div className="bg-gray-800 rounded-2xl rounded-tl-sm px-4 py-3">
          <div className="flex items-center justify-between mb-2">
            <p className="text-sm text-gray-300">{message}</p>
            {step && total && (
              <span className="text-xs text-gray-500 font-mono">
                {step}/{total}
              </span>
            )}
          </div>
          {step && total && (
            <div className="w-full bg-gray-700 rounded-full h-1.5 overflow-hidden">
              <div
                className="bg-blue-500 h-full transition-all duration-500 ease-out"
                style={{ width: `${(step / total) * 100}%` }}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
