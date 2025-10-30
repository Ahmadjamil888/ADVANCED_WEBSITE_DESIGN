export default function LoginLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <head>
        <style>{`
          * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
          }
          
          body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
          }
          
          .login-container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            overflow: hidden;
            width: 100%;
            max-width: 400px;
            padding: 40px;
            position: relative;
          }
          
          .login-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #667eea, #764ba2);
          }
          
          .logo-section {
            text-align: center;
            margin-bottom: 30px;
          }
          
          .logo {
            width: 80px;
            height: 80px;
            border-radius: 16px;
            margin: 0 auto 20px;
            display: block;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
          }
          
          .brand-name {
            font-size: 24px;
            font-weight: 700;
            color: #2d3748;
            margin-bottom: 8px;
          }
          
          .brand-tagline {
            font-size: 14px;
            color: #718096;
          }
          
          .form-title {
            font-size: 28px;
            font-weight: 700;
            color: #2d3748;
            text-align: center;
            margin-bottom: 30px;
          }
          
          .oauth-section {
            margin-bottom: 30px;
          }
          
          .oauth-button {
            width: 100%;
            padding: 14px 20px;
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            background: white;
            color: #4a5568;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            margin-bottom: 12px;
          }
          
          .oauth-button:hover {
            border-color: #cbd5e0;
            transform: translateY(-2px);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
          }
          
          .oauth-button:active {
            transform: translateY(0);
          }
          
          .oauth-icon {
            width: 20px;
            height: 20px;
          }
          
          .divider {
            position: relative;
            text-align: center;
            margin: 30px 0;
          }
          
          .divider::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 0;
            right: 0;
            height: 1px;
            background: #e2e8f0;
          }
          
          .divider-text {
            background: white;
            padding: 0 20px;
            color: #718096;
            font-size: 14px;
            font-weight: 500;
          }
          
          .form-group {
            margin-bottom: 20px;
          }
          
          .form-input {
            width: 100%;
            padding: 14px 16px;
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            font-size: 16px;
            color: #2d3748;
            background: #f7fafc;
            transition: all 0.3s ease;
          }
          
          .form-input:focus {
            outline: none;
            border-color: #667eea;
            background: white;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
          }
          
          .form-input::placeholder {
            color: #a0aec0;
          }
          
          .submit-button {
            width: 100%;
            padding: 16px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-bottom: 20px;
          }
          
          .submit-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
          }
          
          .submit-button:active {
            transform: translateY(0);
          }
          
          .submit-button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
          }
          
          .toggle-section {
            text-align: center;
            margin-bottom: 20px;
          }
          
          .toggle-button {
            background: none;
            border: none;
            color: #667eea;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            text-decoration: underline;
          }
          
          .toggle-button:hover {
            color: #764ba2;
          }
          
          .terms-section {
            text-align: center;
            font-size: 12px;
            color: #718096;
            line-height: 1.5;
          }
          
          .terms-link {
            color: #667eea;
            text-decoration: none;
          }
          
          .terms-link:hover {
            text-decoration: underline;
          }
          
          .loading-spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 2px solid #ffffff;
            border-radius: 50%;
            border-top-color: transparent;
            animation: spin 1s ease-in-out infinite;
            margin-right: 8px;
          }
          
          @keyframes spin {
            to { transform: rotate(360deg); }
          }
          
          @media (max-width: 480px) {
            .login-container {
              padding: 30px 20px;
              margin: 10px;
            }
            
            .form-title {
              font-size: 24px;
            }
          }
        `}</style>
      </head>
      <body>{children}</body>
    </html>
  )
}