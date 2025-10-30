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
            font-family: Arial, Helvetica, sans-serif;
            background-color: #f3f4f6;
            color: #111827;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
          }
          
          .login-container {
            max-width: 1200px;
            margin: 2.5rem;
            background: white;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            border-radius: 0.5rem;
            display: flex;
            justify-content: center;
            flex: 1;
          }
          
          .login-form-section {
            width: 50%;
            max-width: 416px;
            padding: 1.5rem 3rem;
          }
          
          .logo-container {
            text-align: center;
            margin-bottom: 3rem;
          }
          
          .logo {
            width: 8rem;
            height: 8rem;
            border-radius: 0.5rem;
            margin: 0 auto;
          }
          
          .form-container {
            display: flex;
            flex-direction: column;
            align-items: center;
          }
          
          .form-title {
            font-size: 1.875rem;
            font-weight: 800;
            margin-bottom: 2rem;
          }
          
          .oauth-button {
            width: 100%;
            max-width: 20rem;
            font-weight: 700;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            border-radius: 0.5rem;
            padding: 0.75rem;
            background-color: #e0e7ff;
            color: #374151;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s ease;
            border: none;
            cursor: pointer;
            margin-bottom: 1.25rem;
          }
          
          .oauth-button:hover {
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
          }
          
          .oauth-button:focus {
            outline: none;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
          }
          
          .oauth-icon {
            background: white;
            padding: 0.5rem;
            border-radius: 50%;
            margin-right: 1rem;
          }
          
          .divider {
            margin: 3rem 0;
            border-bottom: 1px solid #e5e7eb;
            text-align: center;
            position: relative;
          }
          
          .divider-text {
            line-height: 1;
            padding: 0 0.5rem;
            display: inline-block;
            font-size: 0.875rem;
            color: #6b7280;
            font-weight: 500;
            background: white;
            transform: translateY(50%);
          }
          
          .email-form {
            margin: 0 auto;
            max-width: 20rem;
            width: 100%;
          }
          
          .form-input {
            width: 100%;
            padding: 1rem 2rem;
            border-radius: 0.5rem;
            font-weight: 500;
            background-color: #f3f4f6;
            border: 1px solid #e5e7eb;
            color: #374151;
            font-size: 0.875rem;
            margin-bottom: 1.25rem;
          }
          
          .form-input:focus {
            outline: none;
            border-color: #9ca3af;
            background-color: white;
          }
          
          .form-input::placeholder {
            color: #6b7280;
          }
          
          .submit-button {
            margin-top: 1.25rem;
            font-weight: 600;
            background-color: #6366f1;
            color: #f9fafb;
            width: 100%;
            padding: 1rem;
            border-radius: 0.5rem;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            border: none;
            cursor: pointer;
          }
          
          .submit-button:hover {
            background-color: #4f46e5;
          }
          
          .submit-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
          }
          
          .submit-button svg {
            width: 1.5rem;
            height: 1.5rem;
            margin-left: -0.5rem;
            margin-right: 0.75rem;
          }
          
          .toggle-button {
            color: #6366f1;
            font-size: 0.875rem;
            background: none;
            border: none;
            cursor: pointer;
            margin-top: 1.5rem;
          }
          
          .toggle-button:hover {
            color: #4f46e5;
          }
          
          .terms-text {
            margin-top: 1.5rem;
            font-size: 0.75rem;
            color: #6b7280;
            text-align: center;
          }
          
          .terms-link {
            border-bottom: 1px dotted #6b7280;
            text-decoration: none;
            color: inherit;
          }
          
          .background-section {
            flex: 1;
            background-color: #e0e7ff;
            text-align: center;
            display: none;
          }
          
          .background-image {
            margin: 3rem 4rem;
            width: calc(100% - 8rem);
            background-size: contain;
            background-position: center;
            background-repeat: no-repeat;
            height: calc(100% - 6rem);
          }
          
          @media (min-width: 1024px) {
            .background-section {
              display: flex;
            }
            
            .login-form-section {
              width: 41.666667%;
            }
          }
          
          @media (max-width: 640px) {
            .login-container {
              margin: 0;
              border-radius: 0;
            }
            
            .login-form-section {
              width: 100%;
              padding: 1.5rem;
            }
          }
        `}</style>
      </head>
      <body>{children}</body>
    </html>
  )
}