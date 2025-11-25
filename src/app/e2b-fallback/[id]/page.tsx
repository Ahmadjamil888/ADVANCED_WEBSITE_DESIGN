"use client";

import React from "react";
import { useParams, useRouter } from "next/navigation";

export default function E2BFallbackPage() {
  const params = useParams();
  const router = useRouter();
  const id = (params?.id as string) || "unknown";

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: '#0b1220',
      color: 'white',
      padding: '24px'
    }}>
      <div style={{
        maxWidth: 720,
        width: '100%',
        background: 'rgba(255,255,255,0.04)',
        border: '1px solid rgba(255,255,255,0.08)',
        borderRadius: 16,
        padding: 24
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
          <div style={{
            width: 36,
            height: 36,
            borderRadius: 8,
            background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontWeight: 700
          }}>AI</div>
          <h1 style={{ margin: 0, fontSize: 20, fontWeight: 700 }}>Live model not yet deployed</h1>
        </div>

        <p style={{ color: 'rgba(255,255,255,0.75)', margin: '8px 0 16px 0' }}>
          The requested subdomain appears to be a placeholder: <code style={{ color: '#93c5fd' }}>fallback-{id}.zehanxtech.com</code>.
          This link is used when the E2B sandbox deployment didn’t complete.
        </p>

        <div style={{
          background: 'rgba(255,255,255,0.04)',
          border: '1px dashed rgba(255,255,255,0.12)',
          borderRadius: 12,
          padding: 16,
          marginBottom: 16
        }}>
          <p style={{ margin: 0, color: 'rgba(255,255,255,0.85)' }}>
            To get a working live URL:
          </p>
          <ul style={{ margin: '8px 0 0 18px', color: 'rgba(255,255,255,0.75)' }}>
            <li>Set a valid <code>E2B_API_KEY</code> in your environment.</li>
            <li>Run the pipeline again so it returns a real sandbox URL like <code>https://&lt;sandboxId&gt;.e2b.dev</code>.</li>
          </ul>
        </div>

        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          <button
            onClick={() => router.push('/ai-workspace')}
            style={{
              background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
              border: 'none',
              color: 'white',
              padding: '10px 16px',
              borderRadius: 8,
              cursor: 'pointer',
              fontWeight: 600
            }}
          >
            Go to AI Workspace
          </button>
          <button
            onClick={() => router.back()}
            style={{
              background: 'rgba(255,255,255,0.06)',
              border: '1px solid rgba(255,255,255,0.12)',
              color: 'white',
              padding: '10px 16px',
              borderRadius: 8,
              cursor: 'pointer',
              fontWeight: 600
            }}
          >
            Go back
          </button>
        </div>
      </div>
    </div>
  );
}
