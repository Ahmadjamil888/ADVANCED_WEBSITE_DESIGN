"use client";

import React, { useState, useEffect } from 'react';
import { supabase } from '@/lib/supabase';
import { useAuth } from '@/contexts/AuthContext';

interface ApiKey {
    id: string;
    name: string;
    description?: string;
    key_preview: string;
    created_at: string;
    is_active: boolean;
    max_daily_requests: number;
    max_monthly_requests: number;
    total_usage: number;
    daily_usage: number;
    monthly_usage: number;
    last_used_at?: string;
    key?: string; // Only present when first generated
}

interface ApiKeysModalProps {
    isOpen: boolean;
    onClose: () => void;
}

export default function ApiKeysModal({ isOpen, onClose }: ApiKeysModalProps) {
    const { user } = useAuth();
    const [apiKeys, setApiKeys] = useState<ApiKey[]>([]);
    const [loading, setLoading] = useState(false);
    const [showCreateForm, setShowCreateForm] = useState(false);
    const [newKeyName, setNewKeyName] = useState('');
    const [newKeyDescription, setNewKeyDescription] = useState('');
    const [generatedKey, setGeneratedKey] = useState('');
    const [copySuccess, setCopySuccess] = useState(false);

    const inngestCallbackUrl = `${window.location.origin}/api/inngest`;

    useEffect(() => {
        if (isOpen && user) {
            loadApiKeys();
        }
    }, [isOpen, user]);

    const getAuthToken = async () => {
        if (!supabase) return null;
        const { data: { session } } = await supabase.auth.getSession();
        return session?.access_token || null;
    };

    const loadApiKeys = async () => {
        if (!user) return;

        setLoading(true);
        try {
            const token = await getAuthToken();
            if (!token) {
                console.error('No auth token available');
                return;
            }

            const response = await fetch('/api/api-keys', {
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json'
                }
            });

            if (response.ok) {
                const data = await response.json();
                setApiKeys(data.apiKeys || []);
            } else {
                console.error('Failed to load API keys:', response.statusText);
            }
        } catch (err) {
            console.error('Error loading API keys:', err);
        } finally {
            setLoading(false);
        }
    };

    const createApiKey = async () => {
        if (!user || !newKeyName.trim()) return;

        setLoading(true);
        try {
            const token = await getAuthToken();
            if (!token) {
                console.error('No auth token available');
                return;
            }

            const response = await fetch('/api/api-keys', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    name: newKeyName.trim(),
                    description: newKeyDescription.trim() || null
                })
            });

            if (response.ok) {
                const data = await response.json();
                if (data.success && data.apiKey) {
                    setGeneratedKey(data.apiKey.key);
                    setNewKeyName('');
                    setNewKeyDescription('');
                    setShowCreateForm(false);
                    loadApiKeys();
                }
            } else {
                console.error('Failed to create API key:', response.statusText);
            }
        } catch (err) {
            console.error('Error creating API key:', err);
        } finally {
            setLoading(false);
        }
    };

    const revokeApiKey = async (keyId: string) => {
        if (!confirm('Are you sure you want to revoke this API key? This action cannot be undone.')) {
            return;
        }

        try {
            const token = await getAuthToken();
            if (!token) {
                console.error('No auth token available');
                return;
            }

            const response = await fetch(`/api/api-keys?id=${keyId}`, {
                method: 'DELETE',
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json'
                }
            });

            if (response.ok) {
                loadApiKeys();
            } else {
                console.error('Failed to revoke API key:', response.statusText);
            }
        } catch (err) {
            console.error('Error revoking API key:', err);
        }
    };

    const copyToClipboard = async (text: string) => {
        try {
            await navigator.clipboard.writeText(text);
            setCopySuccess(true);
            setTimeout(() => setCopySuccess(false), 2000);
        } catch (err) {
            console.error('Failed to copy to clipboard:', err);
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = text;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            setCopySuccess(true);
            setTimeout(() => setCopySuccess(false), 2000);
        }
    };

    if (!isOpen) return null;

    return (
        <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000
        }}>
            <div style={{
                backgroundColor: 'white',
                borderRadius: '12px',
                padding: '24px',
                maxWidth: '800px',
                width: '90%',
                maxHeight: '80vh',
                overflow: 'auto'
            }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
                    <h2 style={{ fontSize: '24px', fontWeight: '600', color: '#111827' }}>API Keys Management</h2>
                    <button
                        onClick={onClose}
                        style={{
                            padding: '8px',
                            background: 'transparent',
                            border: 'none',
                            cursor: 'pointer',
                            borderRadius: '6px'
                        }}
                    >
                        <svg width="24" height="24" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </button>
                </div>

                {/* Callback URL Section */}
                <div style={{
                    backgroundColor: '#f9fafb',
                    border: '1px solid #e5e7eb',
                    borderRadius: '8px',
                    padding: '16px',
                    marginBottom: '24px'
                }}>
                    <h3 style={{ fontSize: '16px', fontWeight: '500', marginBottom: '8px' }}>Inngest Callback URL</h3>
                    <code style={{
                        backgroundColor: '#f3f4f6',
                        padding: '8px 12px',
                        borderRadius: '6px',
                        fontSize: '14px',
                        display: 'block',
                        wordBreak: 'break-all'
                    }}>
                        {inngestCallbackUrl}
                    </code>
                    <p style={{ fontSize: '14px', color: '#6b7280', marginTop: '8px' }}>
                        Use this URL as your Inngest callback endpoint for webhook integrations.
                    </p>
                </div>

                {/* Generated Key Display */}
                {generatedKey && (
                    <div style={{
                        backgroundColor: '#ecfdf5',
                        border: '2px solid #10b981',
                        borderRadius: '12px',
                        padding: '20px',
                        marginBottom: '24px',
                        boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                    }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
                            <svg width="20" height="20" fill="none" stroke="#10b981" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                            <h3 style={{ fontSize: '18px', fontWeight: '600', color: '#065f46', margin: 0 }}>
                                🎉 API Key Generated Successfully!
                            </h3>
                        </div>
                        
                        <div style={{
                            backgroundColor: '#f0fdf4',
                            border: '1px solid #bbf7d0',
                            borderRadius: '8px',
                            padding: '12px',
                            marginBottom: '12px'
                        }}>
                            <code style={{
                                fontSize: '14px',
                                fontFamily: 'Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
                                display: 'block',
                                wordBreak: 'break-all',
                                color: '#065f46',
                                lineHeight: '1.4'
                            }}>
                                {generatedKey}
                            </code>
                        </div>
                        
                        <p style={{ fontSize: '14px', color: '#065f46', marginBottom: '16px', fontWeight: '500' }}>
                            ⚠️ Please copy this key now. You won't be able to see it again for security reasons.
                        </p>
                        
                        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                            <button
                                onClick={() => copyToClipboard(generatedKey)}
                                style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '6px',
                                    padding: '8px 16px',
                                    backgroundColor: copySuccess ? '#059669' : '#10b981',
                                    color: 'white',
                                    border: 'none',
                                    borderRadius: '8px',
                                    cursor: 'pointer',
                                    fontSize: '14px',
                                    fontWeight: '500',
                                    transition: 'all 0.2s'
                                }}
                            >
                                {copySuccess ? (
                                    <>
                                        <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                                        </svg>
                                        Copied!
                                    </>
                                ) : (
                                    <>
                                        <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                                        </svg>
                                        Copy to Clipboard
                                    </>
                                )}
                            </button>
                            <button
                                onClick={() => setGeneratedKey('')}
                                style={{
                                    padding: '8px 16px',
                                    backgroundColor: 'transparent',
                                    color: '#065f46',
                                    border: '1px solid #10b981',
                                    borderRadius: '8px',
                                    cursor: 'pointer',
                                    fontSize: '14px',
                                    fontWeight: '500',
                                    transition: 'all 0.2s'
                                }}
                            >
                                Dismiss
                            </button>
                        </div>
                    </div>
                )}

                {/* Create New API Key */}
                <div style={{ marginBottom: '24px' }}>
                    {!showCreateForm ? (
                        <button
                            onClick={() => setShowCreateForm(true)}
                            style={{
                                padding: '12px 24px',
                                backgroundColor: '#2563eb',
                                color: 'white',
                                border: 'none',
                                borderRadius: '8px',
                                cursor: 'pointer',
                                fontSize: '16px',
                                fontWeight: '500'
                            }}
                        >
                            Create New API Key
                        </button>
                    ) : (
                        <div style={{
                            border: '1px solid #e5e7eb',
                            borderRadius: '8px',
                            padding: '16px'
                        }}>
                            <h3 style={{ fontSize: '16px', fontWeight: '500', marginBottom: '16px' }}>Create New API Key</h3>

                            <div style={{ marginBottom: '16px' }}>
                                <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', marginBottom: '4px' }}>
                                    Key Name *
                                </label>
                                <input
                                    type="text"
                                    value={newKeyName}
                                    onChange={(e) => setNewKeyName(e.target.value)}
                                    placeholder="e.g., Production API Key"
                                    style={{
                                        width: '100%',
                                        padding: '8px 12px',
                                        border: '1px solid #d1d5db',
                                        borderRadius: '6px',
                                        fontSize: '14px'
                                    }}
                                />
                            </div>

                            <div style={{ marginBottom: '16px' }}>
                                <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', marginBottom: '4px' }}>
                                    Description (Optional)
                                </label>
                                <textarea
                                    value={newKeyDescription}
                                    onChange={(e) => setNewKeyDescription(e.target.value)}
                                    placeholder="Describe what this API key will be used for..."
                                    rows={3}
                                    style={{
                                        width: '100%',
                                        padding: '8px 12px',
                                        border: '1px solid #d1d5db',
                                        borderRadius: '6px',
                                        fontSize: '14px',
                                        resize: 'vertical'
                                    }}
                                />
                            </div>

                            <div style={{ display: 'flex', gap: '8px' }}>
                                <button
                                    onClick={createApiKey}
                                    disabled={!newKeyName.trim() || loading}
                                    style={{
                                        padding: '8px 16px',
                                        backgroundColor: newKeyName.trim() ? '#10b981' : '#9ca3af',
                                        color: 'white',
                                        border: 'none',
                                        borderRadius: '6px',
                                        cursor: newKeyName.trim() ? 'pointer' : 'not-allowed',
                                        fontSize: '14px'
                                    }}
                                >
                                    {loading ? 'Creating...' : 'Create Key'}
                                </button>
                                <button
                                    onClick={() => {
                                        setShowCreateForm(false);
                                        setNewKeyName('');
                                        setNewKeyDescription('');
                                    }}
                                    style={{
                                        padding: '8px 16px',
                                        backgroundColor: 'transparent',
                                        color: '#6b7280',
                                        border: '1px solid #d1d5db',
                                        borderRadius: '6px',
                                        cursor: 'pointer',
                                        fontSize: '14px'
                                    }}
                                >
                                    Cancel
                                </button>
                            </div>
                        </div>
                    )}
                </div>

                {/* API Keys List */}
                <div>
                    <h3 style={{ fontSize: '18px', fontWeight: '500', marginBottom: '16px' }}>Your API Keys</h3>

                    {loading ? (
                        <div style={{ textAlign: 'center', padding: '32px' }}>
                            <div style={{
                                width: '32px',
                                height: '32px',
                                border: '3px solid #f3f4f6',
                                borderTop: '3px solid #2563eb',
                                borderRadius: '50%',
                                animation: 'spin 1s linear infinite',
                                margin: '0 auto'
                            }}></div>
                            <p style={{ marginTop: '16px', color: '#6b7280' }}>Loading API keys...</p>
                        </div>
                    ) : apiKeys.length === 0 ? (
                        <div style={{
                            textAlign: 'center',
                            padding: '32px',
                            backgroundColor: '#f9fafb',
                            borderRadius: '8px',
                            border: '1px solid #e5e7eb'
                        }}>
                            <p style={{ color: '#6b7280', fontSize: '16px' }}>No API keys found</p>
                            <p style={{ color: '#9ca3af', fontSize: '14px', marginTop: '4px' }}>
                                Create your first API key to get started
                            </p>
                        </div>
                    ) : (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                            {apiKeys.map((key) => (
                                <div
                                    key={key.id}
                                    style={{
                                        border: '1px solid #e5e7eb',
                                        borderRadius: '12px',
                                        padding: '20px',
                                        backgroundColor: key.is_active ? 'white' : '#f9fafb',
                                        boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1)'
                                    }}
                                >
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '12px' }}>
                                        <div style={{ flex: 1 }}>
                                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                                                <svg width="16" height="16" fill="none" stroke="#6b7280" viewBox="0 0 24 24">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z" />
                                                </svg>
                                                <h4 style={{ fontSize: '16px', fontWeight: '600', margin: 0, color: '#111827' }}>
                                                    {key.name}
                                                </h4>
                                            </div>
                                            {key.description && (
                                                <p style={{ fontSize: '14px', color: '#6b7280', marginBottom: '8px', margin: 0 }}>
                                                    {key.description}
                                                </p>
                                            )}
                                        </div>
                                        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                                            <span style={{
                                                padding: '4px 12px',
                                                borderRadius: '16px',
                                                fontSize: '12px',
                                                fontWeight: '600',
                                                backgroundColor: key.is_active ? '#ecfdf5' : '#f3f4f6',
                                                color: key.is_active ? '#065f46' : '#6b7280',
                                                border: `1px solid ${key.is_active ? '#bbf7d0' : '#d1d5db'}`
                                            }}>
                                                {key.is_active ? '✅ Active' : '❌ Inactive'}
                                            </span>
                                        </div>
                                    </div>
                                    
                                    <div style={{
                                        backgroundColor: '#f8fafc',
                                        border: '1px solid #e2e8f0',
                                        borderRadius: '8px',
                                        padding: '12px',
                                        marginBottom: '12px'
                                    }}>
                                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                                            <code style={{
                                                fontSize: '14px',
                                                fontFamily: 'Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
                                                color: '#475569',
                                                flex: 1
                                            }}>
                                                {key.key_preview}
                                            </code>
                                            <button
                                                onClick={() => copyToClipboard(key.key_preview)}
                                                style={{
                                                    padding: '4px',
                                                    backgroundColor: 'transparent',
                                                    border: 'none',
                                                    cursor: 'pointer',
                                                    borderRadius: '4px',
                                                    color: '#6b7280'
                                                }}
                                                title="Copy key preview"
                                            >
                                                <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                                                </svg>
                                            </button>
                                        </div>
                                    </div>

                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '12px', color: '#6b7280', marginBottom: '12px' }}>
                                        <div style={{ display: 'flex', gap: '16px' }}>
                                            <span>📅 Created: {new Date(key.created_at).toLocaleDateString()}</span>
                                            {key.last_used_at && (
                                                <span>🕒 Last used: {new Date(key.last_used_at).toLocaleDateString()}</span>
                                            )}
                                        </div>
                                    </div>

                                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: '12px', marginBottom: '16px' }}>
                                        <div style={{ textAlign: 'center', padding: '8px', backgroundColor: '#f1f5f9', borderRadius: '6px' }}>
                                            <div style={{ fontSize: '18px', fontWeight: '600', color: '#1e293b' }}>{key.total_usage}</div>
                                            <div style={{ fontSize: '11px', color: '#64748b' }}>Total Requests</div>
                                        </div>
                                        <div style={{ textAlign: 'center', padding: '8px', backgroundColor: '#fef3c7', borderRadius: '6px' }}>
                                            <div style={{ fontSize: '18px', fontWeight: '600', color: '#92400e' }}>{key.daily_usage}</div>
                                            <div style={{ fontSize: '11px', color: '#92400e' }}>Today</div>
                                        </div>
                                        <div style={{ textAlign: 'center', padding: '8px', backgroundColor: '#e0f2fe', borderRadius: '6px' }}>
                                            <div style={{ fontSize: '18px', fontWeight: '600', color: '#0369a1' }}>{key.monthly_usage}</div>
                                            <div style={{ fontSize: '11px', color: '#0369a1' }}>This Month</div>
                                        </div>
                                    </div>

                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                        <div style={{ fontSize: '12px', color: '#6b7280' }}>
                                            Limits: {key.max_daily_requests}/day • {key.max_monthly_requests}/month
                                        </div>
                                        <button
                                            onClick={() => revokeApiKey(key.id)}
                                            style={{
                                                display: 'flex',
                                                alignItems: 'center',
                                                gap: '4px',
                                                padding: '6px 12px',
                                                backgroundColor: '#fee2e2',
                                                color: '#dc2626',
                                                border: '1px solid #fecaca',
                                                borderRadius: '6px',
                                                cursor: 'pointer',
                                                fontSize: '12px',
                                                fontWeight: '500',
                                                transition: 'all 0.2s'
                                            }}
                                            onMouseOver={(e) => {
                                                e.currentTarget.style.backgroundColor = '#fecaca';
                                            }}
                                            onMouseOut={(e) => {
                                                e.currentTarget.style.backgroundColor = '#fee2e2';
                                            }}
                                        >
                                            <svg width="14" height="14" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                            </svg>
                                            Revoke
                                        </button>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                {/* Documentation Section */}
                <div style={{
                    marginTop: '32px',
                    padding: '16px',
                    backgroundColor: '#f9fafb',
                    borderRadius: '8px',
                    border: '1px solid #e5e7eb'
                }}>
                    <h3 style={{ fontSize: '16px', fontWeight: '500', marginBottom: '12px' }}>API Documentation</h3>
                    <div style={{ fontSize: '14px', color: '#6b7280', lineHeight: '1.5' }}>
                        <p style={{ marginBottom: '8px' }}>
                            <strong>Authentication:</strong> Include your API key in the Authorization header:
                        </p>
                        <code style={{
                            display: 'block',
                            backgroundColor: '#f3f4f6',
                            padding: '8px 12px',
                            borderRadius: '4px',
                            marginBottom: '12px',
                            fontSize: '12px'
                        }}>
                            Authorization: Bearer YOUR_API_KEY
                        </code>
                        <p style={{ marginBottom: '8px' }}>
                            <strong>Base URL:</strong> {window.location.origin}/api
                        </p>
                        <p>
                            <strong>Rate Limits:</strong> 1,000 requests per day, 10,000 requests per month by default.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}