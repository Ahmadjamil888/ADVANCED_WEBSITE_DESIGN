"use client";

import React, { useState, useEffect } from 'react';
import { supabase } from '@/lib/supabase';
import { useAuth } from '@/contexts/AuthContext';

interface ApiKey {
    id: string;
    name: string;
    key_preview: string;
    created_at: string;
    is_active: boolean;
    max_daily_requests: number;
    max_monthly_requests: number;
    total_usage: number;
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

    const inngestCallbackUrl = `${window.location.origin}/api/inngest`;

    useEffect(() => {
        if (isOpen && user) {
            loadApiKeys();
        }
    }, [isOpen, user]);

    const loadApiKeys = async () => {
        if (!user || !supabase) return;

        setLoading(true);
        try {
            const { data, error } = await supabase
                .from('api_keys')
                .select('*')
                .eq('user_id', user.id)
                .eq('is_revoked', false)
                .order('created_at', { ascending: false });

            if (data && !error) {
                setApiKeys(data);
            }
        } catch (err) {
            console.error('Error loading API keys:', err);
        } finally {
            setLoading(false);
        }
    };

    const generateApiKey = () => {
        const prefix = 'zx_';
        const randomPart = Array.from(crypto.getRandomValues(new Uint8Array(32)))
            .map(b => b.toString(16).padStart(2, '0'))
            .join('');
        return prefix + randomPart;
    };

    const createApiKey = async () => {
        if (!user || !supabase || !newKeyName.trim()) return;

        setLoading(true);
        try {
            const apiKey = generateApiKey();
            const keyPreview = apiKey.substring(0, 12) + '...';

            // Hash the key for storage (in production, use proper hashing)
            const keyHash = btoa(apiKey);

            const { data, error } = await supabase
                .from('api_keys')
                .insert({
                    user_id: user.id,
                    key_hash: keyHash,
                    key_preview: keyPreview,
                    name: newKeyName.trim(),
                    description: newKeyDescription.trim() || null,
                    max_daily_requests: 1000,
                    max_monthly_requests: 10000,
                    max_tokens_per_request: 2048
                })
                .select()
                .single();

            if (data && !error) {
                setGeneratedKey(apiKey);
                setNewKeyName('');
                setNewKeyDescription('');
                setShowCreateForm(false);
                loadApiKeys();
            }
        } catch (err) {
            console.error('Error creating API key:', err);
        } finally {
            setLoading(false);
        }
    };

    const revokeApiKey = async (keyId: string) => {
        if (!supabase || !confirm('Are you sure you want to revoke this API key? This action cannot be undone.')) {
            return;
        }

        try {
            const { error } = await supabase
                .from('api_keys')
                .update({ is_revoked: true, is_active: false })
                .eq('id', keyId);

            if (!error) {
                loadApiKeys();
            }
        } catch (err) {
            console.error('Error revoking API key:', err);
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
                        border: '1px solid #10b981',
                        borderRadius: '8px',
                        padding: '16px',
                        marginBottom: '24px'
                    }}>
                        <h3 style={{ fontSize: '16px', fontWeight: '500', color: '#065f46', marginBottom: '8px' }}>
                            API Key Generated Successfully
                        </h3>
                        <code style={{
                            backgroundColor: '#f0fdf4',
                            padding: '8px 12px',
                            borderRadius: '6px',
                            fontSize: '14px',
                            display: 'block',
                            wordBreak: 'break-all',
                            color: '#065f46'
                        }}>
                            {generatedKey}
                        </code>
                        <p style={{ fontSize: '14px', color: '#065f46', marginTop: '8px' }}>
                            Please copy this key now. You won't be able to see it again for security reasons.
                        </p>
                        <button
                            onClick={() => {
                                navigator.clipboard.writeText(generatedKey);
                                alert('API key copied to clipboard!');
                            }}
                            style={{
                                marginTop: '8px',
                                padding: '6px 12px',
                                backgroundColor: '#10b981',
                                color: 'white',
                                border: 'none',
                                borderRadius: '6px',
                                cursor: 'pointer',
                                fontSize: '14px'
                            }}
                        >
                            Copy to Clipboard
                        </button>
                        <button
                            onClick={() => setGeneratedKey('')}
                            style={{
                                marginTop: '8px',
                                marginLeft: '8px',
                                padding: '6px 12px',
                                backgroundColor: '#6b7280',
                                color: 'white',
                                border: 'none',
                                borderRadius: '6px',
                                cursor: 'pointer',
                                fontSize: '14px'
                            }}
                        >
                            Dismiss
                        </button>
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
                                        borderRadius: '8px',
                                        padding: '16px',
                                        backgroundColor: key.is_active ? 'white' : '#f9fafb'
                                    }}
                                >
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                                        <div style={{ flex: 1 }}>
                                            <h4 style={{ fontSize: '16px', fontWeight: '500', marginBottom: '4px' }}>
                                                {key.name}
                                            </h4>
                                            <p style={{ fontSize: '14px', color: '#6b7280', marginBottom: '8px' }}>
                                                {key.key_preview}
                                            </p>
                                            <div style={{ display: 'flex', gap: '16px', fontSize: '12px', color: '#9ca3af' }}>
                                                <span>Created: {new Date(key.created_at).toLocaleDateString()}</span>
                                                <span>Usage: {key.total_usage} requests</span>
                                                <span>Daily Limit: {key.max_daily_requests}</span>
                                                <span>Monthly Limit: {key.max_monthly_requests}</span>
                                            </div>
                                        </div>
                                        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                                            <span style={{
                                                padding: '4px 8px',
                                                borderRadius: '12px',
                                                fontSize: '12px',
                                                fontWeight: '500',
                                                backgroundColor: key.is_active ? '#ecfdf5' : '#f3f4f6',
                                                color: key.is_active ? '#065f46' : '#6b7280'
                                            }}>
                                                {key.is_active ? 'Active' : 'Inactive'}
                                            </span>
                                            <button
                                                onClick={() => revokeApiKey(key.id)}
                                                style={{
                                                    padding: '6px 12px',
                                                    backgroundColor: '#dc2626',
                                                    color: 'white',
                                                    border: 'none',
                                                    borderRadius: '6px',
                                                    cursor: 'pointer',
                                                    fontSize: '12px'
                                                }}
                                            >
                                                Revoke
                                            </button>
                                        </div>
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