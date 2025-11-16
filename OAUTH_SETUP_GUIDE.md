# OAuth Setup Guide - Google & Apple Authentication

## Overview

This guide explains how to configure Google and Apple OAuth with Supabase for the AI Model Generator platform.

---

## Table of Contents

1. [Google OAuth Setup](#google-oauth-setup)
2. [Apple OAuth Setup](#apple-oauth-setup)
3. [Supabase Configuration](#supabase-configuration)
4. [Testing OAuth](#testing-oauth)
5. [Troubleshooting](#troubleshooting)

---

## Google OAuth Setup

### Step 1: Create a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click "Select a Project" → "New Project"
3. Enter project name: `ZehanxTech AI Model Generator`
4. Click "Create"

### Step 2: Enable Google+ API

1. In the left sidebar, click "APIs & Services" → "Library"
2. Search for "Google+ API"
3. Click on it and select "Enable"

### Step 3: Create OAuth 2.0 Credentials

1. Go to "APIs & Services" → "Credentials"
2. Click "Create Credentials" → "OAuth 2.0 Client IDs"
3. If prompted, configure the OAuth consent screen first:
   - Click "Configure Consent Screen"
   - Select "External"
   - Fill in required fields:
     - App name: `ZehanxTech`
     - User support email: `support@zehanxtech.com`
     - Developer contact: `admin@zehanxtech.com`
   - Add scopes: `email`, `profile`, `openid`
   - Click "Save and Continue"

### Step 4: Create OAuth Client ID

1. Back on Credentials page, click "Create Credentials" → "OAuth 2.0 Client IDs"
2. Select "Web application"
3. Add Authorized redirect URIs:
   ```
   http://localhost:3000/auth/callback
   https://zehanxtech.com/auth/callback
   https://[YOUR_SUPABASE_PROJECT].supabase.co/auth/v1/callback
   ```
4. Click "Create"
5. Copy the **Client ID** and **Client Secret**

### Step 5: Add to Supabase

1. Go to [Supabase Dashboard](https://app.supabase.com)
2. Select your project
3. Go to "Authentication" → "Providers"
4. Find "Google" and enable it
5. Paste your **Client ID** and **Client Secret**
6. Click "Save"

---

## Apple OAuth Setup

### Step 1: Apple Developer Account

1. Go to [Apple Developer](https://developer.apple.com/)
2. Sign in or create an account
3. Enroll in Apple Developer Program ($99/year)

### Step 2: Create App ID

1. Go to "Certificates, Identifiers & Profiles"
2. Click "Identifiers" → "App IDs"
3. Click the "+" button
4. Select "App IDs"
5. Fill in:
   - App Name: `ZehanxTech AI Model Generator`
   - Bundle ID: `com.zehanxtech.aimodelgenerator`
6. Under "Capabilities", check "Sign in with Apple"
7. Click "Continue" → "Register"

### Step 3: Create Service ID

1. Go to "Identifiers" → "Services IDs"
2. Click the "+" button
3. Fill in:
   - Name: `ZehanxTech Web Service`
   - Identifier: `com.zehanxtech.web`
4. Check "Sign in with Apple"
5. Click "Configure"
6. Add Domains and Return URLs:
   ```
   Primary Domain: zehanxtech.com
   Return URLs:
   - https://zehanxtech.com/auth/callback
   - https://[YOUR_SUPABASE_PROJECT].supabase.co/auth/v1/callback
   ```
7. Click "Save" → "Continue" → "Register"

### Step 4: Create Private Key

1. Go to "Keys" → "App Store Connect API"
2. Click the "+" button
3. Name: `ZehanxTech Sign in with Apple`
4. Check "Sign in with Apple"
5. Click "Configure"
6. Select the Service ID you created
7. Click "Save" → "Continue" → "Register"
8. Download the private key (save it securely)

### Step 5: Add to Supabase

1. Go to [Supabase Dashboard](https://app.supabase.com)
2. Select your project
3. Go to "Authentication" → "Providers"
4. Find "Apple" and enable it
5. Fill in:
   - **Client ID**: Your Service ID (e.g., `com.zehanxtech.web`)
   - **Team ID**: Your Apple Team ID (found in Apple Developer account)
   - **Key ID**: From the private key you downloaded
   - **Private Key**: Contents of the downloaded .p8 file
6. Click "Save"

---

## Supabase Configuration

### Redirect URLs

Make sure these redirect URLs are configured in your Supabase project:

1. Go to "Authentication" → "URL Configuration"
2. Add these URLs:
   - **Site URL**: `https://zehanxtech.com`
   - **Redirect URLs**:
     ```
     http://localhost:3000/auth/callback
     https://zehanxtech.com/auth/callback
     ```

### Email Templates (Optional)

1. Go to "Authentication" → "Email Templates"
2. Customize confirmation and reset email templates
3. Update sender email if needed

---

## Testing OAuth

### Local Testing

1. Start your development server:
   ```bash
   npm run dev
   ```

2. Go to `http://localhost:3000/login`

3. Click "Google" or "Apple" button

4. You should be redirected to the provider's login

5. After authentication, you'll be redirected to `/auth/callback`

6. Then redirected to `/ai-model-generator`

### Production Testing

1. Deploy to production
2. Go to `https://zehanxtech.com/login`
3. Test Google and Apple OAuth flows
4. Verify users are created in Supabase

---

## Implementation Details

### Login Page

The login page (`src/app/login/page.tsx`) includes:

```typescript
const handleGoogleAuth = async () => {
  const { error } = await supabase.auth.signInWithOAuth({
    provider: 'google',
    options: {
      redirectTo: `${window.location.origin}/auth/callback`,
      queryParams: {
        access_type: 'offline',
        prompt: 'consent',
      },
    }
  })
  if (error) alert(error.message)
}

const handleAppleAuth = async () => {
  const { error } = await supabase.auth.signInWithOAuth({
    provider: 'apple',
    options: {
      redirectTo: `${window.location.origin}/auth/callback`,
    }
  })
  if (error) alert(error.message)
}
```

### Auth Callback

The callback page (`src/app/auth/callback/page.tsx`) handles:

1. OAuth provider redirects
2. Session establishment
3. Redirect to dashboard or login

---

## Troubleshooting

### "Invalid redirect URI"

**Problem**: OAuth provider rejects redirect URL

**Solution**:
1. Verify redirect URL matches exactly in provider settings
2. Include protocol (http/https)
3. No trailing slashes
4. Check Supabase URL Configuration

### "Client ID not found"

**Problem**: OAuth fails with client ID error

**Solution**:
1. Verify Client ID is correctly pasted in Supabase
2. Check provider settings for typos
3. Ensure provider is enabled in Supabase

### "User cancelled login"

**Problem**: User closes OAuth dialog

**Solution**:
- This is normal behavior
- User is redirected back to login page
- No error is thrown

### "Email already exists"

**Problem**: User tries to sign up with existing email

**Solution**:
1. User should use "Sign In" instead
2. Or use different email
3. Supabase handles this automatically

### "Apple OAuth not working on Android"

**Problem**: Apple OAuth only works on Apple devices

**Solution**:
- This is Apple's limitation
- Show Google OAuth as primary option
- Apple OAuth is optional

---

## Security Best Practices

1. **Keep Private Keys Secure**
   - Never commit private keys to git
   - Use environment variables
   - Rotate keys periodically

2. **Validate Redirect URLs**
   - Only allow trusted domains
   - Use HTTPS in production
   - Verify in Supabase settings

3. **Monitor OAuth Activity**
   - Check Supabase logs regularly
   - Monitor failed login attempts
   - Review connected applications

4. **Update Credentials**
   - Rotate OAuth credentials annually
   - Update when team members leave
   - Review provider settings regularly

---

## Environment Variables

Add these to your `.env.local`:

```env
NEXT_PUBLIC_SUPABASE_URL=https://[PROJECT].supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
```

---

## User Flow

```
1. User visits login page
   ↓
2. Clicks "Google" or "Apple" button
   ↓
3. Redirected to provider's login
   ↓
4. User authenticates with provider
   ↓
5. Provider redirects to /auth/callback
   ↓
6. Callback page establishes session
   ↓
7. Redirected to /ai-model-generator
   ↓
8. User is logged in and can use dashboard
```

---

## Supported Providers

| Provider | Status | Notes |
|----------|--------|-------|
| Google | ✅ Enabled | Works on all devices |
| Apple | ✅ Enabled | Works on Apple devices |
| GitHub | ⏳ Optional | Can be added later |
| Discord | ⏳ Optional | Can be added later |

---

## Support

For issues with OAuth setup:

1. Check Supabase documentation: https://supabase.com/docs/guides/auth
2. Check provider documentation:
   - Google: https://developers.google.com/identity
   - Apple: https://developer.apple.com/sign-in-with-apple/
3. Contact support: support@zehanxtech.com

---

## Checklist

- [ ] Google OAuth configured in Google Cloud Console
- [ ] Google OAuth credentials added to Supabase
- [ ] Apple OAuth configured in Apple Developer
- [ ] Apple OAuth credentials added to Supabase
- [ ] Redirect URLs configured in Supabase
- [ ] Auth callback page created
- [ ] Login page updated with OAuth handlers
- [ ] Tested Google OAuth locally
- [ ] Tested Apple OAuth locally
- [ ] Tested OAuth in production
- [ ] Verified users created in Supabase
- [ ] Verified redirect to dashboard works

---

**Last Updated**: November 16, 2025
**Version**: 1.0.0
