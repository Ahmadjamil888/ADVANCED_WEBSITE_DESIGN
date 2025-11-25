# Login & Deployment Configuration Guide

## Overview

This guide explains how to configure the login system and update the application URL from localhost to zehanxtech.com.

---

## 🔐 Login Configuration

### What Was Updated

The login system has been updated to redirect users to the correct page after authentication:

**File**: `src/app/login/page.tsx`

#### Changes Made

1. **Post-Login Redirect**: Updated to redirect to `/ai-model-generator` instead of `/ai-workspace`
   ```typescript
   // After successful login, users are redirected to:
   router.push('/ai-model-generator')
   ```

2. **Google OAuth Redirect**: Updated to use zehanxtech.com in production
   ```typescript
   const redirectUrl = process.env.NODE_ENV === 'production' 
     ? 'https://zehanxtech.com/ai-model-generator'
     : `${window.location.origin}/ai-model-generator`
   ```

### Login Flow

```
User visits /login
    ↓
Enters credentials or clicks "Continue with Google"
    ↓
Authentication with Supabase
    ↓
Redirected to /ai-model-generator
    ↓
User can start creating AI models
```

---

## 🌐 URL Configuration

### Development (localhost)

For local development, the app uses `http://localhost:3000`:

```bash
# .env.local (development)
NEXT_PUBLIC_APP_URL=http://localhost:3000
```

### Production (zehanxtech.com)

For production deployment, update to `https://zehanxtech.com`:

```bash
# .env.local (production)
NEXT_PUBLIC_APP_URL=https://zehanxtech.com
```

### Where NEXT_PUBLIC_APP_URL is Used

1. **Stripe Checkout Redirect URLs**
   - Success URL: `${NEXT_PUBLIC_APP_URL}/ai-workspace?success=true`
   - Cancel URL: `${NEXT_PUBLIC_APP_URL}/ai-workspace?canceled=true`

2. **Google OAuth Redirect**
   - Production: `https://zehanxtech.com/ai-model-generator`
   - Development: `http://localhost:3000/ai-model-generator`

3. **Email Verification Links** (Supabase)
   - Automatically uses the configured domain

---

## 📋 Environment Variables

### Required Variables

Create a `.env.local` file with the following:

```env
# Application URL
NEXT_PUBLIC_APP_URL=https://zehanxtech.com

# Supabase
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_key

# AI APIs
GROQ_API_KEY=your_groq_key
E2B_API_KEY=your_e2b_key

# Stripe (for billing)
STRIPE_PUBLIC_KEY=pk_live_your_key
STRIPE_SECRET_KEY=sk_live_your_key

# Security
PAGE_ACCESS_PASSWORD=your_password
```

### Getting API Keys

| Service | URL | Purpose |
|---------|-----|---------|
| Supabase | https://supabase.com/dashboard | Database & Auth |
| Groq | https://console.groq.com | Code Generation |
| E2B | https://e2b.dev/dashboard | Sandbox Environment |
| Stripe | https://dashboard.stripe.com | Payment Processing |

---

## 🚀 Deployment Steps

### Step 1: Update Environment Variables

1. Go to your hosting platform (Vercel, Netlify, etc.)
2. Set `NEXT_PUBLIC_APP_URL=https://zehanxtech.com`
3. Set all other required environment variables

### Step 2: Configure Supabase

1. Go to Supabase Dashboard
2. Navigate to Authentication → URL Configuration
3. Add `https://zehanxtech.com` to allowed redirect URLs
4. Add `https://zehanxtech.com/login` as the site URL

### Step 3: Configure Google OAuth

1. Go to Google Cloud Console
2. Update OAuth redirect URIs:
   - `https://zehanxtech.com/auth/callback`
   - `https://zehanxtech.com/ai-model-generator`
3. Update authorized JavaScript origins:
   - `https://zehanxtech.com`

### Step 4: Configure Stripe

1. Go to Stripe Dashboard
2. Update webhook URLs to:
   - `https://zehanxtech.com/api/billing/webhook`
3. Update redirect URLs:
   - Success: `https://zehanxtech.com/ai-workspace?success=true`
   - Cancel: `https://zehanxtech.com/ai-workspace?canceled=true`

### Step 5: Update DNS

1. Point your domain `zehanxtech.com` to your hosting provider
2. Configure SSL/TLS certificate (usually automatic with Vercel/Netlify)
3. Wait for DNS propagation (5-30 minutes)

---

## ✅ Testing the Login Flow

### Local Testing

```bash
# 1. Start development server
npm run dev

# 2. Visit login page
http://localhost:3000/login

# 3. Sign in with email/password or Google

# 4. Should redirect to
http://localhost:3000/ai-model-generator
```

### Production Testing

```
1. Visit https://zehanxtech.com/login
2. Sign in with email/password or Google
3. Should redirect to https://zehanxtech.com/ai-model-generator
4. Verify all features work correctly
```

---

## 🔍 Troubleshooting

### "Redirect URL mismatch" Error

**Problem**: Google OAuth shows redirect URL mismatch

**Solution**:
1. Check Google Cloud Console OAuth settings
2. Ensure `https://zehanxtech.com/ai-model-generator` is in redirect URIs
3. Clear browser cache and try again

### "Invalid URL" Error

**Problem**: Application shows invalid URL error

**Solution**:
1. Verify `NEXT_PUBLIC_APP_URL` is set correctly
2. Check for typos in environment variables
3. Restart the application

### Stripe Checkout Not Working

**Problem**: Stripe checkout fails or redirects incorrectly

**Solution**:
1. Verify `STRIPE_SECRET_KEY` is set
2. Check Stripe webhook configuration
3. Verify success/cancel URLs in Stripe dashboard

### Email Verification Not Working

**Problem**: Users don't receive verification emails

**Solution**:
1. Check Supabase email settings
2. Verify domain is added to Supabase URL configuration
3. Check spam folder for emails

---

## 📊 Login Page Features

### Authentication Methods

1. **Email & Password**
   - Sign up with email
   - Sign in with existing account
   - Password recovery

2. **Google OAuth**
   - One-click sign in
   - Automatic account creation
   - Profile information import

### User Experience

- Beautiful split-screen design
- Responsive on mobile devices
- Loading states and error messages
- Remember me option
- Forgot password link

---

## 🔒 Security Considerations

### HTTPS Only

- Production uses HTTPS only
- All cookies are secure and httpOnly
- CSRF protection enabled

### Environment Variables

- Never commit `.env.local` to git
- Use `.env.example` for documentation
- Rotate API keys regularly

### Session Management

- Sessions expire after inactivity
- Secure session cookies
- Automatic logout on browser close

---

## 📝 Configuration Checklist

Before deploying to production:

- [ ] Update `NEXT_PUBLIC_APP_URL` to `https://zehanxtech.com`
- [ ] Configure Supabase redirect URLs
- [ ] Configure Google OAuth redirect URIs
- [ ] Configure Stripe webhook URLs
- [ ] Set all required environment variables
- [ ] Test login flow locally
- [ ] Test login flow in production
- [ ] Verify email verification works
- [ ] Verify Google OAuth works
- [ ] Test Stripe checkout flow
- [ ] Monitor error logs

---

## 🎯 Post-Login User Journey

After successful login, users are taken to `/ai-model-generator` where they can:

1. **Select an AI Model** (Mixtral, Llama 2, or Gemma)
2. **Describe Their Model** in natural language
3. **Generate & Train** the model automatically
4. **Get a Live Deployment URL** with REST API endpoints

---

## 📞 Support

For issues with:

- **Authentication**: Check Supabase documentation
- **Google OAuth**: Check Google Cloud Console
- **Stripe**: Check Stripe documentation
- **E2B**: Check E2B documentation
- **Groq**: Check Groq documentation

---

**Last Updated**: November 15, 2025
**Version**: 1.0.0
