// Environment variable validation for AI Workspace
export function validateEnvironment() {
  const required = {
    'INNGEST_SIGNING_KEY': process.env.INNGEST_SIGNING_KEY,
    'INNGEST_EVENT_KEY': process.env.INNGEST_EVENT_KEY,
    'E2B_API_KEY': process.env.E2B_API_KEY,
    'KAGGLE_USERNAME': process.env.KAGGLE_USERNAME,
    'KAGGLE_KEY': process.env.KAGGLE_KEY,
  };

  const missing = Object.entries(required)
    .filter(([key, value]) => !value)
    .map(([key]) => key);

  if (missing.length > 0) {
    console.warn('⚠️  Missing environment variables:', missing.join(', '));
    return false;
  }

  console.log('✅ All required environment variables are configured');
  return true;
}

export function getEnvironmentInfo() {
  return {
    inngest: {
      configured: !!(process.env.INNGEST_SIGNING_KEY && process.env.INNGEST_EVENT_KEY),
      signingKey: process.env.INNGEST_SIGNING_KEY ? '✅ Set' : '❌ Missing',
      eventKey: process.env.INNGEST_EVENT_KEY ? '✅ Set' : '❌ Missing',
    },
    e2b: {
      configured: !!process.env.E2B_API_KEY,
      apiKey: process.env.E2B_API_KEY ? '✅ Set' : '❌ Missing',
    },
    kaggle: {
      configured: !!(process.env.KAGGLE_USERNAME && process.env.KAGGLE_KEY),
      username: process.env.KAGGLE_USERNAME ? '✅ Set' : '❌ Missing',
      key: process.env.KAGGLE_KEY ? '✅ Set' : '❌ Missing',
    },
    supabase: {
      configured: !!(process.env.NEXT_PUBLIC_SUPABASE_URL && process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY),
      url: process.env.NEXT_PUBLIC_SUPABASE_URL ? '✅ Set' : '❌ Missing',
      anonKey: process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ? '✅ Set' : '❌ Missing',
    }
  };
}