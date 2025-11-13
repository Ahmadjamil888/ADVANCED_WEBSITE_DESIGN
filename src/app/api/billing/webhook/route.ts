import { NextRequest, NextResponse } from 'next/server';
import Stripe from 'stripe';
import { getSupabaseOrThrow } from '@/lib/supabase';

const STRIPE_API_VERSION: Stripe.StripeConfig['apiVersion'] = '2025-10-29.clover';

function getStripeClient() {
  const secretKey = process.env.STRIPE_SECRET_KEY;
  if (!secretKey) {
    return null;
  }
  return new Stripe(secretKey, {
    apiVersion: STRIPE_API_VERSION,
  });
}

export async function POST(req: NextRequest) {
  const stripe = getStripeClient();
  const webhookSecret = process.env.STRIPE_WEBHOOK_SECRET;

  if (!stripe || !webhookSecret) {
    return NextResponse.json(
      { error: 'Stripe webhook is not configured properly.' },
      { status: 500 },
    );
  }

  const body = await req.text();
  const signature = req.headers.get('stripe-signature');

  if (!signature) {
    return NextResponse.json({ error: 'No signature' }, { status: 400 });
  }

  let event: Stripe.Event;

  try {
    event = stripe.webhooks.constructEvent(body, signature, webhookSecret);
  } catch (err: any) {
    console.error('Webhook signature verification failed:', err.message);
    return NextResponse.json({ error: err.message }, { status: 400 });
  }

  const supabase = getSupabaseOrThrow();

  try {
    if (event.type === 'checkout.session.completed') {
      const session = event.data.object as Stripe.Checkout.Session;
      const userId = session.metadata?.userId;
      const planType = session.metadata?.planType;

      if (userId && planType) {
        const modelsLimit = planType === 'pro' ? 10 : 30;
        const hasApiAccess = planType === 'enterprise';

        await (supabase.from('billing').update as any)({
          plan_type: planType,
          models_limit: modelsLimit,
          has_api_access: hasApiAccess,
          is_paid: true,
          billing_cycle_start: new Date().toISOString(),
          billing_cycle_end: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString(),
        }).eq('user_id', userId);

        await (supabase.from('users').update as any)({
          subscription_plan: planType,
          subscription_expires_at: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString(),
        }).eq('id', userId);
      }
    }

    if (event.type === 'customer.subscription.deleted') {
      const subscription = event.data.object as Stripe.Subscription;
      const userId = subscription.metadata?.userId;

      if (userId) {
        await (supabase.from('billing').update as any)({
          plan_type: 'free',
          models_limit: 1,
          has_api_access: false,
          is_paid: false,
        }).eq('user_id', userId);

        await (supabase.from('users').update as any)({
          subscription_plan: 'free',
          subscription_expires_at: null,
        }).eq('id', userId);
      }
    }

    return NextResponse.json({ received: true });
  } catch (error: any) {
    console.error('Webhook processing error:', error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}

