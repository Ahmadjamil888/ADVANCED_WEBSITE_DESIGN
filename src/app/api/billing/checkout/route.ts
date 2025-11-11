import { NextRequest, NextResponse } from 'next/server';
import Stripe from 'stripe';
import { getSupabaseOrThrow } from '@/lib/supabase';

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY || '', {
  apiVersion: '2024-11-20.acacia',
});

const PLAN_PRICES: Record<string, number> = {
  pro: 5000, // $50.00 in cents
  enterprise: 45000, // $450.00 in cents
};

export async function POST(req: NextRequest) {
  try {
    const { planType } = await req.json();

    if (!['pro', 'enterprise'].includes(planType)) {
      return NextResponse.json({ error: 'Invalid plan type' }, { status: 400 });
    }

    const supabase = getSupabaseOrThrow();
    const { data: { user } } = await supabase.auth.getUser();

    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    // Create Stripe checkout session
    const session = await stripe.checkout.sessions.create({
      payment_method_types: ['card'],
      line_items: [
        {
          price_data: {
            currency: 'usd',
            product_data: {
              name: `${planType.toUpperCase()} Plan`,
              description: planType === 'pro' 
                ? '10 AI Models, Priority Support'
                : '30 AI Models, API Access, 24/7 Support',
            },
            unit_amount: PLAN_PRICES[planType],
            recurring: {
              interval: 'month',
            },
          },
          quantity: 1,
        },
      ],
      mode: 'subscription',
      success_url: `${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}/ai-workspace?success=true&plan=${planType}`,
      cancel_url: `${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}/ai-workspace?canceled=true`,
      client_reference_id: user.id,
      metadata: {
        userId: user.id,
        planType,
      },
    });

    return NextResponse.json({ sessionId: session.id, url: session.url });
  } catch (error: any) {
    console.error('Stripe checkout error:', error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}

