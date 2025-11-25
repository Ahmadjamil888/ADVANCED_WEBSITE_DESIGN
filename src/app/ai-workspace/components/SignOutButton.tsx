'use client';

import { useRouter } from 'next/navigation';
import { supabase } from '@/lib/supabase';
import styles from './SignOutButton.module.css';

export function SignOutButton() {
  const router = useRouter();

  const handleSignOut = async () => {
    if (supabase) {
      await supabase.auth.signOut();
    }
    router.push('/login');
  };

  return (
    <button onClick={handleSignOut} className={styles.button}>
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
      </svg>
      Sign Out
    </button>
  );
}
