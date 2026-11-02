// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Top-level auth gate.
 *
 * On mount, fetches `/api/config` (via `initAuth`) and either:
 *  - dev mode (`auth_required: false`) -> render children directly.
 *  - cloud mode + at `/auth/callback` -> finalize OIDC code exchange,
 *    rewrite URL back to `/`, render children.
 *  - cloud mode + no user -> redirect to Cognito hosted UI via
 *    `signIn()`. Nothing renders after that (full-page navigation).
 *  - cloud mode + signed-in user -> render children.
 *
 * Wraps the entire app from `main.tsx`. All routes / providers below
 * are unaware of auth — they just see a fully signed-in environment
 * (or dev mode) by the time they mount.
 */
import { useEffect, useState, type ReactNode } from "react";
import {
  initAuth,
  getCurrentUser,
  signIn,
  handleCallback,
  isAuthRequired,
} from "@/lib/auth";

/**
 * Internal state machine for the gate. We can't render children until
 * we know whether auth is required and (in cloud mode) the user is
 * signed in. `loading` is the initial state; `ready` permits rendering;
 * `error` shows a failure message if `/api/config` or the OIDC
 * callback exchange fails unrecoverably.
 */
type GateState = "loading" | "ready" | "error";

/**
 * Props for `AuthGate`. Children are rendered only after auth is
 * established (or skipped in dev mode).
 */
interface AuthGateProps {
  children: ReactNode;
}

/**
 * Auth gate component. See module docstring for behavior.
 *
 * @param children - The app tree to render once auth is ready.
 */
export function AuthGate({ children }: AuthGateProps) {
  const [state, setState] = useState<GateState>("loading");

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        await initAuth();
        if (cancelled) return;

        // Returning from the Cognito hosted UI redirect — finalize the
        // code exchange, then strip the callback path from the URL so
        // a reload doesn't re-enter the callback flow.
        if (window.location.pathname === "/auth/callback") {
          await handleCallback();
          window.history.replaceState({}, "", "/");
          if (!cancelled) setState("ready");
          return;
        }

        if (isAuthRequired()) {
          const user = await getCurrentUser();
          if (cancelled) return;
          if (!user || user.expired) {
            // Full-page redirect to Cognito hosted UI. Nothing
            // renders after this — the browser navigates away.
            await signIn();
            return;
          }
        }

        if (!cancelled) setState("ready");
      } catch (e) {
        // Any unrecoverable failure (e.g. /api/config 500, broken
        // OIDC callback). Show a static error so the user can reload.
        console.error("[AuthGate] failed:", e);
        if (!cancelled) setState("error");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  if (state === "loading") {
    return (
      <div className="flex h-screen items-center justify-center text-sm text-muted-foreground">
        Loading…
      </div>
    );
  }
  if (state === "error") {
    return (
      <div className="flex h-screen items-center justify-center text-sm text-destructive">
        Authentication error. Please reload the page.
      </div>
    );
  }
  return <>{children}</>;
}
