// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * SPA auth wrapper around oidc-client-ts.
 *
 * Configures Authorization Code + PKCE against the Cognito hosted UI
 * (cloud mode) or short-circuits to anonymous (dev mode). Token storage is owned
 * by oidc-client-ts: all three tokens — access, id and refresh — are serialized
 * into sessionStorage as part of the User record, and the PKCE code_verifier and
 * nonce into localStorage. None of them are memory-only; see the `userStore`
 * comment below. On token rotation, fires a `tokenChanged` event so
 * ws-client.ts can reconnect with the new token.
 *
 * Public API consumed by other modules:
 *  - `initAuth()` — bootstrap; called by AuthGate on app mount.
 *  - `getAccessToken()` — used by api.ts (Bearer header) and
 *    ws-client.ts (the WebSocket's first frame).
 *  - `authEvents` — `EventTarget` that fires `tokenChanged` whenever
 *    the access token rotates; ws-client.ts listens to trigger
 *    reconnect with the fresh token.
 *  - `signIn()`, `signOut()`, `handleCallback()` — OIDC flow control.
 *  - `isAuthRequired()` / `getCurrentUser()` / `getUserEmail()` — UI
 *    helpers (AuthGate, header user menu).
 */
import {
  UserManager,
  WebStorageStateStore,
  type User,
  type UserManagerSettings,
} from "oidc-client-ts";

/**
 * Backend `/api/config` response shape.
 *
 * Dev mode: `auth_required: false`, `cognito: null`.
 * Cloud mode: `auth_required: true`, all Cognito fields populated.
 *
 * `user_pool_id` exists so the SPA can construct the OIDC issuer URL
 * (`https://cognito-idp.<region>.amazonaws.com/<pool>`) and skip the
 * discovery fetch — Cognito's `/.well-known/openid-configuration`
 * endpoint does NOT set `Access-Control-Allow-Origin`, so a browser
 * fetch from the SPA origin fails CORS.
 */
interface AppConfig {
  auth_required: boolean;
  cognito: {
    domain: string;
    client_id: string;
    region: string;
    user_pool_id: string;
  } | null;
  logout_redirect: string;
}

// Module-level singletons. The OIDC ``UserManager`` owns the access /
// refresh / id tokens; we don't keep a separate copy.
let _config: AppConfig | null = null;
let _userManager: UserManager | null = null;

/**
 * Event bus for auth-related notifications. Currently fires:
 *  - `tokenChanged` — `CustomEvent` whenever the access token rotates
 *    (initial sign-in, silent renew success, callback completion).
 *    `ws-client.ts` listens here to close + reopen its WebSocket with
 *    the freshest token.
 */
export const authEvents = new EventTarget();

/**
 * Bootstrap auth — fetch `/api/config` and configure `UserManager` if
 * `auth_required` is true. Idempotent; subsequent calls return the
 * cached config without re-fetching.
 *
 * Called once by `AuthGate` on mount; safe to call from anywhere else
 * (e.g. `getAccessToken` calls it on first use).
 *
 * @returns The fetched `AppConfig` (cached after first call).
 */
export async function initAuth(): Promise<AppConfig> {
  if (_config !== null) return _config;
  const res = await fetch("/api/config");
  if (!res.ok) {
    throw new Error(`/api/config returned ${res.status}`);
  }
  _config = (await res.json()) as AppConfig;

  if (_config.auth_required && _config.cognito) {
    const { domain, client_id, region, user_pool_id } = _config.cognito;
    // Build OIDC metadata explicitly instead of letting oidc-client-ts
    // discover it. Cognito's `/.well-known/openid-configuration`
    // endpoint does not set `Access-Control-Allow-Origin`, so a
    // browser fetch from the SPA origin fails CORS:
    //   "Access to fetch at '<domain>/.well-known/openid-configuration'
    //    from origin '<spa>' has been blocked by CORS policy."
    // Cognito's endpoint URLs are deterministic from the user pool id
    // + region + hosted UI domain, so we can construct the metadata
    // ourselves and skip the discovery fetch entirely. Same workaround
    // applies to `/oauth2/userInfo` (also no CORS) — we set
    // `loadUserInfo: false` below; the ID token already carries every
    // claim the SPA needs (`sub`, `email`).
    const issuer = `https://cognito-idp.${region}.amazonaws.com/${user_pool_id}`;
    const metadata = {
      issuer,
      authorization_endpoint: `${domain}/oauth2/authorize`,
      token_endpoint: `${domain}/oauth2/token`,
      userinfo_endpoint: `${domain}/oauth2/userInfo`,
      end_session_endpoint: `${domain}/logout`,
      jwks_uri: `${issuer}/.well-known/jwks.json`,
    };

    const settings: UserManagerSettings = {
      authority: issuer,
      metadata,
      client_id,
      redirect_uri: `${window.location.origin}/auth/callback`,
      post_logout_redirect_uri: `${window.location.origin}/`,
      response_type: "code",
      scope: "openid email",
      automaticSilentRenew: true,
      // Skip the userinfo fetch — Cognito's /oauth2/userInfo also
      // lacks CORS, and the ID token already carries every claim
      // the SPA needs.
      loadUserInfo: false,
      // Disable the iframe-based session-monitor — it pings
      // /oauth2/checkSession on the Cognito domain, which the
      // browser blocks the same way. Token rotation is driven by
      // automaticSilentRenew + the addAccessTokenExpired fallback.
      monitorSession: false,
      // The whole User record — refresh token AND access and id tokens — is
      // serialized into sessionStorage so a page reload can recover the session.
      //
      // Stated explicitly because this line used to claim access and id tokens
      // "stay in oidc-client-ts's in-memory store". They do not: userStore is
      // where UserManager persists the User, and its toStorageString() includes
      // all three tokens. Removing this line would not change that either —
      // sessionStorage is the library's own default for userStore. Treating the
      // tokens as memory-only is what the comment made a reader believe, so it
      // mattered more than the setting.
      //
      // sessionStorage rather than localStorage keeps the tokens scoped to the
      // tab and cleared when it closes. Note stateStore is left at its default,
      // which is localStorage, so the PKCE code_verifier and nonce live there.
      userStore: new WebStorageStateStore({ store: window.sessionStorage }),
    };

    _userManager = new UserManager(settings);

    // Access-token expired: kick off a fresh sign-in. In practice
    // `automaticSilentRenew` should refresh before this fires; this
    // is the fallback when refresh fails (e.g. revoked refresh
    // token) and the user must re-authenticate.
    _userManager.events.addAccessTokenExpired(() => {
      void signIn();
    });

    // Token loaded after sign-in / silent renew / callback. Notify
    // subscribers so ws-client.ts can reconnect with the fresh token.
    _userManager.events.addUserLoaded(() => {
      authEvents.dispatchEvent(new CustomEvent("tokenChanged"));
    });

    // Recover any prior session that survived a page reload via the
    // sessionStorage refresh token (returned via getUser()).
    await _userManager.getUser();
  }

  return _config;
}

/**
 * Returns the current access token.
 *
 * Dev mode (`auth_required: false`): returns `""` — callers must NOT
 * send an `Authorization` header at all in that case (don't send
 * `Authorization: Bearer ` with an empty value).
 *
 * Cloud mode: returns the current access token. If the token is
 * expired, returns `""`; the caller's request will then be rejected
 * by the backend with 401, which surfaces to the user via the
 * existing error pipeline. (`automaticSilentRenew` handles refresh
 * proactively before expiry; this is the safety net.)
 *
 * @returns Bearer token string, or `""` if no token is available.
 */
export async function getAccessToken(): Promise<string> {
  if (!_config) await initAuth();
  if (!_config!.auth_required || !_userManager) return "";
  const user = await _userManager.getUser();
  if (!user || user.expired) {
    // automaticSilentRenew should have refreshed already; if we got
    // here, the refresh failed and the next request will 401. The
    // user will have to sign in again — handled by the access-token-
    // expired event listener above.
    return "";
  }
  return user.access_token || "";
}

/**
 * Redirect to the Cognito hosted UI to start the OIDC flow.
 *
 * After successful sign-in, Cognito redirects back to
 * `<origin>/auth/callback` where `AuthGate` calls `handleCallback()`.
 *
 * @throws If called before `initAuth()` in cloud mode.
 */
export async function signIn(): Promise<void> {
  if (!_userManager) {
    throw new Error("signIn called before initAuth in cloud mode");
  }
  await _userManager.signinRedirect();
}

/**
 * Complete the OIDC code exchange after Cognito redirects to
 * `/auth/callback`. Fires `tokenChanged` so ws-client.ts can connect
 * (the resulting user is owned by oidc-client-ts).
 *
 * Called by `AuthGate` when the URL pathname matches `/auth/callback`.
 *
 * @returns The signed-in `User` (oidc-client-ts type).
 * @throws If called before `initAuth()` in cloud mode.
 */
export async function handleCallback(): Promise<User> {
  if (!_userManager) {
    throw new Error("handleCallback called before initAuth");
  }
  const user = await _userManager.signinRedirectCallback();
  authEvents.dispatchEvent(new CustomEvent("tokenChanged"));
  return user;
}

/**
 * Sign out — revokes the refresh token at Cognito, clears local tokens,
 * and redirects to Cognito's `/logout` endpoint, which clears the
 * Cognito session and then redirects back to `_config.logout_redirect`.
 *
 * Why the explicit revoke: Cognito's `/logout` only clears the
 * hosted-UI session cookie. The refresh token (rideing in
 * sessionStorage) and any access token still in flight remain valid
 * against the JWKS validator until natural expiry. POSTing the
 * refresh token to `/oauth2/revoke` first invalidates the family so a
 * stolen refresh token can't mint new access tokens after logout. The
 * outstanding access token still lives until its 1h expiry — we
 * accept that window because adding a server-side denylist would
 * require keeping the JWT validator stateful, which is more cost than
 * the marginal risk warrants.
 *
 * Revoke failure is logged but doesn't block the redirect — a stale
 * sign-out experience (user clicks "Sign out", nothing happens) would
 * be worse than a refresh token that lives until natural expiry.
 *
 * In dev mode (or before initAuth completed), navigates to `/` since
 * there's no Cognito session to clear.
 */
export async function signOut(): Promise<void> {
  if (!_config || !_config.cognito || !_userManager) {
    window.location.href = "/";
    return;
  }
  const { domain, client_id } = _config.cognito;
  // Grab the refresh token BEFORE removeUser() clears it.
  const user = await _userManager.getUser();
  const refreshToken = user?.refresh_token;
  if (refreshToken) {
    try {
      const body = new URLSearchParams({
        token: refreshToken,
        client_id,
      });
      // Public client (no client secret) — Cognito accepts the
      // revoke with just the token + client_id pair. The endpoint
      // returns 200 on success, 200 on already-revoked tokens, and
      // 400 only on malformed requests; either way we proceed to
      // /logout below.
      await fetch(`${domain}/oauth2/revoke`, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: body.toString(),
      });
    } catch (err) {
      // Network error against Cognito — log and continue. A failed
      // revoke means the refresh token lives until natural expiry,
      // which is the same posture as not revoking at all.
      console.warn("[auth] refresh-token revoke failed:", err);
    }
  }
  await _userManager.removeUser();
  const logoutUrl =
    `${domain}/logout` +
    `?client_id=${encodeURIComponent(client_id)}` +
    `&logout_uri=${encodeURIComponent(_config.logout_redirect)}`;
  window.location.href = logoutUrl;
}

/**
 * Whether the backend is in cloud mode (Cognito-protected). Returns
 * `false` in dev mode or before `initAuth` has run.
 *
 * Used by the header to decide whether to render the "Sign out"
 * button (no point in dev mode — there's nothing to sign out of).
 */
export function isAuthRequired(): boolean {
  return _config?.auth_required ?? false;
}

/**
 * Returns the current oidc-client-ts `User` (with all token claims +
 * profile data) or `null` in dev mode / before sign-in.
 *
 * `AuthGate` calls this to decide whether to render children or
 * redirect to the hosted UI.
 */
export async function getCurrentUser(): Promise<User | null> {
  if (!_userManager) return null;
  return _userManager.getUser();
}

/**
 * Returns the signed-in user's email (or `null` in dev mode / before
 * sign-in). Read from the current OIDC user's profile claims — the SPA
 * requests the `email` scope, so the ID token carries this claim
 * without a separate userinfo fetch.
 *
 * Used by the header's user menu to show "Signed in as <email>".
 */
export async function getUserEmail(): Promise<string | null> {
  const user = await getCurrentUser();
  const email = user?.profile?.email;
  return typeof email === "string" ? email : null;
}
