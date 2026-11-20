import { mkdir, open, rm, stat } from 'node:fs/promises';
import { join } from 'node:path';
import { loadSession, parseSession, refreshTokenOf, saveSession } from './auth.mjs';
import { configDirectory } from './paths.mjs';
import { openSecretStore } from './secrets.mjs';

// LeetGPU's Supabase project, the same one named by its `sb-…-auth-token` storage key.
export const AUTH_ORIGIN = 'https://yhdtysacdkqoquvkdwdd.supabase.co';
// The website's public anon key: a client identifier that ships in its JavaScript,
// not a secret. Verified on 2026-11-19 (claims: ref yhdtysacdkqoquvkdwdd, role anon).
const ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InloZHR5c2FjZGtxb3F1dmtkd2RkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Mzg2MzQ3MTksImV4cCI6MjA1NDIxMDcxOX0.aevKbBm0HgYbEI6CQ6UobY728lYwhi7dawnI3F_d0QM';
// Renew slightly early so a command never starts with a token about to lapse.
const MARGIN_SECONDS = 60;
const lockPath = join(configDirectory, '.renew.lock');

const renewError = (code, message) => Object.assign(new Error(message), { code });

/** One exchange. A refresh token is single use: the response carries its replacement. */
export async function exchangeRefreshToken(refreshToken, { fetchImpl = globalThis.fetch } = {}) {
  let response;
  try {
    response = await fetchImpl(`${AUTH_ORIGIN}/auth/v1/token?grant_type=refresh_token`, {
      method: 'POST', redirect: 'error', signal: AbortSignal.timeout(20_000),
      headers: { apikey: ANON_KEY, Authorization: `Bearer ${ANON_KEY}`, 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({ refresh_token: refreshToken }),
    });
  } catch {
    throw renewError('RENEW_UNAVAILABLE', 'Could not reach LeetGPU sign-in to renew the session. Check your connection and retry.');
  }
  // The server no longer accepts this token (revoked, expired, or already spent elsewhere).
  if (response.status === 400 || response.status === 401 || response.status === 403) {
    throw renewError('RENEW_REJECTED', 'The saved LeetGPU sign-in is no longer valid. Run login again.');
  }
  if (!response.ok) throw renewError('RENEW_UNAVAILABLE', `LeetGPU sign-in returned HTTP ${response.status}. Retry later.`);
  let body;
  try { body = await response.json(); } catch { body = null; }
  const session = parseSession(body);
  const next = refreshTokenOf(body);
  if (!session || !next) throw renewError('RENEW_UNAVAILABLE', 'LeetGPU sign-in returned an unexpected response. Retry later.');
  return { session, refreshToken: next };
}

async function validSession(nowSeconds) {
  try {
    const session = await loadSession();
    return session && session.expires_at - nowSeconds > MARGIN_SECONDS ? session : null;
  } catch (error) {
    if (error.code === 'SESSION_INVALID') return null;
    throw error;
  }
}

// Two commands must never spend the same refresh token, so renewal is serialized per config directory.
async function acquireLock({ waitMs = 30_000, staleMs = 60_000 } = {}) {
  await mkdir(configDirectory, { recursive: true, mode: 0o700 });
  const deadline = Date.now() + waitMs;
  for (;;) {
    try {
      const handle = await open(lockPath, 'wx', 0o600);
      await handle.close();
      return () => rm(lockPath, { force: true });
    } catch (error) {
      if (error.code !== 'EEXIST') throw error;
    }
    try {
      // A lock left by a killed process must not block sign-in forever.
      if (Date.now() - (await stat(lockPath)).mtimeMs > staleMs) { await rm(lockPath, { force: true }); continue; }
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
      continue;
    }
    if (Date.now() >= deadline) throw renewError('RENEW_UNAVAILABLE', 'Another LeetGPU command is still renewing the session. Retry in a moment.');
    await new Promise(resolve => setTimeout(resolve, 100));
  }
}

/**
 * Renew from the stored refresh token. Returns null when there is none, so the
 * caller can fall back to asking for a login.
 */
export async function renewSession({ store = openSecretStore(), fetchImpl, now = () => Date.now() / 1000, lock } = {}) {
  if (!await store.get()) return null;
  const release = await acquireLock(lock);
  try {
    // Another command may have renewed while this one waited.
    const current = await validSession(now());
    if (current) return current;
    const refreshToken = await store.get();
    if (!refreshToken) return null;
    let renewed;
    try { renewed = await exchangeRefreshToken(refreshToken, { fetchImpl }); } catch (error) {
      if (error.code === 'RENEW_REJECTED') await store.delete();
      throw error;
    }
    // Store the replacement first: the old token is already spent, so losing
    // the new one here would end the sign-in.
    await store.set(renewed.refreshToken);
    await saveSession(renewed.session);
    return renewed.session;
  } finally { await release(); }
}

/** The session for an authenticated command: the saved one if still good, otherwise a renewed one. */
export async function ensureSession(options = {}) {
  const now = options.now ?? (() => Date.now() / 1000);
  let session = null;
  let invalid = null;
  try { session = await loadSession(); } catch (error) {
    if (error.code !== 'SESSION_INVALID') throw error;
    invalid = error;
  }
  if (session && session.expires_at - now() > MARGIN_SECONDS) return session;
  let renewed;
  try { renewed = await renewSession(options); } catch (error) {
    // Inside the margin the old token still works; use it if renewal is only unreachable.
    if (session && error.code === 'RENEW_UNAVAILABLE') return session;
    throw error;
  }
  if (renewed) return renewed;
  if (session) return session;
  if (invalid) throw invalid;
  return null;
}

export const __test = { lockPath, MARGIN_SECONDS };
