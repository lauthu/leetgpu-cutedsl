import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, rm, stat, utimes, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';

const directory = await mkdtemp(join(tmpdir(), 'leetgpu-renew-test-'));
process.env.LEETGPU_CONFIG_DIR = join(directory, 'config');
process.env.LEETGPU_SECRET_STORE = 'file';
const { exchangeRefreshToken, renewSession, ensureSession, AUTH_ORIGIN, __test } = await import('./renew.mjs');
const { saveSession, loadSession, sessionPath } = await import('./auth.mjs');
const { openSecretStore } = await import('./secrets.mjs');
const store = openSecretStore();
test.after(async () => { await rm(directory, { recursive: true, force: true }); });

const nowSeconds = () => Math.floor(Date.now() / 1000);
const jwt = exp => `head.${Buffer.from(JSON.stringify({ sub: 'renew-test-user', exp })).toString('base64url')}.signature`;
const oldRefresh = 'old-refresh-token-1';

// A sign-in server that hands out a fresh access token and a replacement refresh token per call.
function signIn({ status = 200, delayMs = 0, body } = {}) {
  const calls = [];
  const fetchImpl = async (url, init) => {
    calls.push({ url: String(url), init });
    if (delayMs) await new Promise(resolve => setTimeout(resolve, delayMs));
    if (status !== 200) return new Response(JSON.stringify({ error_code: 'refresh_token_not_found', msg: 'private-server-detail' }), { status });
    const expires_at = nowSeconds() + 3600;
    return Response.json(body ?? { access_token: jwt(expires_at), expires_at, refresh_token: `next-refresh-token-${calls.length}`, user: { email: 'private@example.test' } });
  };
  return { calls, fetchImpl };
}

async function expire({ refresh = oldRefresh } = {}) {
  await mkdir(join(directory, 'config'), { recursive: true, mode: 0o700 });
  await writeFile(sessionPath, JSON.stringify({ access_token: jwt(1), expires_at: 1 }), { mode: 0o600 });
  if (refresh) await store.set(refresh); else await store.delete();
}

test('exchange posts the token only in the body to the fixed sign-in origin and returns its replacement', async () => {
  const server = signIn();
  const result = await exchangeRefreshToken(oldRefresh, { fetchImpl: server.fetchImpl });
  assert.equal(server.calls.length, 1);
  const { url, init } = server.calls[0];
  assert.equal(url, `${AUTH_ORIGIN}/auth/v1/token?grant_type=refresh_token`);
  assert.equal(AUTH_ORIGIN, 'https://yhdtysacdkqoquvkdwdd.supabase.co');
  assert.equal(init.method, 'POST');
  assert.equal(init.redirect, 'error');
  assert.deepEqual(JSON.parse(init.body), { refresh_token: oldRefresh });
  assert.equal(JSON.stringify(init.headers).includes(oldRefresh), false);
  assert.equal(init.headers.Authorization, `Bearer ${init.headers.apikey}`);
  assert.deepEqual(Object.keys(result.session), ['access_token', 'expires_at']);
  assert.equal(result.refreshToken, 'next-refresh-token-1');
});

test('exchange separates a dead sign-in from a temporary failure and leaks no server or error text', async () => {
  for (const status of [400, 401, 403]) {
    await assert.rejects(exchangeRefreshToken(oldRefresh, signIn({ status })), error => error.code === 'RENEW_REJECTED' && /login again/u.test(error.message) && !error.message.includes('private'));
  }
  for (const status of [429, 500, 503]) {
    await assert.rejects(exchangeRefreshToken(oldRefresh, signIn({ status })), error => error.code === 'RENEW_UNAVAILABLE' && !error.message.includes('private'));
  }
  const broken = async () => { throw new Error(`network detail ${oldRefresh}`); };
  await assert.rejects(exchangeRefreshToken(oldRefresh, { fetchImpl: broken }), error => error.code === 'RENEW_UNAVAILABLE' && !error.message.includes(oldRefresh));
  for (const body of [{ access_token: jwt(nowSeconds() + 60) }, { refresh_token: 'next-refresh-token-9' }, { access_token: jwt(1), refresh_token: 'next-refresh-token-9' }]) {
    await assert.rejects(exchangeRefreshToken(oldRefresh, signIn({ body })), error => error.code === 'RENEW_UNAVAILABLE');
  }
});

test('a session that is still good is used without touching the network', async () => {
  const session = { access_token: jwt(nowSeconds() + 3600), expires_at: nowSeconds() + 3600 };
  await saveSession(session);
  await store.set(oldRefresh);
  const server = signIn();
  assert.deepEqual(await ensureSession({ fetchImpl: server.fetchImpl }), session);
  assert.equal(server.calls.length, 0);
  assert.equal(await store.get(), oldRefresh);
});

test('an expired session renews, rotates the stored token and saves no refresh token in the session file', async () => {
  await expire();
  const server = signIn();
  const session = await ensureSession({ fetchImpl: server.fetchImpl });
  assert.equal(server.calls.length, 1);
  assert.ok(session.expires_at > nowSeconds() + 3000);
  assert.deepEqual(await loadSession(), session);
  assert.equal(await store.get(), 'next-refresh-token-1');
  const saved = await readFile(sessionPath, 'utf8');
  assert.equal(saved.includes('refresh'), false);
  assert.equal(saved.includes('private@example.test'), false);
  await assert.rejects(stat(__test.lockPath), { code: 'ENOENT' });
});

test('a session inside the safety margin renews early but survives an unreachable sign-in server', async () => {
  const closing = { access_token: jwt(nowSeconds() + 30), expires_at: nowSeconds() + 30 };
  await saveSession(closing);
  await store.set(oldRefresh);
  assert.deepEqual(await ensureSession(signIn({ status: 503 })), closing);
  assert.equal(await store.get(), oldRefresh);
  const server = signIn();
  assert.ok((await ensureSession({ fetchImpl: server.fetchImpl })).expires_at > nowSeconds() + 3000);
  assert.equal(server.calls.length, 1);
});

test('a rejected refresh token is forgotten, while a temporary failure keeps it for the next try', async () => {
  await expire();
  await assert.rejects(ensureSession(signIn({ status: 503 })), { code: 'RENEW_UNAVAILABLE' });
  assert.equal(await store.get(), oldRefresh);
  await assert.rejects(ensureSession(signIn({ status: 400 })), { code: 'RENEW_REJECTED' });
  assert.equal(await store.get(), null);
  await assert.rejects(stat(__test.lockPath), { code: 'ENOENT' });
});

test('without a stored refresh token the old behaviour is unchanged', async () => {
  await expire({ refresh: null });
  const server = signIn();
  await assert.rejects(ensureSession({ fetchImpl: server.fetchImpl }), { code: 'SESSION_INVALID' });
  await rm(sessionPath, { force: true });
  assert.equal(await ensureSession({ fetchImpl: server.fetchImpl }), null);
  assert.equal(await renewSession({ fetchImpl: server.fetchImpl }), null);
  assert.equal(server.calls.length, 0);
});

test('commands racing to renew spend the refresh token exactly once', async () => {
  await expire();
  const server = signIn({ delayMs: 150 });
  const sessions = await Promise.all([1, 2, 3].map(() => ensureSession({ fetchImpl: server.fetchImpl })));
  assert.equal(server.calls.length, 1);
  assert.deepEqual(sessions[1], sessions[0]);
  assert.deepEqual(sessions[2], sessions[0]);
  assert.equal(await store.get(), 'next-refresh-token-1');
});

test('a lock left by a killed command is cleared, and a live one gives up with a clear message', async () => {
  await expire();
  await writeFile(__test.lockPath, '');
  const old = new Date(Date.now() - 120_000);
  await utimes(__test.lockPath, old, old);
  const server = signIn();
  await ensureSession({ fetchImpl: server.fetchImpl });
  assert.equal(server.calls.length, 1);
  await expire();
  await writeFile(__test.lockPath, '');
  await assert.rejects(ensureSession({ fetchImpl: server.fetchImpl, lock: { waitMs: 250 } }), error => error.code === 'RENEW_UNAVAILABLE' && /still renewing/u.test(error.message));
  assert.equal(server.calls.length, 1);
  assert.equal(await store.get(), oldRefresh);
  await rm(__test.lockPath, { force: true });
});
