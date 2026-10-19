import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, stat, symlink, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import vm from 'node:vm';

const directory = await mkdtemp(join(tmpdir(), 'leetgpu-history-auth-test-'));
process.env.LEETGPU_HISTORY_CONFIG_DIR = join(directory, 'config');
process.env.LEETGPU_CONFIG_DIR = join(directory, 'config');
const { parseSession, saveSession, loadSession, logout, sessionPath, __test } = await import('./auth.mjs');
test.after(async () => { await rm(directory, { recursive: true, force: true }); });

test('session parser returns only token and future expiry', () => {
  assert.deepEqual(parseSession({ access_token: 'access', expires_at: 200, refresh_token: 'secret' }, 100), { access_token: 'access', expires_at: 200 });
  assert.equal(parseSession({ access_token: 'expired', expires_at: 99 }, 100), null);
  assert.equal(parseSession('{invalid', 100), null);
  assert.equal(parseSession({ access_token: 'unsafe\ntoken', expires_at: 200 }, 100), null);
  const token = `head.${Buffer.from(JSON.stringify({ exp: 200 })).toString('base64url')}.signature`;
  assert.deepEqual(parseSession({ session: { access_token: token } }, 100), { access_token: token, expires_at: 200 });
});

test('session files are private, preserve no refresh token, and logout removes them', async () => {
  assert.equal(await loadSession(), null);
  const session = { access_token: 'test-token', expires_at: Math.floor(Date.now() / 1000) + 3600 };
  await saveSession({ ...session, refresh_token: 'must-not-save' });
  assert.deepEqual(await loadSession(), session);
  assert.equal((await stat(sessionPath)).mode & 0o777, 0o600);
  assert.equal((await stat(join(directory, 'config'))).mode & 0o777, 0o700);
  assert.equal((await readFile(sessionPath, 'utf8')).includes('must-not-save'), false);
  await writeFile(sessionPath, JSON.stringify({ access_token: 'expired', expires_at: 1 }));
  await assert.rejects(loadSession(), /invalid or expired/u);
  await logout();
  assert.equal(await loadSession(), null);
});

test('load refuses session symlinks', async () => {
  const external = join(directory, 'external');
  await writeFile(external, '{}', { mode: 0o644 });
  await symlink(external, sessionPath);
  await assert.rejects(loadSession(), /symbolic link/u);
  await logout();
  assert.equal((await stat(external)).mode & 0o777, 0o644);
});

test('only exact LeetGPU page origins and owned loopback sockets are accepted', () => {
  const target = { type: 'page', url: 'https://leetgpu.com/challenges', webSocketDebuggerUrl: 'ws://localhost:1234/devtools/page/abc-123' };
  assert.equal(__test.pageSocket(target, 1234), 'ws://127.0.0.1:1234/devtools/page/abc-123');
  for (const url of ['http://leetgpu.com', 'https://leetgpu.com.evil.test', 'https://leetgpu.com:444', 'https://accounts.google.com']) {
    assert.equal(__test.pageSocket({ ...target, url }, 1234), null);
  }
  assert.equal(__test.pageSocket({ ...target, webSocketDebuggerUrl: 'ws://evil.test:1234/devtools/page/abc' }, 1234), null);
  assert.equal(__test.pageSocket(target, 5678), null);
  assert.equal(__test.parseDevToolsPort('1234\n/devtools/browser/abc-123\n').port, 1234);
  assert.equal(__test.parseDevToolsPort('1234\nhttps://evil.test\n'), null);
});

test('page extraction never accesses storage on a different origin and returns no refresh token', () => {
  const expression = `(${__test.captureSessionInPage.toString()})()`;
  const blocked = { location: { origin: 'https://accounts.google.com' }, get localStorage() { throw new Error('Storage accessed on wrong origin'); } };
  assert.equal(vm.runInNewContext(expression, blocked), null);
  const session = { access_token: 'access', expires_at: Math.floor(Date.now() / 1000) + 1000, refresh_token: 'do-not-capture' };
  const captured = vm.runInNewContext(expression, {
    location: { origin: 'https://leetgpu.com' },
    localStorage: { getItem: key => key === 'sb-yhdtysacdkqoquvkdwdd-auth-token' ? JSON.stringify(session) : null },
  });
  assert.deepEqual(JSON.parse(JSON.stringify(captured)), { access_token: session.access_token, expires_at: session.expires_at });
});

test('CDP closes sockets on success, timeout, and abort', async () => {
  const sockets = [];
  class FakeSocket extends EventTarget {
    constructor() { super(); sockets.push(this); queueMicrotask(() => this.dispatchEvent(new Event('open'))); }
    send(value) {
      if (JSON.parse(value).method === 'success') queueMicrotask(() => this.dispatchEvent(new MessageEvent('message', { data: JSON.stringify({ id: 1, result: { ok: true } }) })));
    }
    close() { this.closed = true; }
  }
  assert.deepEqual(await __test.callCdp('ws://127.0.0.1', 'success', {}, { WebSocketClass: FakeSocket }), { ok: true });
  await assert.rejects(__test.callCdp('ws://127.0.0.1', 'timeout', {}, { WebSocketClass: FakeSocket, timeoutMs: 10 }), /timed out/u);
  const controller = new AbortController();
  const pending = __test.callCdp('ws://127.0.0.1', 'abort', {}, { WebSocketClass: FakeSocket, signal: controller.signal });
  controller.abort(new Error('Canceled by test'));
  await assert.rejects(pending, /Canceled by test/u);
  assert.equal(sockets.length, 3);
  assert.equal(sockets.every(socket => socket.closed), true);
});

test('cleanup signals only its owned browser process', async () => {
  const signals = [];
  const browser = { pid: 123, exitCode: null, signalCode: null, kill(signal) { signals.push(signal); this.signalCode = signal; } };
  await __test.stopOwnedBrowser(browser);
  assert.deepEqual(signals, ['SIGTERM']);
  await __test.stopOwnedBrowser(browser);
  assert.deepEqual(signals, ['SIGTERM']);
});
