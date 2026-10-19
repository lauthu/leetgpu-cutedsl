import test from 'node:test';
import assert from 'node:assert/strict';
import { request } from 'node:http';
import { manualLogin, parsePastedSession, __test } from './manual-auth.mjs';

const expiry = Math.floor(Date.now() / 1000) + 3600;
const token = `header.${Buffer.from(JSON.stringify({ exp: expiry })).toString('base64url')}.signature`;
const session = { access_token: token, expires_at: expiry, refresh_token: 'do-not-return-refresh-token' };

async function send(url, { method = 'POST', headers = {}, body = JSON.stringify({ value: JSON.stringify(session) }) } = {}) {
  return await new Promise((resolve, reject) => {
    const req = request(url, { method, headers: { 'Content-Type': 'application/json', Origin: new URL(url).origin, ...headers } }, res => {
      let text = '';
      res.setEncoding('utf8');
      res.on('data', chunk => { text += chunk; });
      res.on('end', () => resolve({ status: res.statusCode, headers: res.headers, text }));
    });
    req.on('error', reject);
    req.end(method === 'POST' ? body : undefined);
  });
}

test('pasted JSON and raw JWT normalize to token and expiry only', () => {
  const expected = { access_token: token, expires_at: expiry };
  assert.deepEqual(parsePastedSession(JSON.stringify(session)), expected);
  assert.deepEqual(parsePastedSession(`  ${token}\n`), expected);
  assert.deepEqual(parsePastedSession(JSON.stringify(token)), expected);
  assert.equal(parsePastedSession(JSON.stringify({ access_token: token, expires_at: 1 })), null);
  assert.equal(parsePastedSession('not-a-session'), null);
  assert.equal(parsePastedSession(''), null);
});

test('receiver requires exact nonce path, Host, Origin and JSON content type', async () => {
  const receiver = await __test.createLoginReceiver({ timeoutMs: 5000 });
  try {
    assert.match(new URL(receiver.url).pathname, /^\/[a-f0-9]{64}$/u);
    assert.equal((await send(receiver.url, { headers: { Host: 'evil.test' } })).status, 404);
    assert.equal((await send(`${receiver.url}-wrong`)).status, 404);
    assert.equal((await send(receiver.url, { headers: { Origin: 'https://evil.test' } })).status, 403);
    assert.equal((await send(receiver.url, { headers: { Origin: '' } })).status, 403);
    assert.equal((await send(receiver.url, { headers: { 'Content-Type': 'text/plain' } })).status, 415);
    assert.equal((await send(receiver.url, { method: 'OPTIONS' })).status, 405);
    const page = await send(receiver.url, { method: 'GET' });
    assert.equal(page.status, 200);
    assert.equal(page.headers['cache-control'], 'no-store');
    assert.match(page.headers['content-security-policy'], /frame-ancestors 'none'/u);
    assert.match(page.headers['content-security-policy'], /default-src 'none'/u);
    assert.equal(page.text.includes('sb-yhdtysacdkqoquvkdwdd-auth-token'), true);
    assert.equal(page.text.includes(token), false);
    const submitted = await send(receiver.url);
    assert.equal(submitted.status, 200);
    assert.equal(submitted.text, '{"ok":true}');
    assert.deepEqual(await receiver.result, { access_token: token, expires_at: expiry });
  } finally { receiver.cancel(); }
});

test('invalid, expired, malformed and oversized input never gets echoed', async () => {
  const receiver = await __test.createLoginReceiver({ timeoutMs: 5000 });
  try {
    const secret = 'secret-that-must-not-appear';
    for (const body of [JSON.stringify({ value: secret }), JSON.stringify({ value: JSON.stringify({ access_token: secret, expires_at: 1 }) }), `{broken:${secret}`]) {
      const response = await send(receiver.url, { body });
      assert.equal(response.status, 400);
      assert.equal(response.text.includes(secret), false);
    }
    const response = await send(receiver.url, { body: JSON.stringify({ value: secret.repeat(5000) }) });
    assert.equal(response.status, 413);
    assert.equal(response.text.includes(secret), false);
  } finally {
    receiver.cancel();
    await assert.rejects(receiver.result, /canceled/u);
  }
});

test('timeout closes the loopback listener', async () => {
  const receiver = await __test.createLoginReceiver({ timeoutMs: 30 });
  await assert.rejects(receiver.result, /timed out/u);
  await assert.rejects(send(receiver.url, { method: 'GET' }), /ECONNREFUSED/u);
});

test('fixed port stays on loopback and requires the same forwarded port', async () => {
  const temporary = await __test.createLoginReceiver({ timeoutMs: 5000 });
  const port = Number(new URL(temporary.url).port);
  temporary.cancel();
  await assert.rejects(temporary.result, /canceled/u);
  const receiver = await __test.createLoginReceiver({ port, timeoutMs: 5000 });
  try {
    assert.equal(new URL(receiver.url).hostname, '127.0.0.1');
    assert.equal(Number(new URL(receiver.url).port), port);
    assert.equal((await send(receiver.url, { headers: { Host: `localhost:${port}` } })).status, 404);
    assert.equal((await send(receiver.url, { headers: { Origin: `http://127.0.0.1:${port === 65535 ? port - 1 : port + 1}` } })).status, 403);
    assert.equal((await send(receiver.url)).status, 200);
    assert.deepEqual(await receiver.result, { access_token: token, expires_at: expiry });
  } finally { receiver.cancel(); }
});

test('receiver rejects invalid ports and timeouts before listening', async () => {
  for (const port of [-1, 65536, 0.5, NaN, Infinity, '8787']) {
    await assert.rejects(__test.createLoginReceiver({ port }), /port must be an integer/u);
  }
  for (const timeoutMs of [0, -1, Infinity, NaN]) {
    await assert.rejects(__test.createLoginReceiver({ timeoutMs }), /timeout must be a positive/u);
  }
});

test('success page asks for terminal confirmation rather than claiming persistence', async () => {
  const receiver = await __test.createLoginReceiver({ timeoutMs: 5000 });
  try {
    const response = await send(receiver.url, { method: 'GET' });
    assert.match(response.text, /会话已接收，返回终端确认认证结果/u);
    assert.equal(response.text.includes('连接成功！'), false);
  } finally {
    receiver.cancel();
    await assert.rejects(receiver.result, /canceled/u);
  }
});

test('manual no-browser login prints exact URL and same-port SSH instructions', async () => {
  const messages = [];
  const expected = { access_token: token, expires_at: expiry };
  let receivedOptions;
  let opened = false;
  let canceled = false;
  const url = 'http://127.0.0.1:8787/test-nonce';
  const result = await manualLogin({
    timeoutMs: 1234, port: 8787, openBrowser: false,
    onStatus: message => messages.push(message),
    openBrowserFn: () => { opened = true; },
    createReceiver: async options => {
      receivedOptions = options;
      return { url, result: Promise.resolve(expected), cancel: () => { canceled = true; } };
    },
  });
  assert.deepEqual(result, expected);
  assert.deepEqual(receivedOptions, { timeoutMs: 1234, port: 8787 });
  assert.equal(opened, false);
  assert.equal(canceled, false);
  assert.equal(messages.some(message => message.includes(url)), true);
  assert.equal(messages.some(message => message.includes('ssh -N -L 127.0.0.1:8787:127.0.0.1:8787 <user@remote-host>')), true);
  assert.equal(messages.some(message => message.includes(token)), false);
});

test('browser-open failure leaves the receiver waiting and hides opener errors', async () => {
  const messages = [];
  let resolveSession;
  const result = new Promise(resolve => { resolveSession = resolve; });
  let canceled = false;
  let completed = false;
  const secret = 'sensitive-opener-error';
  const login = manualLogin({
    onStatus: message => messages.push(message),
    openBrowserFn: async () => { throw new Error(secret); },
    createReceiver: async () => ({
      url: 'http://127.0.0.1:43210/test-nonce', result,
      cancel: () => { canceled = true; },
    }),
  });
  login.then(() => { completed = true; });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(completed, false);
  assert.equal(canceled, false);
  assert.equal(messages.some(message => message.includes('still waiting')), true);
  assert.equal(messages.some(message => message.includes(secret)), false);
  resolveSession({ access_token: token, expires_at: expiry });
  assert.deepEqual(await login, { access_token: token, expires_at: expiry });
});

test('manual login propagates receiver timeout after browser-open failure', async () => {
  let canceled = false;
  await assert.rejects(manualLogin({
    openBrowserFn: async () => { throw new Error('No GUI available.'); },
    createReceiver: async () => ({
      url: 'http://127.0.0.1:8787/test-nonce',
      result: Promise.reject(new Error('LeetGPU login timed out.')),
      cancel: () => { canceled = true; },
    }),
  }), /login timed out/u);
  assert.equal(canceled, true);
});

test('cancel closes the receiver without a token result', async () => {
  const receiver = await __test.createLoginReceiver({ timeoutMs: 5000 });
  receiver.cancel();
  await assert.rejects(receiver.result, /canceled/u);
  await assert.rejects(send(receiver.url, { method: 'GET' }), /ECONNREFUSED/u);
});
