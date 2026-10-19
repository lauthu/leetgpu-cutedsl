import test from 'node:test';
import assert from 'node:assert/strict';
import { getEventListeners } from 'node:events';
import { buildSubmission, submitSolution } from './submit.mjs';

const token = `e30.${Buffer.from(JSON.stringify({ sub: 'test-user', exp: 9999999999 })).toString('base64url')}.signature`;
const options = { token, challenge: '1', language: 'cute', gpu: 'H100', files: [{ name: 'solution.py', content: 'def solve(): pass\n' }] };
const payload = buildSubmission(options);

function transport() {
  const sockets = [];
  class FakeWebSocket extends EventTarget {
    sent = [];
    closed = false;
    constructor(url) { super(); this.url = url; sockets.push(this); }
    send(value) { this.sent.push(JSON.parse(value)); }
    close() { this.closed = true; this.dispatchEvent(new Event('close')); }
    open() { this.dispatchEvent(new Event('open')); }
    message(data) { this.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(data) })); }
  }
  return { sockets, WebSocketClass: FakeWebSocket };
}

test('submission payload is private, targets H100, and uses the authenticated JWT subject', () => {
  assert.deepEqual(payload, {
    files: options.files, language: 'cute', accelerator: 'H100', mode: 'accelerated',
    challengeId: 1, userId: 'test-user', public: false,
  });
  for (const change of [
    { challenge: 0 }, { challenge: 1.5 }, { language: 'invalid' }, { gpu: 'H100\n' },
    { files: [] }, { files: [{ name: '../solution.py', content: 'x' }] },
    { files: [{ name: 'solution.py', content: 12 }] }, { files: [options.files[0], options.files[0]] },
    { token: 'not-a-jwt' }, { token: 'e30.e30.signature' },
  ]) assert.throws(() => buildSubmission({ ...options, ...change }));
});

test('sends exactly one private envelope and returns the server submission ID after success', async () => {
  const fake = transport();
  const seen = [];
  const pending = submitSolution({ token, payload: { ...payload, public: true }, submissionId: 'request-id', ...fake, onEvent: event => seen.push(event) });
  const socket = fake.sockets[0];
  assert.equal(socket.url, 'wss://api.leetgpu.com/api/v1/ws/submit');
  socket.open();
  socket.open();
  assert.deepEqual(socket.sent, [{ token, submissionId: 'request-id', action: 'submit', submission: payload, idempotencyKey: 'request-id' }]);
  socket.message({ submissionId: 'server-id', status: 'running', output: 'Testing\n' });
  socket.message({ submissionId: '../../unsafe', status: 'success', metrics: { executionTime: 0.12 } });
  const result = await pending;
  assert.equal(result.status, 'success');
  assert.equal(result.submissionId, 'server-id');
  assert.deepEqual(result.events, seen);
  assert.equal(result.events.length, 2);
  assert.equal(socket.closed, true);
});

for (const status of ['test-case-failed', 'compilation-failed', 'timeout', 'out-of-memory', 'interrupted', 'output-exceeded', 'tampering-detected', 'error']) {
  test(`returns the ${status} terminal event without retrying`, async () => {
    const fake = transport();
    const pending = submitSolution({ token, payload, ...fake });
    fake.sockets[0].open();
    fake.sockets[0].message({ status, output: 'Failed' });
    assert.equal((await pending).status, status);
    assert.equal(fake.sockets.length, 1);
    assert.equal(fake.sockets[0].sent.length, 1);
  });
}

test('early close preserves uncertainty and never retries', async () => {
  const fake = transport();
  const pending = submitSolution({ token, payload, submissionId: 'request-id', ...fake });
  fake.sockets[0].open();
  fake.sockets[0].message({ submissionId: 'server-id', status: 'running' });
  fake.sockets[0].close();
  await assert.rejects(pending, error => error.sent === true && error.submissionId === 'server-id' && /before a final result/.test(error.message));
  assert.equal(fake.sockets.length, 1);
  assert.equal(fake.sockets[0].sent.length, 1);
});

test('timeout closes the connection and records whether anything was sent', async () => {
  for (const sent of [false, true]) {
    const fake = transport();
    const pending = submitSolution({ token, payload, submissionId: 'request-id', timeoutMs: 5, ...fake });
    if (sent) fake.sockets[0].open();
    await assert.rejects(pending, error => /Timed out/.test(error.message) && error.sent === sent && error.submissionId === 'request-id');
    assert.equal(fake.sockets[0].closed, true);
    assert.equal(fake.sockets.length, 1);
  }
});

test('an already-aborted submission never opens a connection or exposes the abort reason', async () => {
  const fake = transport();
  const controller = new AbortController();
  controller.abort(new Error(token));
  await assert.rejects(submitSolution({ token, payload, submissionId: 'request-id', ...fake, signal: controller.signal }), error => {
    assert.match(error.message, /canceled before connecting/u);
    assert.equal(error.message.includes(token), false);
    assert.equal(error.submissionId, 'request-id');
    assert.equal(error.sent, false);
    return true;
  });
  assert.equal(fake.sockets.length, 0);
  assert.equal(getEventListeners(controller.signal, 'abort').length, 0);
});

test('abort preserves the latest server ID, closes once and removes pending listeners without retries', async () => {
  for (const sent of [false, true]) {
    const fake = transport();
    const controller = new AbortController();
    const pending = submitSolution({ token, payload, submissionId: 'request-id', ...fake, signal: controller.signal, timeoutMs: 10 });
    const socket = fake.sockets[0];
    let closeCount = 0;
    const close = socket.close.bind(socket);
    socket.close = () => { closeCount++; close(); };
    if (sent) {
      socket.open();
      socket.message({ submissionId: 'server-id', status: 'running' });
    }
    assert.equal(getEventListeners(controller.signal, 'abort').length, 1);
    controller.abort(new Error(token));
    await assert.rejects(pending, error => {
      assert.match(error.message, /was interrupted/u);
      assert.equal(error.message.includes(token), false);
      assert.equal(error.submissionId, sent ? 'server-id' : 'request-id');
      assert.equal(error.sent, sent);
      return true;
    });
    assert.equal(getEventListeners(controller.signal, 'abort').length, 0);
    for (const name of ['open', 'message', 'error', 'close']) assert.equal(getEventListeners(socket, name).length, 0);
    socket.open();
    socket.message({ status: 'success' });
    await new Promise(resolve => setTimeout(resolve, 20));
    assert.equal(closeCount, 1);
    assert.equal(fake.sockets.length, 1);
    assert.equal(socket.sent.length, Number(sent));
  }
});

test('a finished submission removes its abort listener', async () => {
  const fake = transport();
  const controller = new AbortController();
  const pending = submitSolution({ token, payload, ...fake, signal: controller.signal });
  fake.sockets[0].open();
  fake.sockets[0].message({ status: 'success' });
  assert.equal((await pending).status, 'success');
  assert.equal(getEventListeners(controller.signal, 'abort').length, 0);
  controller.abort();
  assert.equal(fake.sockets[0].sent.length, 1);
});

test('removes secret fields recursively and redacts exact tokens before callbacks and storage', async () => {
  const fake = transport();
  const seen = [];
  const pending = submitSolution({ token, payload, ...fake, onEvent: event => seen.push(event) });
  fake.sockets[0].open();
  fake.sockets[0].message({
    status: 'success', token: 'different-token', access_token: 'different-access',
    nested: [{ refresh_token: 'different-refresh', AccessToken: token, output: `Bearer ${token} ${token}`, [token]: 'value' }],
  });
  const result = await pending;
  assert.deepEqual(result.events, [{ status: 'success', nested: [{ output: 'Bearer [REDACTED] [REDACTED]', '[REDACTED]': 'value' }] }]);
  assert.deepEqual(seen, result.events);
  assert.equal(JSON.stringify(result).includes(token), false);
});

test('malformed events and callback or transport errors expose no raw token details', async () => {
  for (const mode of ['callback', 'invalid-json', 'transport', 'send']) {
    const fake = transport();
    const pending = submitSolution({ token, payload, submissionId: 'request-id', ...fake, onEvent: () => { if (mode === 'callback') throw new Error(token); } });
    const socket = fake.sockets[0];
    if (mode === 'send') socket.send = () => { throw new Error(token); };
    socket.open();
    if (mode === 'callback') socket.message({ status: 'running' });
    if (mode === 'invalid-json') socket.dispatchEvent(new MessageEvent('message', { data: token }));
    if (mode === 'transport') socket.dispatchEvent(Object.assign(new Event('error'), { message: token }));
    await assert.rejects(pending, error => error.sent === true && error.submissionId === 'request-id' && !error.message.includes(token));
    assert.equal(fake.sockets.length, 1);
  }
});
