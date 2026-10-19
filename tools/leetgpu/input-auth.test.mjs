import test from 'node:test';
import assert from 'node:assert/strict';
import { Readable } from 'node:stream';
import { readSessionInput } from './input-auth.mjs';

const expiry = Math.floor(Date.now() / 1000) + 3600;
const token = `header.${Buffer.from(JSON.stringify({ exp: expiry })).toString('base64url')}.signature`;
const normalized = { access_token: token, expires_at: expiry };

test('raw JWT from a pipe is normalized without persistence', async () => {
  assert.deepEqual(await readSessionInput(Readable.from([` \n${token}\n`])), normalized);
});

test('full JSON can arrive in byte chunks and drops all extra session fields', async () => {
  const session = JSON.stringify({ ...normalized, refresh_token: 'private-refresh', user: { email: 'private@example.test', name: '测试' } });
  const bytes = Buffer.from(session);
  const chunks = Array.from(bytes, byte => Buffer.from([byte]));
  assert.deepEqual(await readSessionInput(Readable.from(chunks)), normalized);
});

test('exact input byte limit is accepted and oversized input is rejected', async () => {
  const padded = token.padEnd(100000, ' ');
  assert.deepEqual(await readSessionInput(Readable.from([padded])), normalized);
  await assert.rejects(readSessionInput(Readable.from([padded, ' '])), /exceeds 100000 bytes/u);
  await assert.rejects(readSessionInput(Readable.from(['秘'.repeat(40000)])), /exceeds 100000 bytes/u);
});

test('invalid and expired input errors never contain pasted values', async () => {
  const secret = 'session-secret-must-stay-private';
  const expiredToken = `header.${Buffer.from(JSON.stringify({ exp: 1 })).toString('base64url')}.signature`;
  for (const input of ['', secret, `{broken:${secret}`, expiredToken, JSON.stringify({ access_token: secret, expires_at: 1 })]) {
    await assert.rejects(readSessionInput(Readable.from([input])), error => {
      assert.match(error.message, /session input is invalid or expired/u);
      assert.equal(error.message.includes(secret), false);
      assert.equal(error.message.includes(expiredToken), false);
      return true;
    });
  }
});

test('interactive terminal input is refused without consuming or echoing credentials', async () => {
  let read = false;
  const stream = {
    isTTY: true,
    async *[Symbol.asyncIterator]() { read = true; yield token; },
  };
  await assert.rejects(readSessionInput(stream), /pipe or file, not an interactive terminal/u);
  assert.equal(read, false);
});

test('stream failures are sanitized instead of printing source error details', async () => {
  const secret = 'session-stream-secret';
  const stream = Readable.from((async function* () {
    yield token.slice(0, 10);
    throw new Error(secret);
  })());
  await assert.rejects(readSessionInput(stream), error => {
    assert.equal(error.message, 'Could not read the LeetGPU session from standard input.');
    assert.equal(error.message.includes(secret), false);
    return true;
  });
});

test('unsupported stream data gets a generic error', async () => {
  await assert.rejects(readSessionInput(Readable.from([{ access_token: token }])), error => {
    assert.equal(error.message, 'Could not read the LeetGPU session from standard input.');
    assert.equal(error.message.includes(token), false);
    return true;
  });
});
