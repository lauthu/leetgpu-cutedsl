import test from 'node:test';
import assert from 'node:assert/strict';
import { existingChromeSession } from './chrome-session.mjs';

test('existing Chrome capture returns only bearer and expiry', async () => {
  const expires_at = Math.floor(Date.now() / 1000) + 3600;
  const result = await existingChromeSession({ platformName: 'darwin', run: async (command, args) => {
    assert.equal(command, 'osascript');
    assert.match(args[1], /location.origin !== 'https:\/\/leetgpu.com'/);
    assert.match(args[1], /sb-yhdtysacdkqoquvkdwdd-auth-token/);
    return { stdout: JSON.stringify({ access_token: 'fake-test-token', expires_at, refresh_token: 'never-save' }) };
  } });
  assert.deepEqual(result, { access_token: 'fake-test-token', expires_at });
});

test('existing Chrome errors never expose captured subprocess credentials', async () => {
  for (const stderr of ['JavaScript from Apple Events is disabled', 'Not authorized (-1743)', 'fake-secret']) {
    await assert.rejects(existingChromeSession({ platformName: 'darwin', run: async () => {
      throw Object.assign(new Error('fake-secret'), { stdout: 'fake-secret', stderr });
    } }), error => !error.message.includes('fake-secret'));
  }
});

test('missing or expired Chrome sessions require login', async () => {
  for (const stdout of ['null', JSON.stringify({ access_token: 'fake-test-token', expires_at: 1 })]) {
    await assert.rejects(existingChromeSession({ platformName: 'darwin', run: async () => ({ stdout }) }), /No valid LeetGPU session/);
  }
});
