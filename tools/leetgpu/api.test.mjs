import test from 'node:test';
import assert from 'node:assert/strict';
import { LeetGpuApi, submissionRows, latestSubmission, formatSubmissions } from './api.mjs';

test('history request targets the official origin with filters and bearer auth', async () => {
  const payload = { submissions: [{ id: 'abc', runtime: 0 }], histogram: [1] };
  const api = new LeetGpuApi('test-token', async (url, options) => {
    assert.equal(url.origin, 'https://api.leetgpu.com');
    assert.equal(url.pathname, '/api/v1/submissions');
    assert.deepEqual(Object.fromEntries(url.searchParams), { challengeId: '1', language: 'cute', accelerator: 'A100-80GB' });
    assert.equal(options.headers.Authorization, 'Bearer test-token');
    assert.equal(options.method, 'GET');
    assert.equal(options.redirect, 'error');
    return Response.json(payload);
  });
  assert.deepEqual(await api.list({ challenge: 1, language: 'cute', gpu: 'A100-80GB' }), payload);
});

test('public reads never send the stored bearer token', async () => {
  const api = new LeetGpuApi('private-value', async (_, options) => {
    assert.equal(options.headers.Authorization, undefined);
    return Response.json({ challenges: [] });
  });
  await api.challenges();
});

test('requests reject external origins, missing auth, and malformed identifiers', async () => {
  let called = false;
  const api = new LeetGpuApi(null, async () => { called = true; });
  await assert.rejects(api.get('https://example.com/api/v1/submissions'), /Invalid/);
  await assert.rejects(api.get('//example.com/api/v1/submissions'), /Invalid/);
  await assert.rejects(api.latest(), /login/);
  assert.throws(() => api.code('../me'), /Invalid/);
  assert.throws(() => api.list({ challenge: 0, language: 'cute', gpu: 'H100' }), /positive integer/);
  assert.equal(called, false);
});

for (const status of [401, 403, 404, 429, 500, 302]) {
  test(`HTTP ${status} exposes no raw response content or credentials`, async () => {
    const api = new LeetGpuApi('private-value', async () => new Response('private-value', { status }));
    await assert.rejects(api.latest(), error => !error.message.includes('private-value') && error.status === status);
  });
}

test('handles empty history, no latest submission, malformed JSON and network errors', async () => {
  const empty = new LeetGpuApi('t', async () => new Response(null, { status: 204 }));
  assert.equal(await empty.latest(), null);
  assert.equal(latestSubmission(null), null);
  assert.equal(latestSubmission({ submission: null }), null);
  assert.deepEqual(submissionRows({ submissions: [] }), []);
  assert.deepEqual(submissionRows([]), []);
  assert.throws(() => submissionRows({ unexpected: [] }), /Unexpected/);
  const invalid = new LeetGpuApi('t', async () => new Response('not-json'));
  await assert.rejects(invalid.latest(), /invalid JSON/);
  const broken = new LeetGpuApi('private-value', async () => { throw new Error('private-value'); });
  await assert.rejects(broken.latest(), error => !error.message.includes('private-value'));
});

test('code preserves multiple files and JSON contents without executing or decoding', async () => {
  const files = [{ name: 'a.cu', content: 'int x = 1;\n' }, { name: '../b.h', content: 'hello' }];
  const api = new LeetGpuApi('t', async url => {
    assert.equal(url.pathname, '/api/v1/submissions/id-123/code');
    return Response.json({ files });
  });
  assert.deepEqual((await api.code('id-123')).files, files);
});

test('table preserves zero runtime and strips terminal control characters', () => {
  const text = formatSubmissions([{ id: 'a', status: '\x1b[31msuccess\n', metrics: { executionTime: 0, percentile: 99.58 } }], { challenge: 1 });
  assert.match(text, /0/);
  assert.match(text, /99\.58/);
  assert.match(text, /RUNTIME \(ms\)/);
  assert.ok(!text.includes('\x1b'));
});
