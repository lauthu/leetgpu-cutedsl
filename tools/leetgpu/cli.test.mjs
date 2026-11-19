import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import { mkdtemp, mkdir, writeFile, readFile, rm, readdir, stat } from 'node:fs/promises';
import { request } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const cliUrl = new URL('./cli.mjs', import.meta.url).href;
const expiry = Math.floor(Date.now() / 1000) + 3600;
const token = `header.${Buffer.from(JSON.stringify({ sub: 'cli-test-user', exp: expiry })).toString('base64url')}.signature`;
const session = { access_token: token, expires_at: expiry };

async function withFixture(action, { authenticated = true } = {}) {
  const directory = await mkdtemp(join(tmpdir(), 'leetgpu-cli-test-'));
  const config = join(directory, 'config');
  const legacyConfig = join(directory, 'legacy-config');
  const data = join(directory, 'data');
  await mkdir(config);
  await mkdir(legacyConfig);
  const sessionPath = join(config, 'session.json');
  // The file secret store keeps every test away from the real macOS Keychain.
  const env = { ...process.env, LEETGPU_CONFIG_DIR: config, LEETGPU_HISTORY_CONFIG_DIR: legacyConfig,
    LEETGPU_DATA_DIR: data, LEETGPU_HISTORY_DATA_DIR: join(directory, 'legacy-data'), LEETGPU_SECRET_STORE: 'file' };
  try {
    if (authenticated) await writeFile(sessionPath, JSON.stringify(session), { mode: 0o600 });
    return await action({ directory, config, legacyConfig, data, sessionPath, env });
  } finally { await rm(directory, { recursive: true, force: true }); }
}

function scriptFor(args, setup = '') {
  return `
    globalThis.fetch = async () => { throw new Error('Unexpected network request in test.'); };
    globalThis.WebSocket = class { constructor() { throw new Error('Unexpected WebSocket in test.'); } };
    ${setup}
    const { main } = await import(${JSON.stringify(cliUrl)});
    try { await main(${JSON.stringify(args)}); }
    catch (error) { console.error(error.message); process.exitCode = 1; }
  `;
}

function runFixture(fixture, args, { input = '', setup = '' } = {}) {
  const result = spawnSync(process.execPath, ['--input-type=module', '-e', scriptFor(args, setup)], {
    encoding: 'utf8', env: fixture.env, input, timeout: 10000,
  });
  assert.ifError(result.error);
  assert.equal(result.signal, null);
  assert.equal(result.stdout.includes(token), false);
  assert.equal(result.stderr.includes(token), false);
  return result;
}

async function run(args, payload, action) {
  return await withFixture(async fixture => {
    const result = runFixture(fixture, args, { setup: `globalThis.fetch = async () => Response.json(${JSON.stringify(payload)});` });
    if (action) await action(result);
    return result;
  });
}

test('CLI JSON preserves response metadata and is machine readable', async () => {
  const payload = { submissions: [], histogram: [1, 2] };
  const result = await run(['submissions', 'list', '--challenge', '1', '--language', 'cute', '--gpu', 'H100', '--json'], payload);
  assert.equal(result.status, 0);
  assert.deepEqual(JSON.parse(result.stdout), payload);
  assert.equal(result.stderr, '');
});

test('CLI refuses unsupported filters rather than silently ignoring them', async () => {
  const result = await run(['submissions', 'latest', '--challenge', '1'], {});
  assert.equal(result.status, 1);
  assert.match(result.stderr, /not supported/);
});

test('CLI preserves multi-file data and requires explicit file selection', async () => {
  const payload = { files: [{ name: 'a.cu', content: 'first\n' }, { name: 'b.h', content: 'second\n' }] };
  const ambiguous = await run(['submissions', 'code', 'id-1'], payload);
  assert.equal(ambiguous.status, 1);
  const selected = await run(['submissions', 'code', 'id-1', '--file', 'b.h'], payload);
  assert.equal(selected.status, 0);
  assert.equal(selected.stdout, 'second\n');
});

test('CLI cannot overwrite an existing destination', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'leetgpu-output-test-'));
  const output = join(directory, 'solution.cu');
  try {
    await writeFile(output, 'original');
    const result = await run(['submissions', 'code', 'id-1', '--output', output], { files: [{ name: '../ignored.cu', content: 'replacement' }] });
    assert.equal(result.status, 1);
    assert.equal(await readFile(output, 'utf8'), 'original');
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('stdin login saves only the access token and expiry in the preferred config directory', async () => {
  await withFixture(async fixture => {
    const legacyPath = join(fixture.legacyConfig, 'session.json');
    const legacyContents = JSON.stringify({ access_token: 'legacy-session-must-stay', expires_at: 1 });
    await writeFile(legacyPath, legacyContents);
    const input = JSON.stringify({ ...session, refresh_token: 'private-refresh-value', user: { email: 'private@example.test' } });
    const result = runFixture(fixture, ['login', '--stdin'], { input });
    assert.equal(result.status, 0, result.stderr);
    assert.deepEqual(JSON.parse(await readFile(fixture.sessionPath, 'utf8')), session);
    assert.equal(await readFile(legacyPath, 'utf8'), legacyContents);
    assert.equal((await stat(fixture.sessionPath)).mode & 0o777, 0o600);
    assert.equal((await stat(fixture.config)).mode & 0o777, 0o700);
    assert.equal(result.stdout.includes('private-refresh-value'), false);
    assert.equal(result.stderr.includes('private-refresh-value'), false);
    const status = runFixture(fixture, ['status', '--json']);
    assert.equal(status.status, 0);
    assert.deepEqual(JSON.parse(status.stdout), {
      authenticated: true, expiresAt: new Date(expiry * 1000).toISOString(), sessionPath: fixture.sessionPath, renewable: false,
    });
  }, { authenticated: false });
});

test('stdin login accepts raw JWT and refuses invalid or expired input without writing it', async () => {
  await withFixture(async fixture => {
    const valid = runFixture(fixture, ['login', '--stdin'], { input: `${token}\n` });
    assert.equal(valid.status, 0, valid.stderr);
    assert.deepEqual(JSON.parse(await readFile(fixture.sessionPath, 'utf8')), session);
    const expired = `header.${Buffer.from(JSON.stringify({ sub: 'cli-test-user', exp: 1 })).toString('base64url')}.signature`;
    for (const input of ['invalid-secret-input', expired]) {
      const invalid = runFixture(fixture, ['login', '--stdin'], { input });
      assert.equal(invalid.status, 1);
      assert.match(invalid.stderr, /invalid or expired/u);
      assert.equal(invalid.stderr.includes(input), false);
      assert.deepEqual(JSON.parse(await readFile(fixture.sessionPath, 'utf8')), session);
    }
  }, { authenticated: false });
});

test('status JSON distinguishes missing and expired or invalid sessions with exit code one', async () => {
  await withFixture(async fixture => {
    const missing = runFixture(fixture, ['status', '--json']);
    assert.equal(missing.status, 1);
    assert.deepEqual(JSON.parse(missing.stdout), { authenticated: false, expiresAt: null, reason: 'missing', renewable: false, sessionPath: fixture.sessionPath });
    assert.equal(missing.stderr, '');
    for (const saved of [JSON.stringify({ access_token: 'expired-secret', expires_at: 1 }), '{invalid-secret']) {
      await writeFile(fixture.sessionPath, saved);
      const invalid = runFixture(fixture, ['status', '--json']);
      assert.equal(invalid.status, 1);
      assert.deepEqual(JSON.parse(invalid.stdout), { authenticated: false, expiresAt: null, reason: 'expired-or-invalid', renewable: false, sessionPath: fixture.sessionPath });
      assert.equal(invalid.stderr, '');
      assert.equal(invalid.stdout.includes('secret'), false);
    }
  }, { authenticated: false });
});

test('login modes are mutually exclusive before any browser or network action', async () => {
  await withFixture(async fixture => {
    const modes = ['--browser', '--existing-browser', '--stdin', '--no-browser'];
    for (let first = 0; first < modes.length; first += 1) {
      for (let second = first + 1; second < modes.length; second += 1) {
        const result = runFixture(fixture, ['login', modes[first], modes[second]]);
        assert.equal(result.status, 1);
        assert.match(result.stderr, /one|exclusive/u);
      }
    }
  });
});

test('login port rejects malformed values and modes without a manual receiver', async () => {
  await withFixture(async fixture => {
    for (const port of ['-1', '65536', '1.5', 'nope', 'Infinity']) {
      const result = runFixture(fixture, ['login', '--no-browser', `--port=${port}`]);
      assert.equal(result.status, 1);
      assert.match(result.stderr, /port/u);
    }
    for (const mode of ['--stdin', '--browser', '--existing-browser']) {
      const result = runFixture(fixture, ['login', mode, '--port', '8787']);
      assert.equal(result.status, 1);
      assert.match(result.stderr, /port/u);
    }
  });
});

test('no-browser CLI receives a synthetic session over loopback and prints SSH forwarding instructions', async () => {
  await withFixture(async fixture => {
    const child = spawn(process.execPath, ['--input-type=module', '-e', scriptFor(['login', '--no-browser', '--port', '0'])], {
      env: fixture.env, stdio: ['ignore', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    let submitted = false;
    let submitPromise;
    child.stdout.setEncoding('utf8');
    child.stderr.setEncoding('utf8');
    child.stdout.on('data', chunk => { stdout += chunk; });
    child.stderr.on('data', chunk => {
      stderr += chunk;
      const match = stderr.match(/http:\/\/127\.0\.0\.1:\d+\/[a-f0-9]{64}/u);
      if (submitted || !match) return;
      submitted = true;
      submitPromise = new Promise((resolve, reject) => {
        const req = request(match[0], { method: 'POST', headers: { 'Content-Type': 'application/json', Origin: new URL(match[0]).origin } }, response => {
          response.resume();
          response.on('end', () => resolve(response.statusCode));
        });
        req.on('error', reject);
        req.end(JSON.stringify({ value: JSON.stringify(session) }));
      });
      // Preserve rejection for the assertion below without an unhandled promise.
      submitPromise.catch(() => {});
    });
    const timeout = setTimeout(() => child.kill('SIGKILL'), 10000);
    try {
      const result = await new Promise((resolve, reject) => {
        child.once('error', reject);
        child.once('close', (code, signal) => resolve({ code, signal }));
      });
      assert.deepEqual(result, { code: 0, signal: null }, stderr);
      assert.equal(await submitPromise, 200);
      assert.match(stderr, /ssh -N -L 127\.0\.0\.1:(\d+):127\.0\.0\.1:\1/u);
      assert.equal(stdout.includes(token), false);
      assert.equal(stderr.includes(token), false);
      assert.deepEqual(JSON.parse(await readFile(fixture.sessionPath, 'utf8')), session);
    } finally {
      clearTimeout(timeout);
      if (child.exitCode === null && child.signalCode === null) child.kill('SIGKILL');
    }
  }, { authenticated: false });
});

function fakeSubmissionSetup(outcome, countsPath) {
  const interrupted = outcome.startsWith('SIG');
  return `
    const { writeFileSync } = await import('node:fs');
    let connections = 0;
    let sends = 0;
    globalThis.fetch = async () => Response.json({ accelerators: ['H100'], supportedLanguages: { H100: ['cute'] } });
    globalThis.WebSocket = class extends EventTarget {
      constructor() { super(); connections += 1; queueMicrotask(() => this.dispatchEvent(new Event('open'))); }
      send(data) {
        sends += 1;
        const request = JSON.parse(data);
        writeFileSync(${JSON.stringify(countsPath)}, JSON.stringify({ connections, sends,
          accelerator: request.submission.accelerator, language: request.submission.language, public: request.submission.public }));
        queueMicrotask(() => {
          this.dispatchEvent(new MessageEvent('message', { data: JSON.stringify({
            submissionId: 'server-submission-id', status: ${JSON.stringify(outcome === 'uncertain' || interrupted ? 'running' : outcome)},
            message: 'echo ' + request.token, access_token: request.token,
          }) }));
          if (${JSON.stringify(outcome)} === 'uncertain') this.dispatchEvent(new Event('close'));
          if (${interrupted}) process.kill(process.pid, ${JSON.stringify(outcome)});
        });
      }
      close() {}
    };
  `;
}

for (const [outcome, exitCode] of [['success', 0], ['compilation-failed', 2], ['uncertain', 1], ['SIGINT', 1], ['SIGTERM', 1], ['SIGHUP', 1]]) {
  test(`CLI submit ${outcome} records one private attempt without credentials and exits ${exitCode}`, async () => {
    await withFixture(async fixture => {
      const sourcePath = join(fixture.directory, 'solution.py');
      const countsPath = join(fixture.directory, 'attempts.json');
      await writeFile(sourcePath, '# synthetic submission source\n');
      const result = runFixture(fixture, ['submissions', 'submit', '--challenge', '1', '--language', 'cute', '--gpu', 'H100', '--file', sourcePath, '--json'], {
        setup: fakeSubmissionSetup(outcome, countsPath),
      });
      assert.equal(result.status, exitCode, result.stderr);
      assert.deepEqual(JSON.parse(await readFile(countsPath, 'utf8')), { connections: 1, sends: 1, accelerator: 'H100', language: 'cute', public: false });
      const receiptDirectory = join(fixture.data, 'receipts');
      const receipts = await readdir(receiptDirectory);
      assert.equal(receipts.length, 1);
      const receiptPath = join(receiptDirectory, receipts[0]);
      const rawReceipt = await readFile(receiptPath, 'utf8');
      assert.equal(rawReceipt.includes(token), false);
      assert.equal(rawReceipt.includes('access_token'), false);
      const receipt = JSON.parse(rawReceipt);
      assert.equal(receipt.submissionId, 'server-submission-id');
      const uncertain = outcome === 'uncertain' || outcome.startsWith('SIG');
      assert.equal(receipt.status, uncertain ? 'unconfirmed' : outcome);
      assert.equal(receipt.public, false);
      assert.equal((await stat(receiptPath)).mode & 0o777, 0o600);
      if (uncertain) {
        assert.equal(receipt.sent, true);
        assert.match(result.stderr, /Check history before retrying/u);
      } else {
        const output = JSON.parse(result.stdout);
        assert.equal(output.receiptPath, receiptPath);
        assert.equal(output.status, outcome);
        assert.equal(output.events[0].message, 'echo [REDACTED]');
      }
    });
  });
}

const challengeList = { challenges: [{ id: 1, title: 'Vector Addition' }, { id: 2, title: 'Matrix Multiplication' }] };

// Serves the public challenge list and accelerator settings, and records what the socket was asked to do.
function fakePlatformSetup(status, recordPath) {
  return `
    const { writeFileSync } = await import('node:fs');
    globalThis.fetch = async url => Response.json(String(url).endsWith('/challenges')
      ? ${JSON.stringify(challengeList)} : { accelerators: ['H100'], supportedLanguages: { H100: ['cute'] } });
    globalThis.WebSocket = class extends EventTarget {
      constructor() { super(); queueMicrotask(() => this.dispatchEvent(new Event('open'))); }
      send(data) {
        const request = JSON.parse(data);
        writeFileSync(${JSON.stringify(recordPath)}, JSON.stringify({ action: request.action, challengeId: request.submission.challengeId }));
        const emit = event => this.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(event) }));
        queueMicrotask(() => {
          emit({ status: 'running', type: 'stdout', output: 'Spinning up H100...\\n' });
          emit({ status: 'running', type: 'stdout', output: 'partial ' });
          emit({ status: ${JSON.stringify(status)}, type: 'stdout', output: '\\u001b[32m\\u2713 done\\u001b[0m\\r\\n' });
        });
      }
      close() {}
    };
  `;
}

for (const [status, exitCode] of [['success', 0], ['compilation-failed', 2]]) {
  test(`CLI run ${status} sends a run action, streams plain output, writes no receipt and exits ${exitCode}`, async () => {
    await withFixture(async fixture => {
      const sourcePath = join(fixture.directory, 'solution.py');
      const recordPath = join(fixture.directory, 'request.json');
      await writeFile(sourcePath, '# synthetic run source\n');
      const result = runFixture(fixture, ['submissions', 'run', '--challenge', '2', '--language', 'cute', '--gpu', 'H100', '--file', sourcePath], {
        setup: fakePlatformSetup(status, recordPath),
      });
      assert.equal(result.status, exitCode, result.stderr);
      assert.deepEqual(JSON.parse(await readFile(recordPath, 'utf8')), { action: 'run', challengeId: 2 });
      assert.equal(result.stdout, `Run finished: ${status}\n`);
      // One status line per change, platform text without ANSI or carriage returns, and no half-open line.
      assert.match(result.stderr, /\[running\]\nSpinning up H100\.\.\.\npartial \n\[[a-z-]+\]\n✓ done\n$/u);
      assert.equal(/\u001b|\r/u.test(result.stderr), false);
      await assert.rejects(stat(join(fixture.data, 'receipts')), { code: 'ENOENT' });
    });
  });
}

test('CLI run --json keeps stdout machine readable and omits platform text from stderr', async () => {
  await withFixture(async fixture => {
    const sourcePath = join(fixture.directory, 'solution.py');
    await writeFile(sourcePath, '# synthetic run source\n');
    const result = runFixture(fixture, ['submissions', 'run', '--challenge', '2', '--language', 'cute', '--gpu', 'H100', '--file', sourcePath, '--json'], {
      setup: fakePlatformSetup('success', join(fixture.directory, 'request.json')),
    });
    assert.equal(result.status, 0, result.stderr);
    const output = JSON.parse(result.stdout);
    assert.deepEqual(Object.keys(output), ['status', 'events']);
    assert.equal(output.events.length, 3);
    assert.equal(result.stderr.includes('Spinning up'), false);
  });
});

test('CLI submit infers the challenge from the nearest matching directory and names it', async () => {
  await withFixture(async fixture => {
    const directory = join(fixture.directory, 'solutions', 'matrix-multiplication', 'drafts');
    await mkdir(directory, { recursive: true });
    const sourcePath = join(directory, 'v1.py');
    const recordPath = join(fixture.directory, 'request.json');
    await writeFile(sourcePath, '# synthetic submission source\n');
    const result = runFixture(fixture, ['submissions', 'submit', '--language', 'cute', '--gpu', 'H100', '--file', sourcePath, '--json'], {
      setup: fakePlatformSetup('success', recordPath),
    });
    assert.equal(result.status, 0, result.stderr);
    assert.deepEqual(JSON.parse(await readFile(recordPath, 'utf8')), { action: 'submit', challengeId: 2 });
    assert.match(result.stderr, /Submitting challenge 2 \(Matrix Multiplication\) \/ cute \/ H100/u);
    assert.equal((await readdir(join(fixture.data, 'receipts'))).length, 1);
  });
});

test('CLI refuses to guess a challenge and opens no connection when nothing matches', async () => {
  await withFixture(async fixture => {
    const sourcePath = join(fixture.directory, 'solution.py');
    const recordPath = join(fixture.directory, 'request.json');
    await writeFile(sourcePath, '# synthetic submission source\n');
    for (const extra of [[], ['--challenge', 'no-such-challenge'], ['--challenge', '0']]) {
      const result = runFixture(fixture, ['submissions', 'submit', '--language', 'cute', '--gpu', 'H100', '--file', sourcePath, ...extra], {
        setup: fakePlatformSetup('success', recordPath),
      });
      assert.equal(result.status, 1);
      assert.match(result.stderr, /challenge/iu);
      await assert.rejects(stat(recordPath), { code: 'ENOENT' });
      await assert.rejects(stat(join(fixture.data, 'receipts')), { code: 'ENOENT' });
    }
  });
});

test('CLI list accepts a title slug and queries the resolved numeric ID', async () => {
  await withFixture(async fixture => {
    const urlsPath = join(fixture.directory, 'urls.json');
    const result = runFixture(fixture, ['submissions', 'list', '--challenge', 'Matrix Multiplication', '--language', 'cute', '--gpu', 'H100', '--json'], {
      setup: `
        const { writeFileSync } = await import('node:fs');
        const urls = [];
        globalThis.fetch = async (url, init) => {
          urls.push({ url: String(url), authorized: Boolean(init.headers.Authorization) });
          writeFileSync(${JSON.stringify(urlsPath)}, JSON.stringify(urls));
          return Response.json(String(url).endsWith('/challenges') ? ${JSON.stringify(challengeList)} : { submissions: [] });
        };
      `,
    });
    assert.equal(result.status, 0, result.stderr);
    assert.deepEqual(JSON.parse(await readFile(urlsPath, 'utf8')), [
      { url: 'https://api.leetgpu.com/api/v1/challenges', authorized: false },
      { url: 'https://api.leetgpu.com/api/v1/submissions?challengeId=2&language=cute&accelerator=H100', authorized: true },
    ]);
  });
});

const refreshPath = fixture => join(fixture.config, 'refresh-token');
const renewedExpiry = expiry + 7200;
const renewedToken = `header.${Buffer.from(JSON.stringify({ sub: 'cli-test-user', exp: renewedExpiry })).toString('base64url')}.renewed`;

// The sign-in server answers the refresh exchange; every other request is the LeetGPU API.
function renewalSetup(status, logPath) {
  return `
    const { writeFileSync } = await import('node:fs');
    const log = [];
    globalThis.fetch = async (url, init) => {
      const signIn = String(url).startsWith('https://yhdtysacdkqoquvkdwdd.supabase.co/');
      log.push({ url: String(url), bearer: signIn ? 'sign-in' : init.headers.Authorization, body: signIn ? JSON.parse(init.body) : null });
      writeFileSync(${JSON.stringify(logPath)}, JSON.stringify(log));
      if (!signIn) return Response.json({ submissions: [] });
      if (${status} !== 200) return new Response('{"msg":"private-server-detail"}', { status: ${status} });
      return Response.json({ access_token: ${JSON.stringify(renewedToken)}, expires_at: ${renewedExpiry}, refresh_token: 'rotated-refresh-token', user: { email: 'private@example.test' } });
    };
  `;
}

test('persistent stdin login keeps the refresh token apart from the session file and never prints it', async () => {
  await withFixture(async fixture => {
    const input = JSON.stringify({ ...session, refresh_token: 'first-refresh-token', user: { email: 'private@example.test' } });
    const result = runFixture(fixture, ['login', '--stdin', '--persist'], { input });
    assert.equal(result.status, 0, result.stderr);
    assert.match(result.stdout, /staying signed in/u);
    assert.equal(`${result.stdout}${result.stderr}`.includes('first-refresh-token'), false);
    assert.deepEqual(JSON.parse(await readFile(fixture.sessionPath, 'utf8')), session);
    assert.equal(await readFile(refreshPath(fixture), 'utf8'), 'first-refresh-token\n');
    assert.equal((await stat(refreshPath(fixture))).mode & 0o777, 0o600);
    const status = runFixture(fixture, ['status', '--json']);
    assert.equal(status.status, 0);
    assert.equal(JSON.parse(status.stdout).renewable, true);
  }, { authenticated: false });
});

test('persistent login needs a full session of its own and a place to keep the secret', async () => {
  await withFixture(async fixture => {
    const jwtOnly = runFixture(fixture, ['login', '--stdin', '--persist'], { input: `${token}\n` });
    assert.equal(jwtOnly.status, 1);
    assert.match(jwtOnly.stderr, /full session JSON including refresh_token/u);
    const shared = runFixture(fixture, ['login', '--existing-browser', '--persist']);
    assert.equal(shared.status, 1);
    assert.match(shared.stderr, /cannot share the session of an open browser tab/u);
    const noStore = spawnSync(process.execPath, ['--input-type=module', '-e', scriptFor(['login', '--stdin', '--persist'])], {
      encoding: 'utf8', env: { ...fixture.env, LEETGPU_SECRET_STORE: 'none' }, input: JSON.stringify({ ...session, refresh_token: 'first-refresh-token' }), timeout: 10000,
    });
    assert.equal(noStore.status, 1);
    assert.match(noStore.stderr, /needs a secret store/u);
    for (const path of [fixture.sessionPath, refreshPath(fixture)]) await assert.rejects(stat(path), { code: 'ENOENT' });
  }, { authenticated: false });
});

test('an expired access token renews before the request and the API sees only the new token', async () => {
  await withFixture(async fixture => {
    const logPath = join(fixture.directory, 'requests.json');
    await writeFile(fixture.sessionPath, JSON.stringify({ access_token: 'expired-access', expires_at: 1 }), { mode: 0o600 });
    await writeFile(refreshPath(fixture), 'first-refresh-token\n', { mode: 0o600 });
    const before = runFixture(fixture, ['status', '--json']);
    assert.equal(before.status, 0);
    assert.deepEqual(JSON.parse(before.stdout), { authenticated: false, expiresAt: null, sessionPath: fixture.sessionPath, renewable: true, reason: 'expired-or-invalid' });
    const result = runFixture(fixture, ['submissions', 'list', '--challenge', '1', '--language', 'cute', '--gpu', 'H100', '--json'], { setup: renewalSetup(200, logPath) });
    assert.equal(result.status, 0, result.stderr);
    assert.deepEqual(JSON.parse(result.stdout), { submissions: [] });
    assert.deepEqual(JSON.parse(await readFile(logPath, 'utf8')), [
      { url: 'https://yhdtysacdkqoquvkdwdd.supabase.co/auth/v1/token?grant_type=refresh_token', bearer: 'sign-in', body: { refresh_token: 'first-refresh-token' } },
      { url: 'https://api.leetgpu.com/api/v1/submissions?challengeId=1&language=cute&accelerator=H100', bearer: `Bearer ${renewedToken}`, body: null },
    ]);
    assert.deepEqual(JSON.parse(await readFile(fixture.sessionPath, 'utf8')), { access_token: renewedToken, expires_at: renewedExpiry });
    assert.equal(await readFile(refreshPath(fixture), 'utf8'), 'rotated-refresh-token\n');
    for (const text of [result.stdout, result.stderr]) assert.equal(/refresh-token|renewed|private@/u.test(text), false);
    const after = JSON.parse(runFixture(fixture, ['status', '--json']).stdout);
    assert.equal(after.authenticated, true);
    assert.equal(after.renewable, true);
  }, { authenticated: false });
});

test('a rejected renewal asks for a new login, forgets the dead token and sends nothing to the API', async () => {
  await withFixture(async fixture => {
    const logPath = join(fixture.directory, 'requests.json');
    await writeFile(fixture.sessionPath, JSON.stringify({ access_token: 'expired-access', expires_at: 1 }), { mode: 0o600 });
    await writeFile(refreshPath(fixture), 'first-refresh-token\n', { mode: 0o600 });
    const result = runFixture(fixture, ['submissions', 'latest', '--json'], { setup: renewalSetup(400, logPath) });
    assert.equal(result.status, 1);
    assert.match(result.stderr, /no longer valid\. Run login again/u);
    assert.equal(result.stderr.includes('private-server-detail'), false);
    assert.equal(JSON.parse(await readFile(logPath, 'utf8')).length, 1);
    await assert.rejects(stat(refreshPath(fixture)), { code: 'ENOENT' });
    assert.equal(runFixture(fixture, ['status', '--json']).status, 1);
  }, { authenticated: false });
});

test('a one-hour login replaces an earlier persistent sign-in, and logout removes both secrets', async () => {
  await withFixture(async fixture => {
    await writeFile(refreshPath(fixture), 'first-refresh-token\n', { mode: 0o600 });
    const hourly = runFixture(fixture, ['login', '--stdin'], { input: JSON.stringify({ ...session, refresh_token: 'unrelated-refresh-token' }) });
    assert.equal(hourly.status, 0, hourly.stderr);
    assert.match(hourly.stderr, /Removed the previously saved persistent sign-in/u);
    await assert.rejects(stat(refreshPath(fixture)), { code: 'ENOENT' });
    assert.deepEqual(JSON.parse(await readFile(fixture.sessionPath, 'utf8')), session);
    await writeFile(refreshPath(fixture), 'first-refresh-token\n', { mode: 0o600 });
    const out = runFixture(fixture, ['logout']);
    assert.equal(out.status, 0, out.stderr);
    for (const path of [fixture.sessionPath, refreshPath(fixture)]) await assert.rejects(stat(path), { code: 'ENOENT' });
    assert.equal(runFixture(fixture, ['status', '--json']).status, 1);
  }, { authenticated: false });
});

test('CLI reads the challenge ID from an official folder name and refuses an unlisted one', async () => {
  await withFixture(async fixture => {
    const recordPath = join(fixture.directory, 'request.json');
    // The name after the prefix deliberately differs from the title slug.
    const directory = join(fixture.directory, 'solutions', '1_vector_add');
    await mkdir(directory, { recursive: true });
    const sourcePath = join(directory, 'v2_vectorized.py');
    await writeFile(sourcePath, '# synthetic run source\n');
    const result = runFixture(fixture, ['submissions', 'run', '--language', 'cute', '--gpu', 'H100', '--file', sourcePath], {
      setup: fakePlatformSetup('success', recordPath),
    });
    assert.equal(result.status, 0, result.stderr);
    assert.deepEqual(JSON.parse(await readFile(recordPath, 'utf8')), { action: 'run', challengeId: 1 });
    assert.match(result.stderr, /Running challenge 1 \(Vector Addition\) \/ cute \/ H100/u);
    await rm(recordPath);
    const unlisted = join(fixture.directory, 'solutions', '999_not_listed');
    await mkdir(unlisted);
    await writeFile(join(unlisted, 'v1.py'), '# synthetic run source\n');
    const refused = runFixture(fixture, ['submissions', 'submit', '--language', 'cute', '--gpu', 'H100', '--file', join(unlisted, 'v1.py')], {
      setup: fakePlatformSetup('success', recordPath),
    });
    assert.equal(refused.status, 1);
    assert.match(refused.stderr, /Challenge 999 \(from 999_not_listed\) is not in the LeetGPU challenge list/u);
    await assert.rejects(stat(recordPath), { code: 'ENOENT' });
    await assert.rejects(stat(join(fixture.data, 'receipts')), { code: 'ENOENT' });
  });
});
