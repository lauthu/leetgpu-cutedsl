#!/usr/bin/env node
import { parseArgs } from 'node:util';
import { pathToFileURL } from 'node:url';
import { writeFile, readFile, mkdir, open } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { randomUUID, createHash } from 'node:crypto';
import { LeetGpuApi, submissionRows, latestSubmission, formatSubmissions, table } from './api.mjs';
import { browserLogin, saveSession, loadSession, logout, sessionPath } from './auth.mjs';
import { dataDirectory } from './paths.mjs';

const HELP = `Local LeetGPU submission helper (unofficial)

Usage: ./tools/leetgpu/leetgpu COMMAND

  login                        Open a local session-import page in your normal browser
  login --existing-browser     Read only an already-open LeetGPU tab (macOS Chrome)
  login --browser              Try automatic capture using a separate Chrome window
  login --no-browser --port N   Serve a loopback import page for an SSH tunnel
  login --stdin                Import a session from a pipe/file without echoing it
  status [--json]              Check local session expiry; exit 1 if missing/expired
  logout                       Delete the locally saved access token
  challenges [--json]          List public challenge IDs and titles
  gpus [--json]                List accelerator names and supported languages
  submissions latest [--json]
  submissions list --challenge ID --language LANGUAGE --gpu GPU [--json]
  submissions code ID [--json] [--file NAME] [--output NEW_FILE]
  submissions submit --challenge ID --language LANGUAGE --gpu GPU --file SOURCE [--json]

Languages: cuda, cute, triton, pytorch, jax, mojo
GPU examples: T4, A100-80GB, H100, H200, B200

For read commands, --json emits the unmodified response including metadata.
For submit, --json emits the status, sanitized events, and receipt path.
code prints a single file; for multiple files use --file NAME or --json.
--output writes a new local file and refuses to overwrite an existing file.
List results reflect the server response; complete history/pagination is not assumed.
submit sends one private submission, writes a local receipt, and never retries automatically.
`;

export async function main(args = process.argv.slice(2)) {
  const { positionals, values } = parseArgs({ args, allowPositionals: true, strict: true, options: {
    help: { type: 'boolean', short: 'h' }, json: { type: 'boolean' }, browser: { type: 'boolean' },
    'existing-browser': { type: 'boolean' }, stdin: { type: 'boolean' },
    'no-browser': { type: 'boolean' }, port: { type: 'string' },
    challenge: { type: 'string' }, language: { type: 'string' }, gpu: { type: 'string' },
    file: { type: 'string' }, output: { type: 'string' },
  } });
  if (values.help || !positionals.length) { console.log(HELP); return; }
  const [command, subcommand, id] = positionals;
  const signature = command === 'submissions' ? `${command} ${subcommand}` : command;
  const allowed = {
    login: ['browser', 'existing-browser', 'stdin', 'no-browser', 'port'], status: ['json'], logout: [], challenges: ['json'], gpus: ['json'],
    'submissions latest': ['json'],
    'submissions list': ['challenge', 'language', 'gpu', 'json'],
    'submissions code': ['json', 'file', 'output'],
    'submissions submit': ['challenge', 'language', 'gpu', 'file', 'json'],
  };
  if (!Object.hasOwn(allowed, signature)) throw new Error('Unknown command. Use --help.');
  const expectedCount = signature === 'submissions code' ? 3 : command === 'submissions' ? 2 : 1;
  if (positionals.length !== expectedCount) throw new Error('Incorrect arguments. Use --help.');
  for (const key of Object.keys(values)) {
    if (!allowed[signature].includes(key)) throw new Error(`--${key} is not supported for ${signature}.`);
  }
  if (command === 'login') {
    if (['browser', 'existing-browser', 'stdin', 'no-browser'].filter(key => values[key]).length > 1) {
      throw new Error('Choose one login mode.');
    }
    if (values.port !== undefined && (values.browser || values['existing-browser'] || values.stdin)) {
      throw new Error('--port is only supported for the local import page.');
    }
    const port = values.port === undefined ? 0 : Number(values.port);
    if (values.port !== undefined && (!/^\d+$/u.test(values.port) || !Number.isInteger(port) || port > 65535)) {
      throw new Error('--port must be an integer between 0 and 65535.');
    }
    const login = values['existing-browser'] ? (await import('./chrome-session.mjs')).existingChromeSession
      : values.browser ? browserLogin : (await import('./manual-auth.mjs')).manualLogin;
    const session = values.stdin ? await (await import('./input-auth.mjs')).readSessionInput()
      : await login({ onStatus: message => console.error(message), openBrowser: !values['no-browser'], port });
    await saveSession(session);
    console.log(`Signed in. Access token saved locally; expires ${new Date(session.expires_at * 1000).toISOString()}.`);
    return;
  }
  if (command === 'logout') { await logout(); console.log('Local access token deleted.'); return; }
  if (command === 'status') {
    let session;
    let reason = 'missing';
    try { session = await loadSession(); } catch (error) {
      if (error.code !== 'SESSION_INVALID') throw error;
      reason = 'expired-or-invalid';
    }
    const expiresAt = session ? new Date(session.expires_at * 1000).toISOString() : null;
    if (values.json) printJson({ authenticated: Boolean(session), expiresAt, sessionPath, ...(!session && { reason }) });
    else console.log(session ? `Signed in locally; expires ${expiresAt}.\nSession file: ${sessionPath}` : 'No valid local session. Run login.');
    if (!session) process.exitCode = 1;
    return;
  }
  const isPublic = command === 'challenges' || command === 'gpus';
  const session = isPublic ? null : await loadSession();
  const api = new LeetGpuApi(session?.access_token);
  if (subcommand === 'submit') {
    if (!session) throw new Error('Please run login first.');
    if (!values.file) throw new Error('--file must point to the source file to submit.');
    const { buildSubmission, submitSolution } = await import('./submit.mjs');
    const sourcePath = resolve(values.file);
    const source = await readFile(sourcePath, 'utf8');
    const names = { cuda: 'solution.cu', cute: 'solution.py', triton: 'solution.py', pytorch: 'solution.py', jax: 'solution.py', mojo: 'solution.mojo' };
    const payload = buildSubmission({ token: session.access_token, ...values,
      files: [{ name: names[values.language], content: source }] });
    const available = await api.accelerators();
    if (!available.supportedLanguages?.[values.gpu]?.includes(values.language)) {
      throw new Error('This GPU/language combination is not listed as supported. Check gpus.');
    }
    const requestId = randomUUID();
    const directory = join(dataDirectory, 'receipts');
    await mkdir(directory, { recursive: true, mode: 0o700 });
    const receiptPath = join(directory, `${requestId}.json`);
    const receipt = {
      requestId, challengeId: payload.challengeId, language: payload.language,
      accelerator: payload.accelerator, public: false, sourcePath,
      sourceSha256: createHash('sha256').update(source).digest('hex'),
      startedAt: new Date().toISOString(), status: 'prepared',
    };
    const handle = await open(receiptPath, 'wx', 0o600);
    const controller = new AbortController();
    const interrupt = () => controller.abort();
    const signals = ['SIGINT', 'SIGTERM', 'SIGHUP'];
    for (const signal of signals) process.on(signal, interrupt);
    try {
      await handle.writeFile(JSON.stringify(receipt, null, 2) + '\n');
      console.error(`Submitting challenge ${payload.challengeId} / ${payload.language} / ${payload.accelerator}. Receipt: ${receiptPath}`);
      let result;
      try {
        result = await submitSolution({ token: session.access_token, payload, submissionId: requestId, signal: controller.signal,
          onEvent: event => {
            if (typeof event.status === 'string') console.error(`[${event.status.replace(/[\x00-\x1f\x7f-\x9f]/g, '').slice(0, 80)}]`);
          },
        });
      } catch (error) {
        Object.assign(receipt, { status: 'unconfirmed', submissionId: error.submissionId ?? requestId,
          sent: error.sent ?? null, error: error.message, finishedAt: new Date().toISOString() });
        await handle.truncate(0);
        await handle.write(JSON.stringify(receipt, null, 2) + '\n', 0, 'utf8');
        throw new Error(`${error.message} Check history before retrying. Receipt: ${receiptPath}`);
      }
      Object.assign(receipt, result, { finishedAt: new Date().toISOString() });
      await handle.truncate(0);
      await handle.write(JSON.stringify(receipt, null, 2) + '\n', 0, 'utf8');
      if (values.json) printJson({ ...result, receiptPath });
      else console.log(`Submission ${result.submissionId}: ${result.status}\nReceipt: ${receiptPath}`);
      if (result.status !== 'success') process.exitCode = 2;
    } finally {
      try { await handle.close(); } finally {
        for (const signal of signals) process.removeListener(signal, interrupt);
      }
    }
    return;
  }
  if (command === 'challenges') {
    const payload = await api.challenges();
    if (values.json) { printJson(payload); return; }
    if (!Array.isArray(payload?.challenges)) throw new Error('Unexpected challenges response.');
    console.log(table(['ID', 'TITLE', 'DIFFICULTY'], payload.challenges.map(row => [row.id, row.title, row.difficultyLevel ?? row.difficulty])));
    return;
  }
  if (command === 'gpus') {
    const payload = await api.accelerators();
    if (values.json) { printJson(payload); return; }
    if (!Array.isArray(payload?.accelerators)) throw new Error('Unexpected accelerators response.');
    console.log(table(['GPU', 'LANGUAGES'], payload.accelerators.map(gpu => [gpu, payload.supportedLanguages?.[gpu]?.join(', ')])));
    return;
  }
  if (subcommand === 'latest') {
    const payload = await api.latest();
    if (values.json) { printJson(payload); return; }
    const row = latestSubmission(payload);
    console.log(row ? formatSubmissions([row]) : 'No latest submission returned.');
    return;
  }
  if (subcommand === 'list') {
    if (!['cuda', 'cute', 'triton', 'pytorch', 'jax', 'mojo'].includes(values.language)) {
      throw new Error('--language must be cuda, cute, triton, pytorch, jax, or mojo.');
    }
    const payload = await api.list(values);
    if (values.json) { printJson(payload); return; }
    const rows = submissionRows(payload);
    console.log(rows.length ? formatSubmissions(rows, values) : 'No submissions returned for these filters.');
    console.error('Showing the server-returned records; full-history coverage is not verified.');
    return;
  }
  if (values.json && values.file) throw new Error('Use either --json or --file.');
  const payload = await api.code(id);
  let result;
  if (values.json) result = JSON.stringify(payload, null, 2) + '\n';
  else {
    if (!Array.isArray(payload?.files) || !payload.files.length) throw new Error('No code files returned.');
    const files = payload.files;
    const matches = values.file ? files.filter(file => file.name === values.file) : files;
    if (matches.length !== 1) throw new Error('Choose one exact --file NAME, or use --json to retrieve all files.');
    if (typeof matches[0].content !== 'string') throw new Error('Unexpected code-file response.');
    result = matches[0].content;
  }
  if (values.output) {
    await writeFile(values.output, result, { flag: 'wx', mode: 0o600 });
    console.error('Submission code saved to the requested new file.');
  } else process.stdout.write(result);
}

function printJson(payload) { console.log(JSON.stringify(payload, null, 2)); }

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch(error => {
    console.error(`Error: ${error.message}`);
    process.exitCode = 1;
  });
}
