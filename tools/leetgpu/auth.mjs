import { spawn } from 'node:child_process';
import { constants } from 'node:fs';
import { access, chmod, lstat, mkdir, mkdtemp, open, readdir, readFile, rename, rm } from 'node:fs/promises';
import { homedir, platform, tmpdir } from 'node:os';
import { delimiter, join } from 'node:path';
import { randomUUID } from 'node:crypto';
import { configDirectory } from './paths.mjs';
import { openSecretStore, validSecret } from './secrets.mjs';

const origin = 'https://leetgpu.com';
export const sessionPath = join(configDirectory, 'session.json');

/** Normalize only the access token and expiration; never retain refresh tokens. */
export function parseSession(value, nowSeconds = Date.now() / 1000) {
  if (typeof value === 'string') {
    try { value = JSON.parse(value); } catch { return null; }
  }
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  if (value.currentSession) value = value.currentSession;
  else if (value.session) value = value.session;
  const token = value?.access_token;
  if (typeof token !== 'string' || !token || token.length > 65536 || /[\s\x00-\x1f\x7f]/u.test(token)) return null;
  let expiry = value.expires_at;
  if (!Number.isFinite(expiry)) {
    try { expiry = JSON.parse(Buffer.from(token.split('.')[1], 'base64url').toString('utf8')).exp; } catch { return null; }
  }
  if (!Number.isFinite(expiry) || expiry <= nowSeconds) return null;
  return { access_token: token, expires_at: Math.floor(expiry) };
}

/**
 * The refresh token of a captured or pasted session, or null. It is kept apart
 * from parseSession so that only an explicit persistent sign-in ever sees it.
 */
export function refreshTokenOf(value) {
  if (typeof value === 'string') {
    try { value = JSON.parse(value); } catch { return null; }
  }
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  const token = (value.currentSession || value.session || value).refresh_token;
  return validSecret(token) ? token : null;
}

async function checkDirectory() {
  const info = await lstat(configDirectory);
  if (!info.isDirectory() || info.isSymbolicLink()) throw new Error('The LeetGPU session directory must be a real directory.');
  await chmod(configDirectory, 0o700);
}

export async function saveSession(session) {
  const normalized = parseSession(session);
  if (!normalized) throw new Error('The LeetGPU session is invalid or expired; run login again.');
  await mkdir(configDirectory, { recursive: true, mode: 0o700 });
  await checkDirectory();
  const temporary = join(configDirectory, `.session-${randomUUID()}.tmp`);
  let handle;
  try {
    handle = await open(temporary, constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL, 0o600);
    await handle.writeFile(`${JSON.stringify(normalized)}\n`, 'utf8');
    await handle.close();
    handle = undefined;
    await rename(temporary, sessionPath);
  } finally {
    await handle?.close();
    await rm(temporary, { force: true });
  }
}

export async function loadSession() {
  let handle;
  try {
    await checkDirectory();
    handle = await open(sessionPath, constants.O_RDONLY | constants.O_NOFOLLOW);
    const info = await handle.stat();
    if (!info.isFile() || info.size > 100000) throw Object.assign(new Error('The LeetGPU session file is invalid; run login again.'), { code: 'SESSION_INVALID' });
    await handle.chmod(0o600);
    const session = parseSession(await handle.readFile('utf8'));
    if (!session) throw Object.assign(new Error('The LeetGPU session is invalid or expired; run login again.'), { code: 'SESSION_INVALID' });
    return session;
  } catch (error) {
    if (error.code === 'ENOENT') return null;
    if (error.code === 'ELOOP') throw new Error('The LeetGPU session file must not be a symbolic link.');
    throw error;
  } finally {
    await handle?.close();
  }
}

export async function logout() {
  // Remove the long-lived secret first: without it the session cannot be renewed.
  await openSecretStore().delete();
  try { await checkDirectory(); } catch (error) {
    if (error.code === 'ENOENT') return;
    throw error;
  }
  await rm(sessionPath, { force: true });
}

function pause(milliseconds, signal) {
  signal?.throwIfAborted();
  return new Promise((resolvePromise, reject) => {
    const timer = setTimeout(() => finish(), milliseconds);
    const abort = () => finish(signal.reason);
    function finish(error) {
      clearTimeout(timer);
      signal?.removeEventListener('abort', abort);
      if (error) reject(error); else resolvePromise();
    }
    signal?.addEventListener('abort', abort, { once: true });
  });
}

async function chromeExecutable() {
  const candidates = process.env.LEETGPU_HISTORY_CHROME ? [process.env.LEETGPU_HISTORY_CHROME] : platform() === 'darwin' ? [
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    join(homedir(), 'Applications/Google Chrome.app/Contents/MacOS/Google Chrome'),
    '/Applications/Chromium.app/Contents/MacOS/Chromium',
  ] : (process.env.PATH || '').split(delimiter).flatMap(directory => ['google-chrome', 'chromium', 'chromium-browser'].map(name => join(directory, name)));
  for (const candidate of candidates) {
    try { await access(candidate, constants.X_OK); return candidate; } catch { /* Try the next installed browser. */ }
  }
  throw new Error('Google Chrome or Chromium is required. Set LEETGPU_HISTORY_CHROME to its executable path.');
}

function parseDevToolsPort(value) {
  const [portText, path] = value.trim().split(/\r?\n/u);
  const port = Number(portText);
  if (!Number.isInteger(port) || port < 1 || port > 65535 || !/^\/devtools\/browser\/[a-zA-Z0-9-]+$/u.test(path || '')) return null;
  return { port, browserUrl: `ws://127.0.0.1:${port}${path}` };
}

async function waitForEndpoint(profile, signal) {
  const deadline = Date.now() + 20000;
  while (Date.now() < deadline) {
    signal.throwIfAborted();
    try {
      const endpoint = parseDevToolsPort(await readFile(join(profile, 'DevToolsActivePort'), 'utf8'));
      if (endpoint) return endpoint;
    } catch (error) { if (error.code !== 'ENOENT') throw error; }
    await pause(100, signal);
  }
  throw new Error('The dedicated Chrome window did not start in time.');
}

function callCdp(url, method, params = {}, { signal, WebSocketClass = WebSocket, timeoutMs = 3000 } = {}) {
  signal?.throwIfAborted();
  return new Promise((resolvePromise, reject) => {
    const socket = new WebSocketClass(url);
    let finished = false;
    const timer = setTimeout(() => finish(new Error('The dedicated browser connection timed out.')), timeoutMs);
    const abort = () => finish(signal.reason);
    const finish = (error, value) => {
      if (finished) return;
      finished = true;
      clearTimeout(timer);
      signal?.removeEventListener('abort', abort);
      try { socket.close(); } catch { /* A connection can already be closed. */ }
      if (error) reject(error); else resolvePromise(value);
    };
    signal?.addEventListener('abort', abort, { once: true });
    socket.addEventListener('open', () => {
      if (finished) return;
      socket.send(JSON.stringify({ id: 1, method, params }));
    });
    socket.addEventListener('message', event => {
      try {
        const message = JSON.parse(event.data);
        if (message.id !== 1) return;
        if (message.error) finish(new Error('The dedicated browser rejected the session request.'));
        else finish(undefined, message.result);
      } catch { finish(new Error('The dedicated browser returned an invalid response.')); }
    });
    socket.addEventListener('error', () => finish(new Error('Could not connect to the dedicated browser.')));
    socket.addEventListener('close', () => finish(new Error('The dedicated browser connection closed.')));
  });
}

// This function is evaluated in a page of the owned temporary profile. The origin
// is checked again here because a page may navigate after /json/list was fetched.
// The refresh token leaves the page only for an explicit persistent sign-in.
function captureSessionInPage(keepRefreshToken) {
  if (location.origin !== 'https://leetgpu.com') return null;
  // This is LeetGPU's public Supabase project ID, verified in the website.
  for (const key of ['sb-yhdtysacdkqoquvkdwdd-auth-token']) {
    try {
      const raw = JSON.parse(localStorage.getItem(key));
      const session = raw?.currentSession || raw?.session || raw;
      if (typeof session?.access_token !== 'string') continue;
      let expires_at = session.expires_at;
      if (typeof expires_at !== 'number') {
        const encoded = session.access_token.split('.')[1].replace(/-/gu, '+').replace(/_/gu, '/');
        expires_at = JSON.parse(atob(encoded)).exp;
      }
      if (Number.isFinite(expires_at) && expires_at > Date.now() / 1000) {
        const captured = { access_token: session.access_token, expires_at };
        if (keepRefreshToken === true && typeof session.refresh_token === 'string') captured.refresh_token = session.refresh_token;
        return captured;
      }
    } catch { /* Ignore incomplete or expired auth entries. */ }
  }
  return null;
}

function pageSocket(target, port) {
  try {
    if (target.type !== 'page' || new URL(target.url).origin !== origin) return null;
    const socket = new URL(target.webSocketDebuggerUrl);
    if (socket.protocol !== 'ws:' || !['127.0.0.1', 'localhost'].includes(socket.hostname) || Number(socket.port) !== port || !/^\/devtools\/page\/[a-zA-Z0-9-]+$/u.test(socket.pathname)) return null;
    socket.hostname = '127.0.0.1';
    return socket.href;
  } catch { return null; }
}

// What a signed-in page yields. A persistent sign-in that finds the page signed
// in but without a usable refresh token fails at once: waiting cannot fix it,
// and a silent wait would look the same as nobody signing in.
function pickSession(value, persist) {
  const session = parseSession(value);
  if (!session || !persist) return session;
  const refreshToken = refreshTokenOf(value);
  if (!refreshToken) {
    throw Object.assign(new Error('Signed in, but the page session has no usable refresh token, so it cannot be kept. Run login --browser without --persist for a one-hour session.'), { code: 'NO_REFRESH_TOKEN' });
  }
  return { ...session, refresh_token: refreshToken };
}

async function waitForSession(endpoint, signal, persist = false) {
  while (true) {
    signal.throwIfAborted();
    try {
      const response = await fetch(`http://127.0.0.1:${endpoint.port}/json/list`, {
        signal: AbortSignal.any([signal, AbortSignal.timeout(3000)]), redirect: 'error',
      });
      if (!response.ok) throw new Error('Browser targets unavailable.');
      const targets = await response.json();
      if (!Array.isArray(targets)) throw new Error('Invalid browser targets.');
      for (const target of targets) {
        const socket = pageSocket(target, endpoint.port);
        if (!socket) continue;
        const response = await callCdp(socket, 'Runtime.evaluate', {
          expression: `(${captureSessionInPage.toString()})(${persist})`, returnByValue: true,
        }, { signal });
        const session = pickSession(response?.result?.value, persist);
        if (session) return session;
      }
    } catch (error) {
      if (error.code === 'NO_REFRESH_TOKEN') throw error;
      signal.throwIfAborted();
      // OAuth navigation may briefly replace the owned page target.
    }
    await pause(400, signal);
  }
}

/**
 * A proxy for the dedicated sign-in window, in the form Chrome's --proxy-server
 * takes. Credentials are refused: Chrome ignores them there, and they would be
 * visible in the process list.
 */
export function normalizeProxy(value) {
  let url = null;
  if (typeof value === 'string' && value.length <= 2048) {
    try { url = new URL(value); } catch { /* Reported below without echoing the value. */ }
  }
  if (!url || !['http:', 'https:', 'socks4:', 'socks5:'].includes(url.protocol) || !url.hostname
    || url.username || url.password || !['', '/'].includes(url.pathname) || url.search || url.hash) {
    throw new Error('--proxy must look like http://host:port (http, https, socks4 or socks5, without credentials or a path).');
  }
  return `${url.protocol}//${url.host}`;
}

// LeetGPU's public Supabase project ID names the local-storage key of its session.
const SESSION_KEY = 'sb-yhdtysacdkqoquvkdwdd-auth-token';
// A static page of the same origin: its local storage is the site's, and none of the site's scripts run.
const CAPTURE_URL = `${origin}/robots.txt`;

// Step 1, the sign-in window: a plain Chrome with no debugging or automation
// switches, because Google refuses to sign in to a browser that has them.
// The proxy applies to these windows only; the CLI's own requests stay direct.
function signInArguments(profile, proxy) {
  return [
    `--user-data-dir=${profile}`, '--no-first-run', '--no-default-browser-check', '--disable-sync',
    ...(proxy ? [`--proxy-server=${proxy}`] : []),
    '--new-window', origin,
  ];
}

// Step 2, after sign-in: the same profile reopened without a window, now with
// debugging on loopback, only to read the session it holds.
function captureArguments(profile, proxy) {
  return [
    `--user-data-dir=${profile}`, '--headless=new', '--remote-debugging-address=127.0.0.1', '--remote-debugging-port=0',
    '--no-first-run', '--no-default-browser-check', '--disable-sync',
    ...(proxy ? [`--proxy-server=${proxy}`] : []),
    CAPTURE_URL,
  ];
}

/**
 * Whether Chrome has written the LeetGPU session into the profile's local
 * storage. This looks for the key's bytes in Chrome's storage files; it only
 * signals "signed in", and step 2 does the actual, exact read.
 */
async function sessionWritten(profile) {
  const directory = join(profile, 'Default', 'Local Storage', 'leveldb');
  let names;
  try { names = await readdir(directory); } catch { return false; }
  const key = Buffer.from(SESSION_KEY, 'latin1');
  for (const name of names) {
    if (!/\.(?:log|ldb)$/u.test(name)) continue;
    let data;
    try { data = await readFile(join(directory, name)); } catch { continue; }
    for (let at = data.indexOf(key); at !== -1; at = data.indexOf(key, at + 1)) {
      // `<key>-code-verifier` is written when sign-in starts; only the bare key means it finished.
      if (data[at + key.length] !== 0x2d) return true;
    }
  }
  return false;
}

function isRunning(browser) {
  return browser && browser.pid && browser.exitCode === null && browser.signalCode === null;
}

async function stopOwnedBrowser(browser, endpoint) {
  if (!isRunning(browser)) return;
  if (endpoint) {
    try { await callCdp(endpoint.browserUrl, 'Browser.close', {}, { timeoutMs: 1000 }); } catch { /* Chrome may close before acknowledging. */ }
  }
  const waitUntil = async timeoutMs => {
    const deadline = Date.now() + timeoutMs;
    while (isRunning(browser) && Date.now() < deadline) await pause(50);
  };
  await waitUntil(500);
  if (isRunning(browser)) browser.kill('SIGTERM');
  await waitUntil(1500);
  if (isRunning(browser)) browser.kill('SIGKILL');
  await waitUntil(1000);
  if (isRunning(browser)) throw new Error('The dedicated Chrome process could not be stopped.');
}

/**
 * Wait for step 1 to end. Returns 'signed-in' once the profile holds a session,
 * 'confirmed' when the user presses Enter on an interactive terminal, or
 * 'closed' when the user quits that Chrome.
 */
async function waitForSignIn(browser, profile, signal, input) {
  let confirmed = false;
  const onData = () => { confirmed = true; };
  const interactive = Boolean(input?.isTTY);
  if (interactive) { input.on('data', onData); input.resume(); }
  try {
    while (isRunning(browser) && !confirmed) {
      signal.throwIfAborted();
      if (await sessionWritten(profile)) {
        // Let Chrome finish writing before it is asked to quit.
        await pause(1500, signal);
        return 'signed-in';
      }
      await pause(500, signal);
    }
    return confirmed ? 'confirmed' : 'closed';
  } finally {
    if (interactive) { input.removeListener('data', onData); input.pause(); }
  }
}

// Step 2: read the session out of a profile whose Chrome has exited.
async function readProfileSession({ executable, profile, proxyServer, persist, signal, timeoutMs }) {
  signal.throwIfAborted();
  const controller = new AbortController();
  const forward = () => controller.abort(signal.reason);
  signal.addEventListener('abort', forward, { once: true });
  const timer = setTimeout(() => controller.abort(new Error('No LeetGPU session was found in the sign-in window. Finish signing in before it closes, then run login again.')), timeoutMs);
  let reader;
  let endpoint;
  let stopping = false;
  try {
    await rm(join(profile, 'DevToolsActivePort'), { force: true });
    reader = spawn(executable, captureArguments(profile, proxyServer), { stdio: 'ignore' });
    reader.once('error', () => controller.abort(new Error('Could not reopen the sign-in profile to read the session.')));
    reader.once('exit', () => {
      if (!stopping) controller.abort(new Error('Chrome exited before the session could be read. Run login again.'));
    });
    endpoint = await waitForEndpoint(profile, controller.signal);
    return await waitForSession(endpoint, controller.signal, persist);
  } finally {
    stopping = true;
    clearTimeout(timer);
    signal.removeEventListener('abort', forward);
    await stopOwnedBrowser(reader, endpoint);
  }
}

export async function browserLogin({
  timeoutMs = 600000, onStatus = () => {}, persist = false, proxy, input = process.stdin,
} = {}) {
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) throw new Error('Login timeout must be a positive number.');
  const proxyServer = proxy === undefined ? null : normalizeProxy(proxy);
  const executable = await chromeExecutable();
  const profile = await mkdtemp(join(tmpdir(), 'leetgpu-history-login-'));
  await chmod(profile, 0o700);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(new Error('LeetGPU login timed out. Run login again to retry.')), timeoutMs);
  const interrupt = () => controller.abort(new Error('LeetGPU login canceled.'));
  process.once('SIGINT', interrupt);
  process.once('SIGTERM', interrupt);
  let browser;
  try {
    const fallback = input?.isTTY ? 'quit that Chrome or press Enter here' : 'quit that Chrome';
    onStatus(`Opening a separate Chrome window${proxyServer ? ` through the proxy ${proxyServer}` : ''}. Sign in to LeetGPU there. `
      + `It closes by itself once you are signed in; if it stays open afterwards, ${fallback}.`);
    browser = spawn(executable, signInArguments(profile, proxyServer), { stdio: 'ignore' });
    browser.once('error', () => controller.abort(new Error('Could not start the Chrome sign-in window.')));
    const outcome = await waitForSignIn(browser, profile, controller.signal, input);
    // Chrome must exit before its profile can be reopened.
    await stopOwnedBrowser(browser);
    onStatus('Reading the LeetGPU session from the sign-in window...');
    // The profile is deleted below, so for a persistent sign-in this CLI becomes
    // the only holder of the session and nothing else can spend its refresh token.
    return await readProfileSession({
      executable, profile, proxyServer, persist: persist === true, signal: controller.signal,
      timeoutMs: outcome === 'signed-in' ? 30000 : 15000,
    });
  } finally {
    clearTimeout(timer);
    process.removeListener('SIGINT', interrupt);
    process.removeListener('SIGTERM', interrupt);
    try {
      await stopOwnedBrowser(browser);
    } finally {
      await rm(profile, { recursive: true, force: true, maxRetries: 3, retryDelay: 100 });
    }
  }
}

export const __test = {
  captureSessionInPage, pageSocket, parseDevToolsPort, callCdp, stopOwnedBrowser, pickSession,
  signInArguments, captureArguments, sessionWritten, waitForSignIn, readProfileSession, SESSION_KEY,
};
