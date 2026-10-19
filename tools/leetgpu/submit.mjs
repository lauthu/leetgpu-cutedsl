import { randomUUID } from 'node:crypto';
import { challengeId } from './api.mjs';

const WS_URL = 'wss://api.leetgpu.com/api/v1/ws/submit';
const LANGUAGES = new Set(['cuda', 'triton', 'pytorch', 'cute', 'mojo', 'jax']);
const FINAL_STATUSES = new Set([
  'success', 'test-case-failed', 'compilation-failed', 'timeout', 'out-of-memory',
  'interrupted', 'output-exceeded', 'tampering-detected', 'error',
]);
const validId = value => typeof value === 'string' && /^[A-Za-z0-9_-]{1,128}$/u.test(value);

function tokenSubject(token) {
  try {
    if (typeof token !== 'string' || token.length > 65536 || !/^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$/u.test(token)) throw new Error();
    const subject = JSON.parse(Buffer.from(token.split('.')[1], 'base64url').toString('utf8')).sub;
    if (!validId(subject)) throw new Error();
    return subject;
  } catch {
    throw new Error('The LeetGPU session has no valid user ID; run login again.');
  }
}

export function buildSubmission({ token, challenge, language, gpu, files } = {}) {
  const challengeNumber = Number(challengeId(challenge));
  if (!LANGUAGES.has(language)) throw new Error('Unsupported submission language.');
  if (typeof gpu !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9 -]{0,63}$/u.test(gpu) || gpu.trim() !== gpu) {
    throw new Error('Invalid submission GPU.');
  }
  if (!Array.isArray(files) || !files.length || files.some(file => (
    !file || typeof file !== 'object' || typeof file.name !== 'string'
    || !/^[A-Za-z0-9_][A-Za-z0-9_.-]{0,254}$/u.test(file.name)
    || typeof file.content !== 'string'
  )) || new Set(files.map(file => file.name)).size !== files.length) {
    throw new Error('Submission files must have unique, safe file names and text content.');
  }
  return {
    files: files.map(({ name, content }) => ({ name, content })),
    language, accelerator: gpu, mode: 'accelerated', challengeId: challengeNumber,
    userId: tokenSubject(token), public: false,
  };
}

function sanitize(value, token) {
  if (typeof value === 'string') return value.replaceAll(token, '[REDACTED]');
  if (Array.isArray(value)) return value.map(item => sanitize(item, token));
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value)
      .filter(([key]) => !['token', 'accesstoken', 'refreshtoken'].includes(key.toLowerCase().replaceAll('_', '').replaceAll('-', '')))
      .map(([key, item]) => [key.replaceAll(token, '[REDACTED]'), sanitize(item, token)]));
  }
  return value;
}

/** One attempt only: an interrupted connection may already have created a submission. */
export async function submitSolution({
  token, payload, submissionId = randomUUID(), onEvent = () => {},
  WebSocketClass = globalThis.WebSocket, timeoutMs = 180000, signal,
} = {}) {
  if (!validId(submissionId)) throw new Error('Invalid submission ID.');
  const submission = buildSubmission({ token, challenge: payload?.challengeId, language: payload?.language, gpu: payload?.accelerator, files: payload?.files });
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs <= 0) throw new Error('Submission timeout must be a positive integer.');
  if (typeof onEvent !== 'function') throw new Error('Submission event handler must be a function.');
  let currentId = submissionId;
  let sent = false;
  const failure = message => Object.assign(new Error(message), { submissionId: currentId, sent });
  if (signal?.aborted) throw failure('The submission was canceled before connecting.');
  if (typeof WebSocketClass !== 'function') throw failure('This Node.js runtime does not support WebSocket; use Node.js 22 or newer.');

  return new Promise((resolve, reject) => {
    let socket;
    let settled = false;
    let timer;
    const events = [];
    const listeners = [];
    const abort = () => finish(failure('The submission was interrupted. Check its history before submitting again.'));
    function finish(error, result) {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      signal?.removeEventListener('abort', abort);
      for (const [name, callback] of listeners) socket?.removeEventListener(name, callback);
      // Closing does not cancel the server job. Preserve its ID for a history lookup.
      try { socket?.close(1000); } catch { /* Do not expose transport error details. */ }
      if (error) reject(error); else resolve(result);
    }
    function listen(name, callback) {
      listeners.push([name, callback]);
      socket.addEventListener(name, callback);
    }
    try {
      socket = new WebSocketClass(WS_URL);
      signal?.addEventListener('abort', abort, { once: true });
      if (signal?.aborted) { abort(); return; }
      listen('open', () => {
        if (settled || sent) return;
        try {
          const message = JSON.stringify({ token, submissionId, action: 'submit', submission, idempotencyKey: submissionId });
          // Mark attempted sends conservatively: never automatically repeat after an error.
          sent = true;
          socket.send(message);
        } catch {
          finish(failure('Could not send the submission. Check its history before submitting again.'));
        }
      });
      listen('message', event => {
        if (settled) return;
        try {
          const received = sanitize(JSON.parse(event.data), token);
          if (!received || typeof received !== 'object' || Array.isArray(received)) throw new Error();
          if (validId(received.submissionId)) currentId = received.submissionId;
          events.push(received);
          onEvent(received);
          if (FINAL_STATUSES.has(received.status)) {
            finish(null, { submissionId: currentId, status: received.status, events });
          }
        } catch {
          finish(failure('Could not process a submission event. Check its history before submitting again.'));
        }
      });
      listen('error', () => finish(failure('The submission connection failed. Check its history before submitting again.')));
      listen('close', () => finish(failure('The submission connection closed before a final result. Check its history before submitting again.')));
      timer = setTimeout(() => finish(failure('Timed out waiting for the submission result. Check its history before submitting again.')), timeoutMs);
    } catch {
      finish(failure('Could not open the LeetGPU submission connection.'));
    }
  });
}
