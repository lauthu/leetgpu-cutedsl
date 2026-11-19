export const API_ORIGIN = 'https://api.leetgpu.com';

export class ApiError extends Error {
  constructor(message, status) { super(message); this.name = 'ApiError'; this.status = status; }
}

export class LeetGpuApi {
  constructor(token, fetchImpl = globalThis.fetch) {
    this.token = token;
    this.fetchImpl = fetchImpl;
  }

  async get(path, authenticated = true) {
    const url = new URL(path, API_ORIGIN);
    if (url.origin !== API_ORIGIN || !url.pathname.startsWith('/api/v1/')) {
      throw new ApiError('Invalid LeetGPU API path.');
    }
    if (authenticated && !this.token) throw new ApiError('Please run login first.', 401);
    const headers = { Accept: 'application/json' };
    if (authenticated) headers.Authorization = `Bearer ${this.token}`;
    let response;
    try {
      response = await this.fetchImpl(url, {
        method: 'GET', headers, redirect: 'error', signal: AbortSignal.timeout(20_000),
      });
    } catch {
      throw new ApiError('LeetGPU request failed or timed out. Check your connection and retry.');
    }
    if (response.status === 401) throw new ApiError('Session expired or invalid. Run login again.', 401);
    if (response.status === 403) throw new ApiError('This account cannot access the requested submission.', 403);
    if (response.status === 404) throw new ApiError('Submission not found or not accessible.', 404);
    if (response.status === 429) throw new ApiError('LeetGPU rate limit reached. Wait and retry.', 429);
    if (!response.ok) throw new ApiError(`LeetGPU returned HTTP ${response.status}.`, response.status);
    if (response.status === 204) return null;
    try { return await response.json(); }
    catch { throw new ApiError('LeetGPU returned invalid JSON.'); }
  }

  challenges() { return this.get('/api/v1/challenges', false); }
  accelerators() { return this.get('/api/v1/settings/accelerated/accelerators', false); }
  latest({ challenge, language } = {}) {
    const query = new URLSearchParams();
    if (challenge !== undefined) query.set('challengeId', challengeId(challenge));
    if (language) query.set('language', language);
    return this.get(`/api/v1/submissions/latest${query.size ? `?${query}` : ''}`);
  }
  list({ challenge, language, gpu }) {
    if (!language || !gpu) throw new ApiError('Both --language and --gpu are required.');
    const query = new URLSearchParams({ challengeId: challengeId(challenge), language, accelerator: gpu });
    return this.get(`/api/v1/submissions?${query}`);
  }
  code(id) {
    if (!/^[A-Za-z0-9_-]+$/.test(id ?? '')) throw new ApiError('Invalid submission ID.');
    return this.get(`/api/v1/submissions/${encodeURIComponent(id)}/code`);
  }
}

export function challengeId(value) {
  if (!/^[1-9]\d*$/.test(String(value)) || !Number.isSafeInteger(Number(value))) {
    throw new ApiError('--challenge must be a positive integer.');
  }
  return String(value);
}

/** The website's URL slug for a title, which is also this repository's directory name. */
export function slugify(value) {
  return String(value ?? '').normalize('NFKD').replace(/[̀-ͯ]/g, '').toLowerCase()
    .replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
}

/**
 * Find one challenge by title slug. `names` are tried in order, so pass the
 * nearest directory first; a name that matches nothing is skipped.
 */
export function findChallenge(payload, names) {
  const rows = payload?.challenges;
  if (!Array.isArray(rows)) throw new ApiError('Unexpected challenges response.');
  for (const name of names) {
    const slug = slugify(name);
    if (!slug) continue;
    const matches = rows.filter(row => row && typeof row.title === 'string' && slugify(row.title) === slug);
    if (matches.length > 1) throw new ApiError(`More than one challenge is named ${slug}; pass --challenge ID.`);
    if (matches.length === 1) return { id: challengeId(matches[0].id), title: matches[0].title, slug };
  }
  return null;
}

export function submissionRows(payload) {
  const rows = Array.isArray(payload) ? payload : payload?.submissions;
  if (!Array.isArray(rows) || rows.some(row => !row || typeof row !== 'object' || Array.isArray(row))) {
    throw new ApiError('Unexpected submissions response; use --json to inspect the server response.');
  }
  return rows;
}

export function latestSubmission(payload) {
  if (payload === null) return null;
  const row = Object.hasOwn(payload, 'submission') ? payload.submission : payload;
  if (row === null) return null;
  if (!row || typeof row !== 'object' || !row.id) throw new ApiError('Unexpected latest-submission response.');
  return row;
}

// Only remove terminal controls for table output; --json preserves server data.
export function table(headers, rows) {
  const clean = value => String(value ?? '').replace(/[\x00-\x1f\x7f-\x9f]/g, ' ');
  const matrix = [headers, ...rows].map(row => row.map(clean));
  const widths = headers.map((_, i) => Math.max(...matrix.map(row => row[i].length)));
  return matrix.map(row => row.map((cell, i) => cell.padEnd(widths[i])).join('  ').trimEnd()).join('\n');
}

export function formatSubmissions(rows, filters = {}) {
  // Confirmed against a live history response and the website's ms formatter.
  return table(['ID', 'CHALLENGE', 'STATUS', 'RUNTIME (ms)', 'PERCENTILE', 'LANGUAGE', 'GPU', 'CREATED'], rows.map(row => [
    row.id, row.challengeId ?? filters.challenge, row.status, row.metrics?.executionTime,
    row.metrics?.percentile, row.language ?? filters.language, row.accelerator ?? filters.gpu,
    row.createdAt ?? row.created_at,
  ]));
}
