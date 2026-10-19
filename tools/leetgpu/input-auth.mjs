import { parsePastedSession } from './manual-auth.mjs';

const maxInputBytes = 100000;

/** Read a session from a pipe/file without echoing it or retaining refresh tokens. */
export async function readSessionInput(stream = process.stdin) {
  if (stream?.isTTY) {
    throw new Error('Session input must come from a pipe or file, not an interactive terminal. Use login for interactive authentication.');
  }

  const chunks = [];
  let bytes = 0;
  let oversized = false;
  try {
    for await (const chunk of stream) {
      if (typeof chunk !== 'string' && !(chunk instanceof Uint8Array)) throw new Error('Invalid stream data.');
      const buffer = Buffer.from(chunk);
      bytes += buffer.byteLength;
      if (bytes > maxInputBytes) {
        oversized = true;
        break;
      }
      chunks.push(buffer);
    }
  } catch {
    chunks.length = 0;
    throw new Error('Could not read the LeetGPU session from standard input.');
  }
  if (oversized) {
    chunks.length = 0;
    throw new Error(`LeetGPU session input exceeds ${maxInputBytes} bytes.`);
  }
  const input = Buffer.concat(chunks).toString('utf8');
  chunks.length = 0;
  const session = parsePastedSession(input);
  if (!session) throw new Error('The LeetGPU session input is invalid or expired. Sign in again and retry.');
  return session;
}
