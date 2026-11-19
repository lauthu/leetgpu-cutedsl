import { spawn } from 'node:child_process';
import { createHash, randomUUID } from 'node:crypto';
import { constants } from 'node:fs';
import { mkdir, open, rename, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { configDirectory } from './paths.mjs';

// One Keychain item per config directory, so separate LEETGPU_CONFIG_DIR setups never share a sign-in.
const SERVICE = 'leetgpu-history';
const ACCOUNT = `refresh-token-${createHash('sha256').update(configDirectory).digest('hex').slice(0, 16)}`;
const LABEL = 'leetgpu-cli-sign-in';
const NOT_FOUND = 44;
const filePath = join(configDirectory, 'refresh-token');

/** A refresh token is opaque; this only rejects values that could not be one (or could break a command line). */
export const validSecret = value => typeof value === 'string' && /^[A-Za-z0-9._~+/=-]{8,1024}$/u.test(value);

/** `keychain` on macOS, otherwise `none`; LEETGPU_SECRET_STORE can select `file` or `none` explicitly. */
export function secretStoreKind(env = process.env, platformName = process.platform) {
  const requested = env.LEETGPU_SECRET_STORE;
  if (!requested) return platformName === 'darwin' ? 'keychain' : 'none';
  if (!['keychain', 'file', 'none'].includes(requested)) throw new Error('LEETGPU_SECRET_STORE must be keychain, file, or none.');
  if (requested === 'keychain' && platformName !== 'darwin') throw new Error('The keychain secret store requires macOS.');
  return requested;
}

// Runs /usr/bin/security. The secret only ever travels through stdin or stdout, never argv.
function runSecurity(args, input = '') {
  return new Promise((resolve, reject) => {
    const child = spawn('/usr/bin/security', args, { stdio: ['pipe', 'pipe', 'ignore'] });
    let stdout = '';
    const timer = setTimeout(() => { child.kill(); reject(new Error('timeout')); }, 15000);
    child.stdout.setEncoding('utf8');
    child.stdout.on('data', chunk => { stdout += chunk; });
    child.once('error', error => { clearTimeout(timer); reject(error); });
    child.once('close', code => { clearTimeout(timer); resolve({ code, stdout }); });
    child.stdin.on('error', () => {});
    child.stdin.end(input);
  });
}

function keychainStore(run) {
  const failed = () => new Error('Could not use the macOS Keychain. Unlock the login keychain and retry.');
  const call = async (args, input) => {
    try { return await run(args, input); } catch { throw failed(); }
  };
  const get = async () => {
    const { code, stdout } = await call(['find-generic-password', '-s', SERVICE, '-a', ACCOUNT, '-w']);
    if (code === NOT_FOUND) return null;
    const value = stdout.replace(/\r?\n$/u, '');
    if (code !== 0 || !validSecret(value)) throw failed();
    return value;
  };
  return {
    kind: 'keychain', get,
    async set(value) {
      if (!validSecret(value)) throw new Error('Refusing to store an invalid refresh token.');
      // `security -i` reads the command from stdin; -U replaces an existing item in place.
      await call(['-i'], `add-generic-password -U -s ${SERVICE} -a ${ACCOUNT} -l ${LABEL} -w ${value}\n`);
      // Interactive mode does not report a failed command through its exit code, so read it back.
      if (await get() !== value) throw failed();
    },
    async delete() {
      const { code } = await call(['delete-generic-password', '-s', SERVICE, '-a', ACCOUNT]);
      if (code !== 0 && code !== NOT_FOUND) throw failed();
    },
  };
}

// Plain text in a 0600 file: for tests, and for machines without a keychain when the owner opts in.
const fileStore = {
  kind: 'file',
  async get() {
    let handle;
    try {
      handle = await open(filePath, constants.O_RDONLY | constants.O_NOFOLLOW);
      const info = await handle.stat();
      if (!info.isFile() || info.size > 2048) return null;
      await handle.chmod(0o600);
      const value = (await handle.readFile('utf8')).trim();
      return validSecret(value) ? value : null;
    } catch (error) {
      if (error.code === 'ENOENT') return null;
      if (error.code === 'ELOOP') throw new Error('The saved refresh token must not be a symbolic link.');
      throw error;
    } finally { await handle?.close(); }
  },
  async set(value) {
    if (!validSecret(value)) throw new Error('Refusing to store an invalid refresh token.');
    await mkdir(configDirectory, { recursive: true, mode: 0o700 });
    const temporary = join(configDirectory, `.refresh-${randomUUID()}.tmp`);
    let handle;
    try {
      handle = await open(temporary, constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL, 0o600);
      await handle.writeFile(`${value}\n`, 'utf8');
      await handle.close();
      handle = undefined;
      await rename(temporary, filePath);
    } finally {
      await handle?.close();
      await rm(temporary, { force: true });
    }
  },
  async delete() { await rm(filePath, { force: true }); },
};

const noStore = {
  kind: 'none',
  async get() { return null; },
  async set() { throw new Error('Staying signed in needs a secret store: the macOS Keychain, or LEETGPU_SECRET_STORE=file.'); },
  async delete() {},
};

/** Where the refresh token of a persistent sign-in lives. */
export function openSecretStore({ env = process.env, platformName = process.platform, run = runSecurity } = {}) {
  const kind = secretStoreKind(env, platformName);
  return kind === 'keychain' ? keychainStore(run) : kind === 'file' ? fileStore : noStore;
}

export const __test = { SERVICE, ACCOUNT, filePath };
