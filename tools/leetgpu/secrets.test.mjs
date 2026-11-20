import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, rm, stat, symlink, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';

const directory = await mkdtemp(join(tmpdir(), 'leetgpu-secrets-test-'));
process.env.LEETGPU_CONFIG_DIR = join(directory, 'config');
const { openSecretStore, secretStoreKind, validSecret, __test } = await import('./secrets.mjs');
test.after(async () => { await rm(directory, { recursive: true, force: true }); });

const secret = 'refresh.Token_123-abc+/=';

test('the store defaults to the keychain on macOS only and rejects unknown kinds', () => {
  assert.equal(secretStoreKind({}, 'darwin'), 'keychain');
  assert.equal(secretStoreKind({}, 'linux'), 'none');
  assert.equal(secretStoreKind({ LEETGPU_SECRET_STORE: 'file' }, 'linux'), 'file');
  assert.equal(secretStoreKind({ LEETGPU_SECRET_STORE: 'none' }, 'darwin'), 'none');
  assert.throws(() => secretStoreKind({ LEETGPU_SECRET_STORE: 'keychain' }, 'linux'), /requires macOS/u);
  assert.throws(() => secretStoreKind({ LEETGPU_SECRET_STORE: 'plaintext' }, 'darwin'), /keychain, file, or none/u);
  for (const value of ['short', 'has space inside', 'quote"inside1', 'new\nline-inside', '', null, 'x'.repeat(1025)]) assert.equal(validSecret(value), false);
  assert.equal(validSecret(secret), true);
});

// Mimics /usr/bin/security closely enough to check what reaches argv and what reaches stdin.
function fakeSecurity({ ignoreAdd = false } = {}) {
  const calls = [];
  let stored = null;
  const run = async (args, input = '') => {
    calls.push({ args, input });
    if (args[0] === '-i') {
      if (!ignoreAdd) stored = /-w (\S+)\n$/u.exec(input)[1];
      return { code: 0, stdout: '' };
    }
    if (args[0] === 'find-generic-password') return stored ? { code: 0, stdout: `${stored}\n` } : { code: 44, stdout: '' };
    const had = stored !== null;
    stored = null;
    return { code: had ? 0 : 44, stdout: '' };
  };
  return { calls, run };
}

test('keychain secrets travel through stdin and stdout only, never through arguments', async () => {
  const fake = fakeSecurity();
  const store = openSecretStore({ env: {}, platformName: 'darwin', run: fake.run });
  assert.equal(store.kind, 'keychain');
  assert.equal(await store.get(), null);
  await store.set(secret);
  await store.set('rotated-Token_456');
  assert.equal(await store.get(), 'rotated-Token_456');
  await store.delete();
  await store.delete();
  assert.equal(await store.get(), null);
  assert.equal(fake.calls.some(call => call.args.some(arg => arg.includes(secret) || arg.includes('rotated-Token_456'))), false);
  const adds = fake.calls.filter(call => call.args[0] === '-i');
  assert.equal(adds.length, 2);
  assert.equal(adds[0].input, `add-generic-password -U -s ${__test.SERVICE} -a ${__test.ACCOUNT} -l leetgpu-cli-sign-in -w ${secret}\n`);
  assert.match(__test.ACCOUNT, /^refresh-token-[a-f0-9]{16}$/u);
});

test('keychain failures are generic and an unconfirmed write is an error', async () => {
  const silent = openSecretStore({ env: {}, platformName: 'darwin', run: fakeSecurity({ ignoreAdd: true }).run });
  await assert.rejects(silent.set(secret), error => /Keychain/u.test(error.message) && !error.message.includes(secret));
  const broken = openSecretStore({ env: {}, platformName: 'darwin', run: async () => { throw new Error(secret); } });
  for (const action of [() => broken.get(), () => broken.set(secret), () => broken.delete()]) {
    await assert.rejects(action(), error => /Keychain/u.test(error.message) && !error.message.includes(secret));
  }
  const odd = openSecretStore({ env: {}, platformName: 'darwin', run: async () => ({ code: 1, stdout: '' }) });
  await assert.rejects(odd.get(), /Keychain/u);
  await assert.rejects(odd.set('has space inside'), /invalid refresh token/u);
});

test('file store is private, atomic, refuses symlinks and ignores malformed content', async () => {
  const store = openSecretStore({ env: { LEETGPU_SECRET_STORE: 'file' }, platformName: 'linux' });
  assert.equal(store.kind, 'file');
  assert.equal(await store.get(), null);
  await store.set(secret);
  assert.equal(await store.get(), secret);
  assert.equal((await stat(__test.filePath)).mode & 0o777, 0o600);
  assert.equal((await stat(join(directory, 'config'))).mode & 0o777, 0o700);
  assert.equal(await readFile(__test.filePath, 'utf8'), `${secret}\n`);
  await assert.rejects(store.set('bad secret'), /invalid refresh token/u);
  assert.equal(await store.get(), secret);
  await writeFile(__test.filePath, 'not a token\n');
  assert.equal(await store.get(), null);
  await store.delete();
  await store.delete();
  const external = join(directory, 'external');
  await writeFile(external, `${secret}\n`);
  await symlink(external, __test.filePath);
  await assert.rejects(store.get(), /symbolic link/u);
  await store.delete();
  assert.equal(await readFile(external, 'utf8'), `${secret}\n`);
});

test('without a store nothing is kept and persistence is refused', async () => {
  const store = openSecretStore({ env: {}, platformName: 'linux' });
  assert.equal(store.kind, 'none');
  assert.equal(await store.get(), null);
  await store.delete();
  await assert.rejects(store.set(secret), /needs a secret store/u);
  await mkdir(join(directory, 'config'), { recursive: true });
});
