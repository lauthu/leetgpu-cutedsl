import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { parseSession } from './auth.mjs';

// Read a single known localStorage key in an already-open LeetGPU tab.
// Chrome itself enforces its Apple Events JavaScript preference. Never change it.
const expression = `(() => {
  if (location.origin !== 'https://leetgpu.com') return null;
  try {
    const value = JSON.parse(localStorage.getItem('sb-yhdtysacdkqoquvkdwdd-auth-token'));
    const session = value?.currentSession || value?.session || value;
    if (!session?.access_token) return null;
    return JSON.stringify({access_token: session.access_token, expires_at: session.expires_at});
  } catch { return null; }
})()`;

const appleScript = `
if application "Google Chrome" is not running then return "null"
tell application "Google Chrome"
  repeat with browserWindow in windows
    repeat with browserTab in tabs of browserWindow
      set tabAddress to URL of browserTab
      if tabAddress is "https://leetgpu.com" or tabAddress starts with "https://leetgpu.com/" then
        set captured to execute browserTab javascript ${JSON.stringify(expression)}
        if captured is not missing value and captured is not "null" and captured is not "" then return captured
      end if
    end repeat
  end repeat
end tell
return "null"
`;

export async function existingChromeSession({ run = promisify(execFile), platformName = process.platform } = {}) {
  if (platformName !== 'darwin') throw new Error('Existing-browser capture currently requires macOS Chrome.');
  let stdout;
  try {
    ({ stdout } = await run('osascript', ['-e', appleScript], {
      encoding: 'utf8', timeout: 20000, maxBuffer: 100000,
    }));
  } catch (error) {
    // Do not print the subprocess output, which could contain session data.
    const stderr = String(error.stderr ?? '');
    if (/JavaScript.*Apple|Apple.*JavaScript|JavaScript.*关闭|JavaScript.*停用/iu.test(stderr)) {
      throw new Error('Chrome blocks Apple Events JavaScript. Enable View > Developer > Allow JavaScript from Apple Events, then retry login --existing-browser.');
    }
    if (stderr.includes('-1743')) throw new Error('macOS blocked Chrome automation. Allow the Chrome automation request, then retry login --existing-browser.');
    throw new Error('Could not read the open LeetGPU tab. Check Chrome automation permission, or use login for local manual import.');
  }
  const session = parseSession(stdout.trim());
  if (!session) throw new Error('No valid LeetGPU session found. Sign in at https://leetgpu.com in normal Chrome and keep that tab open.');
  return session;
}
