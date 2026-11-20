import { homedir } from 'node:os';
import { join, resolve } from 'node:path';

// Keep the original directory name so existing installations retain their session.
export const configDirectory = resolve(process.env.LEETGPU_CONFIG_DIR
  || process.env.LEETGPU_HISTORY_CONFIG_DIR
  || join(process.env.XDG_CONFIG_HOME || join(homedir(), '.config'), 'leetgpu-history'));
export const dataDirectory = resolve(process.env.LEETGPU_DATA_DIR
  || join(process.env.XDG_DATA_HOME || join(homedir(), '.local', 'share'), 'leetgpu-history'));
