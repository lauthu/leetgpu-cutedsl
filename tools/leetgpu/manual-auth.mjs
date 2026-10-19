import { createServer } from 'node:http';
import { randomBytes } from 'node:crypto';
import { spawn } from 'node:child_process';
import { parseSession } from './auth.mjs';

const maxBodyBytes = 100000;
const invalidMessage = '内容无效或已过期。请重新登录 LeetGPU，复制完整 Value 后重试。';

export function parsePastedSession(value) {
  if (typeof value !== 'string' || Buffer.byteLength(value, 'utf8') > maxBodyBytes) return null;
  const trimmed = value.trim();
  if (!trimmed) return null;
  try {
    const parsed = JSON.parse(trimmed);
    if (typeof parsed === 'string') return parseSession({ access_token: parsed });
    return parseSession(parsed);
  } catch {
    if (!/^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$/u.test(trimmed)) return null;
    return parseSession({ access_token: trimmed });
  }
}

function page(nonce) {
  return `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>连接 LeetGPU 历史记录</title>
<style nonce="${nonce}">
body{font:16px/1.65 system-ui,sans-serif;color:#17212b;background:#f3f5f7;margin:0;padding:32px 20px}
main{max-width:760px;margin:auto;background:white;padding:30px;border-radius:14px}h1{font-size:26px;margin-top:0}
li{margin:12px 0}code{overflow-wrap:anywhere;background:#eef2f6;padding:2px 5px;border-radius:4px}
textarea{display:block;box-sizing:border-box;width:100%;height:145px;padding:12px;border:1px solid #8794a1;border-radius:7px;font:14px monospace;margin:10px 0 16px}
button{font:inherit;background:#116637;color:white;border:0;border-radius:6px;padding:10px 18px;cursor:pointer}
button:disabled{opacity:.55;cursor:default}.note{color:#526170}#status{min-height:28px;overflow-wrap:anywhere}
</style></head><body><main>
<h1>连接 LeetGPU 历史记录</h1>
<p>使用平时的浏览器登录。登录信息只发送到此临时接收页面；远程使用时请通过 SSH 端口转发连接。</p>
<ol>
<li>在新标签页打开 <a href="https://leetgpu.com" target="_blank" rel="noopener noreferrer">LeetGPU</a>，使用你的账号登录。</li>
<li>在 LeetGPU 标签页打开 Chrome 开发者工具：Mac 按 <code>⌥⌘I</code>，Windows / Linux 按 <code>F12</code> 或 <code>Ctrl+Shift+I</code>。</li>
<li>选择 <strong>Application（应用）</strong> → <strong>Local Storage（本地存储空间）</strong> → <code>https://leetgpu.com</code>。</li>
<li>找到键 <code>sb-yhdtysacdkqoquvkdwdd-auth-token</code>，复制该行的完整 <strong>Value（值）</strong>。可右键该行选择复制值，或双击 Value 后全选复制。</li>
<li>回到此页面，将完整 JSON 粘贴到下方并提交。也可粘贴单独的 <code>access_token</code>（JWT）。</li>
</ol>
<form id="login" autocomplete="off"><label for="session">LeetGPU 会话 Value 或 access_token</label>
<textarea id="session" autocomplete="off" spellcheck="false" autocapitalize="off" aria-label="LeetGPU 会话 Value 或 access_token"></textarea>
<button id="submit" type="submit">连接历史记录</button></form>
<p id="status" role="status" aria-live="polite"></p>
<p class="note">工具只保留 access token 和过期时间；不会保留 refresh token、读取剪贴板或读取浏览器配置。提交成功后，此接收页面会停止服务。</p>
</main><script nonce="${nonce}">
const form=document.getElementById('login');
const input=document.getElementById('session');
const button=document.getElementById('submit');
const status=document.getElementById('status');
form.addEventListener('submit',async event=>{
  event.preventDefault();button.disabled=true;status.textContent='正在连接…';
  try{
    const response=await fetch(location.pathname,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({value:input.value}),credentials:'omit',cache:'no-store'});
    const result=await response.json();
    if(response.ok&&result.ok){input.value='';form.hidden=true;status.textContent='会话已接收，返回终端确认认证结果。';}
    else{status.textContent=result.error||'连接失败，请重试。';button.disabled=false;}
  }catch{input.value='';status.textContent='本机接收页面已关闭或超时。请返回终端重新启动登录。';button.disabled=false;}
});
addEventListener('pagehide',()=>{input.value='';});
</script></body></html>`;
}

async function createLoginReceiver({ timeoutMs = 600000, port = 0 } = {}) {
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) throw new Error('Login timeout must be a positive number.');
  if (!Number.isInteger(port) || port < 0 || port > 65535) throw new Error('Login port must be an integer between 0 and 65535.');
  const path = `/${randomBytes(32).toString('hex')}`;
  const nonce = randomBytes(24).toString('base64');
  let expectedHost;
  let expectedOrigin;
  let settled = false;
  let resolveResult;
  let rejectResult;
  let timer;
  const result = new Promise((resolve, reject) => { resolveResult = resolve; rejectResult = reject; });
  // Startup/opening can take time; prevent an early timeout from becoming an
  // unhandled rejection before the caller has begun awaiting the result.
  result.catch(() => {});
  const server = createServer((request, response) => {
    const headers = {
      'Cache-Control': 'no-store', 'Pragma': 'no-cache', 'Referrer-Policy': 'no-referrer',
      'X-Content-Type-Options': 'nosniff', 'X-Frame-Options': 'DENY',
      'Content-Security-Policy': `default-src 'none'; script-src 'nonce-${nonce}'; style-src 'nonce-${nonce}'; connect-src 'self'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'`,
    };
    const reply = (code, body, type = 'application/json; charset=utf-8') => {
      response.writeHead(code, { ...headers, 'Content-Type': type });
      response.end(type.startsWith('application/json') ? JSON.stringify(body) : body);
    };
    if (request.headers.host !== expectedHost || request.url !== path || settled) {
      request.resume(); reply(404, { error: '页面不存在。' }); return;
    }
    if (request.method === 'GET') { reply(200, page(nonce), 'text/html; charset=utf-8'); return; }
    if (request.method !== 'POST') { request.resume(); reply(405, { error: '请求方式不支持。' }); return; }
    if (request.headers.origin !== expectedOrigin) { request.resume(); reply(403, { error: '请求来源无效。' }); return; }
    if (!/^application\/json(?:\s*;\s*charset=utf-8)?$/iu.test(request.headers['content-type'] || '')) {
      request.resume(); reply(415, { error: '请求格式无效。' }); return;
    }
    const chunks = [];
    let length = 0;
    let rejected = false;
    request.on('data', chunk => {
      if (rejected) return;
      length += chunk.length;
      if (length > maxBodyBytes) { rejected = true; chunks.length = 0; reply(413, { error: '内容过长，请仅复制会话 Value。' }); return; }
      chunks.push(chunk);
    });
    request.on('error', () => { chunks.length = 0; });
    request.on('end', () => {
      if (rejected || settled) return;
      let session;
      try { session = parsePastedSession(JSON.parse(Buffer.concat(chunks).toString('utf8')).value); } catch { /* Never echo pasted input or parser errors. */ }
      chunks.length = 0;
      if (!session) { reply(400, { error: invalidMessage }); return; }
      response.once('finish', () => finish(undefined, session));
      reply(200, { ok: true });
    });
  });
  server.requestTimeout = 15000;
  server.headersTimeout = 10000;
  const interrupt = () => finish(new Error('LeetGPU login canceled.'));
  function finish(error, session) {
    if (settled) return;
    settled = true;
    clearTimeout(timer);
    process.removeListener('SIGINT', interrupt);
    process.removeListener('SIGTERM', interrupt);
    if (error) {
      server.close();
      server.closeAllConnections();
    } else {
      // Give the successful response time to reach fetch before shutting down
      // keep-alive sockets. No further submissions are accepted once settled.
      const shutdown = setTimeout(() => { server.close(); server.closeAllConnections(); }, 200);
      shutdown.unref();
    }
    if (error) rejectResult(error); else resolveResult(session);
  }
  await new Promise((resolve, reject) => {
    const failed = () => reject(new Error('Could not start the loopback login receiver. Check that the requested port is available.'));
    server.once('error', failed);
    server.listen(port, '127.0.0.1', () => { server.removeListener('error', failed); resolve(); });
  });
  server.on('error', () => finish(new Error('The local LeetGPU login page could not continue.')));
  expectedHost = `127.0.0.1:${server.address().port}`;
  expectedOrigin = `http://${expectedHost}`;
  process.once('SIGINT', interrupt);
  process.once('SIGTERM', interrupt);
  timer = setTimeout(() => finish(new Error('LeetGPU login timed out. Run login again to retry.')), timeoutMs);
  return { url: `${expectedOrigin}${path}`, result, cancel: () => finish(new Error('LeetGPU login canceled.')) };
}

async function openDefaultBrowser(url) {
  const command = process.platform === 'darwin' ? 'open' : 'xdg-open';
  const args = process.platform === 'darwin' ? ['-a', 'Google Chrome', url] : [url];
  await new Promise((resolve, reject) => {
    const child = spawn(command, args, { stdio: 'ignore' });
    const timer = setTimeout(() => { child.kill(); reject(new Error('Could not open the local login page.')); }, 5000);
    child.once('error', () => { clearTimeout(timer); reject(new Error('Could not open the local login page.')); });
    child.once('exit', code => { clearTimeout(timer); if (code === 0) resolve(); else reject(new Error('Could not open the local login page.')); });
  });
}

export async function manualLogin({
  timeoutMs = 600000,
  onStatus = () => {},
  openBrowser = true,
  port = 0,
  openBrowserFn = openDefaultBrowser,
  createReceiver = createLoginReceiver,
} = {}) {
  const receiver = await createReceiver({ timeoutMs, port });
  try {
    const receiverPort = new URL(receiver.url).port;
    onStatus(`Open this temporary login page in your local browser: ${receiver.url}`);
    onStatus(`If this CLI runs on a remote machine, first run on your local machine: ssh -N -L 127.0.0.1:${receiverPort}:127.0.0.1:${receiverPort} <user@remote-host>`);
    onStatus(`Use the same local port (${receiverPort}) and the exact URL above; then sign in to LeetGPU and follow the page instructions.`);
    if (openBrowser) {
      try {
        await openBrowserFn(receiver.url);
      } catch {
        // A headless host or unavailable browser must not discard the receiver.
        // Do not surface opener errors, which can include unexpected arguments.
        onStatus('Could not open a browser automatically. The login page is still waiting; open the URL above manually.');
      }
    }
    return await receiver.result;
  } catch (error) {
    receiver.cancel();
    throw error;
  }
}

export const __test = { createLoginReceiver };
