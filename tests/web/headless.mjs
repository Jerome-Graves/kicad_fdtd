// Run a page in headless Chrome with WebGPU and print its <pre id="out"> when it says DONE.
//   node tests/web/headless.mjs "http://localhost:8768/docs/selftest.html?res=0.1" [timeout_s]
import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const url = process.argv[2], timeoutS = +(process.argv[3] || 300);
const chrome = ["C:/Program Files/Google/Chrome/Application/chrome.exe", "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe", "/usr/bin/google-chrome", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"].find(p => fs.existsSync(p));
if (!chrome) { console.error("no Chrome found"); process.exit(2); }
const port = 9333 + Math.floor(Math.random() * 500);
const profile = path.join(os.tmpdir(), "kf_headless_" + port);
const proc = spawn(chrome, ["--headless=new", "--enable-unsafe-webgpu", "--use-angle=d3d11", "--no-first-run", "--disable-gpu-sandbox", `--user-data-dir=${profile}`, `--remote-debugging-port=${port}`, "about:blank"], { stdio: "ignore" });
const sleep = ms => new Promise(r => setTimeout(r, ms));
let ws;
try {
  let targets = null;
  for (let i = 0; i < 50 && !targets; i++) { await sleep(200); try { targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json(); } catch (e) {} }
  const page = targets.find(t => t.type === "page");
  ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((ok, err) => { ws.onopen = ok; ws.onerror = err; });
  let id = 0; const waiting = new Map();
  ws.onmessage = ev => { const m = JSON.parse(ev.data); if (m.id && waiting.has(m.id)) { waiting.get(m.id)(m); waiting.delete(m.id); } };
  const send = (method, params = {}) => new Promise(ok => { const q = ++id; waiting.set(q, ok); ws.send(JSON.stringify({ id: q, method, params })); });
  await send("Page.enable"); await send("Runtime.enable");
  await send("Page.navigate", { url });
  const t0 = Date.now(); let last = "";
  while (Date.now() - t0 < timeoutS * 1000) {
    await sleep(1000);
    const r = await send("Runtime.evaluate", { expression: "(document.getElementById('out')||{}).textContent || ''", returnByValue: true });
    const txt = (r.result && r.result.result && r.result.result.value) || "";
    if (txt !== last) { const fresh = txt.slice(last.length); process.stdout.write(fresh.endsWith("\n") ? fresh : fresh + "\n"); last = txt; }
    if (/(^|\n)DONE/.test(txt)) break;
  }
  if (!/(^|\n)DONE/.test(last)) console.log("TIMEOUT after", timeoutS, "s");
} finally {
  try { ws && ws.close(); } catch (e) {}
  proc.kill();
  await sleep(300);
  try { fs.rmSync(profile, { recursive: true, force: true }); } catch (e) {}
}
