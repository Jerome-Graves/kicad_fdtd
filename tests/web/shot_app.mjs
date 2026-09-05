// Drive the web app in headless Chrome (WebGPU) and save a screenshot for the README.
//   node tests/web/shot_app.mjs http://localhost:8768/docs/ docs/webapp.png
import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const [url, outPng] = process.argv.slice(2);
const chrome = ["C:/Program Files/Google/Chrome/Application/chrome.exe", "/usr/bin/google-chrome"].find(p => fs.existsSync(p));
const port = 9333 + Math.floor(Math.random() * 500), profile = path.join(os.tmpdir(), "kf_shot_" + port);
const proc = spawn(chrome, ["--headless=new", "--enable-unsafe-webgpu", "--use-angle=d3d11", "--no-first-run", "--disable-gpu-sandbox", "--hide-scrollbars", `--user-data-dir=${profile}`, `--remote-debugging-port=${port}`, "--window-size=1500,920", "about:blank"], { stdio: "ignore" });
const sleep = ms => new Promise(r => setTimeout(r, ms));
let ws;
try {
  let targets = null;
  for (let i = 0; i < 50 && !targets; i++) { await sleep(200); try { targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json(); } catch (e) {} }
  ws = new WebSocket(targets.find(t => t.type === "page").webSocketDebuggerUrl);
  await new Promise((ok, err) => { ws.onopen = ok; ws.onerror = err; });
  let id = 0; const waiting = new Map();
  ws.onmessage = ev => { const m = JSON.parse(ev.data); if (m.id && waiting.has(m.id)) { waiting.get(m.id)(m); waiting.delete(m.id); } };
  const send = (method, params = {}) => new Promise(ok => { const q = ++id; waiting.set(q, ok); ws.send(JSON.stringify({ id: q, method, params })); });
  const evalJs = async expr => { const r = await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true }); return r.result && r.result.result ? r.result.result.value : (r.result && r.result.exceptionDetails ? "EXC " + JSON.stringify(r.result.exceptionDetails.exception) : null); };
  await send("Page.enable"); await send("Runtime.enable");
  await send("Emulation.setDeviceMetricsOverride", { width: 1500, height: 920, deviceScaleFactor: 1, mobile: false });
  await send("Page.navigate", { url });
  await sleep(2500);
  console.log(await evalJs(`(async () => {
    const $ = id => document.getElementById(id);
    const text = await (await fetch("/out/boards/BugBot_Motion_Board.kicad_pcb")).text();
    await window.kf.loadText(text, "BugBot_Motion_Board.kicad_pcb");
    const ps = $("pairSel"); ps.value = "/USB_DP|/USB_DM"; ps.dispatchEvent(new Event("change"));
    $("ports").value = "J1.A6 J9.10 J1.A7 J9.9"; $("res").value = "0.15"; $("base").value = "0.3";
    $("setupBtn").click();
    for (let i = 0; i < 200; i++) { await new Promise(r => setTimeout(r, 250)); if (window.kf.setup) break; }
    $("runBtn").click();
    for (let i = 0; i < 600; i++) { await new Promise(r => setTimeout(r, 500)); if (window.kf.result) break; }
    $("showEdges").checked = true; $("edgeLayer").value = "F.Cu"; $("showMesh").checked = false;
    $("snapSl").value = 10; window.kf.showSnap(10); window.kf.zoomWindow();
    return $("status").textContent + " | " + $("resRep").innerText.split("\\n")[0] + " | " + $("engine").textContent;
  })()`));
  await sleep(800);
  const shot = await send("Page.captureScreenshot", { format: "png" });
  fs.writeFileSync(outPng, Buffer.from(shot.result.data, "base64"));
  console.log("saved", outPng, fs.statSync(outPng).size, "bytes");
} finally { try { ws && ws.close(); } catch (e) {} proc.kill(); await sleep(300); try { fs.rmSync(profile, { recursive: true, force: true }); } catch (e) {} }
