// Drive the web app in headless Chrome (WebGPU) and save a screenshot for the README.
//   node tests/web/shot_app.mjs <app url> <out.png> [--board /out/boards/X.kicad_pcb] [--pair "P|N"] [--ports "A B C D"]
//        [--tie /3V3] [--res 0.15] [--base 0.3] [--emc clkMHz,A,trNs] [--wait s] [--snap 10] [--layer F.Cu]
import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const argv = process.argv.slice(2);
const opt = { board: "/out/boards/BugBot_Motion_Board.kicad_pcb", pair: "/USB_DP|/USB_DM", ports: "J1.A6 J9.10 J1.A7 J9.9", tie: "", res: "0.15", base: "0.3", emc: "", wait: "900", snap: "10", layer: "F.Cu" };
const [url, outPng] = [argv[0], argv[1]];
for (let i = 2; i < argv.length; i += 2) opt[argv[i].replace(/^--/, "")] = argv[i + 1];
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
  const evalJs = async expr => { const r = await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true }); return r.result && r.result.result ? r.result.result.value : (r.result && r.result.exceptionDetails ? "EXC " + (r.result.exceptionDetails.exception && r.result.exceptionDetails.exception.description || r.result.exceptionDetails.text) : null); };
  await send("Page.enable"); await send("Runtime.enable");
  await send("Emulation.setDeviceMetricsOverride", { width: 1500, height: 920, deviceScaleFactor: 1, mobile: false });
  await send("Page.navigate", { url });
  await sleep(2500);
  const emc = opt.emc ? opt.emc.split(",") : null;
  console.log(await evalJs(`(async () => { try {
    const $ = id => document.getElementById(id);
    const text = await (await fetch(${JSON.stringify(opt.board)})).text();
    await window.kf.loadText(text, ${JSON.stringify(path.basename(opt.board))});
    const ps = $("pairSel"); ps.value = ${JSON.stringify(opt.pair)}; ps.dispatchEvent(new Event("change"));
    $("ports").value = ${JSON.stringify(opt.ports)}; $("res").value = ${JSON.stringify(opt.res)}; $("base").value = ${JSON.stringify(opt.base)}; $("tie").value = ${JSON.stringify(opt.tie)};
    ${emc ? `$("emcOn").checked = true; $("emcClk").value = "${emc[0]}"; $("emcA").value = "${emc[1]}"; $("emcTr").value = "${emc[2]}";` : ""}
    $("setupBtn").click();
    for (let i = 0; i < 400; i++) { await new Promise(r => setTimeout(r, 250)); if (window.kf.setup) break; }
    $("runBtn").click();
    for (let i = 0; i < ${Math.round(+opt.wait * 2)}; i++) { await new Promise(r => setTimeout(r, 500)); if (window.kf.result) break; }
    $("showEdges").checked = true; $("edgeLayer").value = ${JSON.stringify(opt.layer)}; $("showMesh").checked = false;
    $("snapSl").value = ${+opt.snap}; window.kf.showSnap(${+opt.snap}); window.kf.zoomWindow();
    ${emc ? `$("emcBox").scrollIntoView(); document.querySelector("aside").scrollTop = $("emcRep").offsetTop - 60;` : ""}
    return $("status").textContent + " | " + $("resRep").innerText.split("\\n")[0] + " | " + $("engine").textContent + " | " + ($("emcRep").innerText.split("\\n").pop() || "");
  } catch (e) { return "PAGE ERROR " + (e.stack || e.message); } })()`));
  await sleep(1000);
  const shot = await send("Page.captureScreenshot", { format: "png" });
  fs.writeFileSync(outPng, Buffer.from(shot.result.data, "base64"));
  console.log("saved", outPng, fs.statSync(outPng).size, "bytes");
} finally { try { ws && ws.close(); } catch (e) {} proc.kill(); await sleep(300); try { fs.rmSync(profile, { recursive: true, force: true }); } catch (e) {} }
