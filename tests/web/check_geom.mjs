// Stats and port sites from the browser modules vs the Python side.
//   node tests/web/check_geom.mjs out/boards/BugBot_Motion_Board.kicad_pcb
import fs from "node:fs";
import { readKicadPcb } from "../../docs/app/kicad.js";
import { Board, formatStats } from "../../docs/app/geom.js";

const [pcb, ...ports] = process.argv.slice(2);
let t0 = Date.now();
const g = readKicadPcb(fs.readFileSync(pcb, "utf8"));
const bd = new Board(g);
console.log("board built in", Date.now() - t0, "ms; ref net", bd.refNet, "; stackup", bd.stackupSource);
t0 = Date.now();
const s = bd.stats();
console.log(formatStats(s));
console.log("stats in", Date.now() - t0, "ms");
for (const rp of (ports.length ? ports : ["J1.A6", "J9.10", "J1.A7", "J9.9"])) {
  const [ref, num] = rp.split(".");
  try {
    t0 = Date.now();
    const site = bd.portSite(ref, num);
    console.log("port", rp, site ? `at (${site.x.toFixed(2)}, ${site.y.toFixed(2)}) ${site.sig} over ${site.plane} (${site.ref}) shift ${site.shift.toFixed(1)}` : "NONE", Date.now() - t0, "ms");
  } catch (e) { console.log("port", rp, e.message); }
}
