// Browser thermal/DC vs Python: Motion board, U1=0.5 U3=0.3 U6..U9=0.15, /3V3 1 A U3.4 -> J8.1
// Python: max 78.9 C at U1, mean +44 K, drop 7.8 mV, Jmax 73 A/mm2.
//   node tests/web/check_thermal.mjs out/boards/BugBot_Motion_Board.kicad_pcb
import fs from "node:fs";
import { readKicadPcb } from "../../docs/app/kicad.js";
import { Board } from "../../docs/app/geom.js";
import { Grid, solveDC, solveThermal, report } from "../../docs/app/thermal.js";

const bd = new Board(readKicadPcb(fs.readFileSync(process.argv[2], "utf8")));
let t0 = Date.now();
const g = new Grid(bd, 0.25);
console.log(`grid ${g.nx} x ${g.ny} x ${g.nz} in ${Date.now() - t0} ms`);
t0 = Date.now();
const dc = solveDC(g, "/3V3", ["U3", "4"], ["J8", "1"], 1.0);
console.log(`dc in ${Date.now() - t0} ms (${dc.iterations} it)`);
t0 = Date.now();
const th = solveThermal(g, { U1: 0.5, U3: 0.3, U6: 0.15, U7: 0.15, U8: 0.15, U9: 0.15 }, { h: 10, tamb: 25, extraQ: dc.joule });
console.log(`thermal in ${Date.now() - t0} ms`);
console.log(report(g, th, dc));
