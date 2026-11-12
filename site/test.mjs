// Checks for the explainer page. Run from the repository root:  node site/test.mjs
//  1. the in-browser network gives the same scores as PyTorch on every sample image and round;
//  2. every number in the text of index.html equals the value computed from site/data/.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { prepare, forward, argmax } from "./js/net.js";
import { computeFacts } from "./js/facts.js";

const here = dirname(fileURLToPath(import.meta.url));
const D = {};
for (const n of ["write", "spectra", "net", "init", "read", "edit", "pilot_write"])
  D[n] = JSON.parse(readFileSync(join(here, "data", n + ".json"), "utf8"));
let failed = 0;

// 1. network
const buf = readFileSync(join(here, "data", "samples.bin"));
const raw = new Uint16Array(buf.buffer, buf.byteOffset, buf.length / 2);
const nb = D.net.used_bands.length, per = nb * 4096, n = D.net.labels.length;
let worst = 0, mismatched = 0;
for (const round of D.net.rounds) {
  const net = prepare(round);
  for (let k = 0; k < n; k++) {
    const img = Float32Array.from(raw.subarray(k * per, (k + 1) * per), (v) => v / 10000);
    const { logits } = forward(net, img, D.net.used_bands);
    logits.forEach((v, c) => { worst = Math.max(worst, Math.abs(v - round.expected_logits[k][c])); });
    if (argmax(logits) !== argmax(round.expected_logits[k])) mismatched++;
  }
}
console.log(`network: largest score difference from PyTorch ${worst.toFixed(4)}, predictions that differ: ${mismatched}`);
if (worst > 0.05 || mismatched) failed++;

// 2. numbers in the text
const F = computeFacts(D);
const html = readFileSync(join(here, "index.html"), "utf8");
const used = new Set();
for (const m of html.matchAll(/data-n="([^"]+)"[^>]*>([^<]*)</g)) {
  used.add(m[1]);
  if (!(m[1] in F)) { console.log(`unknown fact: ${m[1]}`); failed++; }
  else if (F[m[1]] !== m[2].trim()) { console.log(`text says "${m[2].trim()}" for ${m[1]}, data says "${F[m[1]]}"`); failed++; }
}
console.log(`text: ${used.size} named numbers checked against the data`);
if (process.argv.includes("--facts")) console.log(F);
process.exit(failed ? 1 : 0);
