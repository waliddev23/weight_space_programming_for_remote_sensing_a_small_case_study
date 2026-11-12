// Every number quoted in the article's text. Each value is computed from site/data/*.json,
// and index.html refers to it by name: <span data-n="w3_final">59.5%</span>.
// `node site/test.mjs` checks that the text in the page matches these values.

const pct = (x, d = 1) => (x * 100).toFixed(d) + "%";
const pts = (x, d = 1) => (x >= 0 ? "+" : "−") + Math.abs(x * 100).toFixed(d);
const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
const int = (n) => n.toLocaleString("en-US");

function corr(a, b) {
  const ma = mean(a), mb = mean(b);
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < a.length; i++) {
    sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) ** 2; sbb += (b[i] - mb) ** 2;
  }
  return sab / Math.sqrt(saa * sbb);
}

export function computeFacts(D) {
  const F = {};
  const w3 = D.write.w3, w2 = D.write.w2, w1 = D.write.w1;
  const t3 = w3.rounds.map((r) => r.test.accuracy), t2 = w2.rounds.map((r) => r.test.accuracy);
  const last = w3.rounds[w3.rounds.length - 1].test;
  const C = D.write.classes;
  const recall = (name) => last.recall[C.indexOf(name)];

  // Part 1
  F.w3_first = pct(t3[0]); F.w3_final = pct(t3[3]); F.w3_peak = pct(Math.max(...t3));
  F.w3_r1 = pct(t3[1]); F.w3_r2 = pct(t3[2]);
  F.w3_gain1 = pts(t3[1] - t3[0]); F.w3_gain2 = pts(t3[2] - t3[1]); F.w3_gain3 = pts(t3[3] - t3[2]);
  F.w3_constant = pct(w3.baselines.constant); F.w3_random = pct(w3.baselines.random);
  F.w3_centroid = pct(w3.baselines.centroid); F.w3_trained = pct(w3.baselines.trained);
  F.w3_share = Math.round((t3[3] / w3.baselines.trained) * 100) + "%";
  F.w2_first = pct(t2[0]); F.w2_final = pct(t2[3]);
  F.w2_centroid = pct(w2.baselines.centroid); F.w2_trained = pct(w2.baselines.trained);
  F.w2_share = Math.round((t2[3] / w2.baselines.trained) * 100) + "%";
  F.w1_acc = pct(w1.accuracy); F.w1_trained = pct(w1.baselines.trained);
  F.w3_margin = pts(t3[3] - w3.baselines.centroid); F.w2_margin = pts(t2[3] - w2.baselines.centroid);
  F.n_test = int(last.n); F.n_practice = int(w3.rounds[0].practice.n);
  F.n_train = int(D.spectra.n_train);
  for (const [k, name] of [["sea", "SeaLake"], ["forest", "Forest"], ["river", "River"], ["pasture", "Pasture"],
    ["highway", "Highway"], ["residential", "Residential Buildings"], ["industrial", "Industrial Buildings"]])
    F["rec_" + k] = Math.round(recall(name) * 100) + "%";
  const R = D.net.rounds[0];
  F.n_weights = int(R.conv1_weight.length + R.conv1_bias.length + R.conv2_weight.length + R.conv2_bias.length
    + R.fc_weight.length + R.fc_bias.length);
  F.n_samples = String(D.net.labels.length);

  // what the model believed about the spectra
  const b0 = D.spectra.believed[0];
  const bel = [], mea = [];
  b0.bands.forEach((band, j) => b0.values[j].forEach((v, c) => { bel.push(v); mea.push(D.spectra.measured[c][band]); }));
  F.spec_corr = corr(bel, mea).toFixed(2);
  F.spec_err = (mean(bel.map((v, i) => Math.abs(v - mea[i]))) * 100).toFixed(1);

  // Part 2
  const run = (b, init) => D.init.runs.filter((r) => r.budget === b && r.init === init).sort((x, y) => x.seed - y.seed);
  for (const [tag, b] of [["1", 0.01], ["10", 0.1], ["100", 1.0]]) {
    const k = run(b, "kaiming"), a = run(b, "fable_as_written"), s = run(b, "fable_rescaled");
    F[`init${tag}_k`] = pct(mean(k.map((r) => r.test)));
    F[`init${tag}_a`] = pct(mean(a.map((r) => r.test)));
    F[`init${tag}_s`] = pct(mean(s.map((r) => r.test)));
    F[`init${tag}_gain`] = pts(mean(s.map((r) => r.test)) - mean(k.map((r) => r.test)));
    F[`init${tag}_n`] = int(k[0].n_train);
    F[`init${tag}_wins`] = String(s.filter((r, i) => r.test > k[i].test).length);
  }
  F.init_stalled = run(0.1, "fable_as_written").filter((r) => Math.abs(r.test - D.init.fable_untrained) < 0.01).length + " of 3";
  F.fc_max = int(Math.round(Math.max(...D.net.rounds[3].fc_weight.map(Math.abs)) / 100) * 100);
  F.rescale = String(Math.round(1 / D.net.rescale));
  F.sure_written = pct(D.net.sure.as_written, 0); F.sure_rescaled = pct(D.net.sure.rescaled, 0);

  // Part 3
  const E = D.edit, M = E.methods.A.methods;
  F.read_score = D.read.models.filter((m) => m.correct).length + " of " + D.read.models.length;
  F.edit_orig = pct(E.behaviour.original.accuracy); F.edit_damaged = pct(E.behaviour.damaged.accuracy);
  F.edit_repaired = pct(E.behaviour.repaired.accuracy); F.edit_zero = pct(M.zero_rows);
  F.edit_norm = pct(M.norm_match); F.edit_svd = pct(M.global_svd); F.edit_clip = pct(M.clip_range);
  F.edit_extra = pts(M.submitted - M.zero_rows);
  F.edit_nbad = String(E.row_std.damaged.filter((v) => v > 3 * E.healthy_sd).length);
  F.edit_nfilters = String(E.shape[0]);
  const fa = D.pilot_write.runs.filter((r) => r.init === "fable"), ka = D.pilot_write.runs.filter((r) => r.init === "kaiming");
  F.pw_fable = pct(mean(fa.map((r) => r.test)), 2); F.pw_kaiming = pct(mean(ka.map((r) => r.test)), 2);
  const d = fa.map((r, i) => r.test - ka[i].test);
  F.pw_min = (Math.min(...d) * 100).toFixed(1); F.pw_max = (Math.max(...d) * 100).toFixed(1);
  return F;
}
