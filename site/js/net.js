// The network the model wrote, run in the browser exactly as PyTorch runs it:
//   conv 3x3 (13 bands -> 16) -> ReLU -> average pool 4 -> conv 3x3 (16 -> 32) -> ReLU -> mean -> linear (32 -> 10)

const S = 64, P = 16;

function taps(weight, nOut, nIn) {
  const out = [];
  for (let o = 0; o < nOut; o++)
    for (let i = 0; i < nIn; i++)
      for (let k = 0; k < 9; k++) {
        const w = weight[(o * nIn + i) * 9 + k];
        if (w !== 0) out.push([o, i, Math.floor(k / 3) - 1, (k % 3) - 1, w]);
      }
  return out;
}

export function prepare(round) {
  return {
    t1: taps(round.conv1_weight, 16, 13), b1: round.conv1_bias,
    t2: taps(round.conv2_weight, 32, 16), b2: round.conv2_bias,
    fw: round.fc_weight, fb: round.fc_bias,
  };
}

function conv(input, slotOf, tapList, bias, nOut, size) {
  const n = size * size, out = new Float64Array(nOut * n);
  for (let o = 0; o < nOut; o++) out.fill(bias[o], o * n, (o + 1) * n);
  for (const [o, i, dy, dx, w] of tapList) {
    const src = slotOf(i) * n, dst = o * n;
    const y0 = Math.max(0, -dy), y1 = Math.min(size, size - dy);
    const x0 = Math.max(0, -dx), x1 = Math.min(size, size - dx);
    for (let y = y0; y < y1; y++) {
      const a = dst + y * size, b = src + (y + dy) * size + dx;
      for (let x = x0; x < x1; x++) out[a + x] += w * input[b + x];
    }
  }
  for (let j = 0; j < out.length; j++) if (out[j] < 0) out[j] = 0;
  return out;
}

// image: Float32Array [usedBands.length * 64 * 64] of reflectance; usedBands: band index of each slot
export function forward(net, image, usedBands) {
  const slot = new Map(usedBands.map((b, j) => [b, j]));
  const a1 = conv(image, (b) => slot.get(b), net.t1, net.b1, 16, S);
  const pooled = new Float64Array(16 * P * P);
  for (let c = 0; c < 16; c++)
    for (let y = 0; y < P; y++)
      for (let x = 0; x < P; x++) {
        let s = 0;
        for (let v = 0; v < 4; v++)
          for (let u = 0; u < 4; u++) s += a1[c * S * S + (y * 4 + v) * S + x * 4 + u];
        pooled[c * P * P + y * P + x] = s / 16;
      }
  const a2 = conv(pooled, (i) => i, net.t2, net.b2, 32, P);
  const f = new Float64Array(32);
  for (let c = 0; c < 32; c++) {
    let s = 0;
    for (let j = 0; j < P * P; j++) s += a2[c * P * P + j];
    f[c] = s / (P * P);
  }
  const logits = new Float64Array(10);
  for (let k = 0; k < 10; k++) {
    let s = net.fb[k];
    for (let c = 0; c < 32; c++) s += net.fw[k * 32 + c] * f[c];
    logits[k] = s;
  }
  return { a1, f, logits };
}

export function softmax(logits, scale = 1) {
  const m = Math.max(...logits);
  const e = Array.from(logits, (v) => Math.exp((v - m) * scale));
  const z = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / z);
}

export function argmax(a) {
  let best = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[best]) best = i;
  return best;
}
