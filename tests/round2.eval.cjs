// node tests/round2.eval.cjs — measures P1..P7 from PROTOCOL.md exactly as written.
const { Brain, synthFrame, makeRng } = require('../core.js');
const W = 64, H = 64, L = W * H * 3;
const out = {};

function run({ T = 400, opts = {}, precision = false, phase = false, dup = false, seed = 11, onStep } = {}) {
  const b = new Brain(W, H); b.params.precision = precision; b.params.phase = phase;
  const rng = makeRng(seed), f = new Float32Array(L);
  for (let t = 0; t < T; t++) {
    const info = synthFrame(t, W, H, f, rng, opts);
    b.step(f); if (dup) b.step(f);
    if (onStep) onStep(t, b, f, info);
  }
  return b;
}
const pct = (x) => (100 * x).toFixed(1);

// P1 — duplicated frames collapse the model toward copy-last
{
  const clean = run({}).stats.skill;
  const d = run({ dup: true });
  // also: the duplicated-trained weights, frozen, on 100 fresh non-duplicated frames
  d.params.learn = false; const rng = makeRng(77), f = new Float32Array(L);
  for (let t = 400; t < 500; t++) { synthFrame(t, W, H, f, rng, {}); d.step(f); }
  out.P1 = { skillNoDup: clean, skillDup: run({ dup: true }).stats.skill, dupWeightsOnRealFrames: d.stats.skill,
             pass: run({ dup: true }).stats.skill < 0.10, kill: run({ dup: true }).stats.skill >= 0.25 };
}

// P2 — precision denoises the present (still noisy room)
{
  const rngClean = makeRng(1), clean = new Float32Array(L);
  let mPost = 0, mCam = 0, n = 0;
  run({ opts: { still: true }, precision: true, onStep: (t, b, f) => {
    if (t < 300) return;
    synthFrame(t, W, H, clean, rngClean, { still: true, noise: 0 });
    let a = 0, c = 0; for (let q = 0; q < L; q++) { a += (b.x[q] - clean[q]) ** 2; c += (f[q] - clean[q]) ** 2; }
    mPost += a / L; mCam += c / L; n++;
  } });
  const ratio = mPost / mCam;
  out.P2 = { msePosterior: mPost / n, mseCamera: mCam / n, ratio, pass: ratio <= 0.5, kill: ratio > 0.75 };
}

// P3 — precision does not hurt 1-step skill
{
  const r = {};
  for (const [name, opts] of [['still', { still: true }], ['moving', {}]]) {
    const off = run({ opts }).stats.skill, on = run({ opts, precision: true }).stats.skill;
    r[name] = { off, on, diffPts: 100 * (on - off) };
  }
  out.P3 = { ...r, pass: r.still.diffPts >= -2 && r.moving.diffPts >= -2 };
}

// P4 — a sudden change breaks through precision in its first frame
{
  const rngClean = makeRng(1), cNew = new Float32Array(L), cOld = new Float32Array(L);
  let res = null;
  run({ T: 301, opts: { still: true, newBlobAt: 300 }, precision: true, onStep: (t, b) => {
    if (t !== 300) return;
    synthFrame(t, W, H, cNew, rngClean, { still: true, noise: 0, newBlobAt: 300 });
    synthFrame(t, W, H, cOld, rngClean, { still: true, noise: 0 });
    let err = 0, con = 0;
    for (let p = 0; p < W * H; p++) {
      let d = 0; for (let c = 0; c < 3; c++) d += cNew[p * 3 + c] - cOld[p * 3 + c];
      if (d / 3 < 0.15) continue; // blob region: where the patch adds > 0.15
      for (let c = 0; c < 3; c++) { const q = p * 3 + c; err += Math.abs(b.x[q] - cNew[q]); con += cNew[q] - cOld[q]; }
    }
    res = { relError: err / con };
  } });
  out.P4 = { ...res, pass: res.relError < 0.30, kill: res.relError >= 0.50 };
}

// P5 — phase on smooth noiseless translation
{
  const opts = { scene: 'translate', vx: 1.5, noise: 0 };
  const lin = run({ opts }), ph = run({ opts, phase: true });
  out.P5 = { linear: lin.stats.skill, phase: ph.stats.skill, gainPts: 100 * (ph.stats.skill - lin.stats.skill),
             phaseWeights: [ph.w[20], ph.w[21]] };
  out.P5.pass = out.P5.gainPts >= 5; out.P5.kill = out.P5.gainPts < 2;
}

// P6 — phase on the default moving noisy room
{
  const lin = run({}), ph = run({ phase: true });
  out.P6 = { linear: lin.stats.skill, phase: ph.stats.skill, gainPts: 100 * (ph.stats.skill - lin.stats.skill),
             phaseWeights: [ph.w[20], ph.w[21]] };
}

// P7 — phase does not give object permanence across an occluder
{
  const opts = { scene: 'occluder', noise: 0.01 };
  function emergenceError(phase) {
    let sum = 0, n = 0, prevBehind = false, since = 99;
    run({ T: 400, opts, phase, onStep: (t, b, f, info) => {
      // "behind" while the blob centre is inside the bar
      const behind = info.bu > info.bar0 && info.bu < info.bar1;
      if (prevBehind && !behind) since = 0;
      prevBehind = behind;
      if (since < 5 && t > 60) { // first 5 frames after re-emergence, after warm-up
        let e = 0; for (let q = 0; q < L; q++) { const d = f[q] - b.pred[q]; e += d * d; }
        // b.pred is the prediction for the NEXT frame; record the error of the
        // prediction that was made for THIS frame instead: use brain.err (|f - pred_prev|)
        let e2 = 0; for (let p = 0; p < W * H; p++) e2 += b.err[p] * b.err[p];
        sum += e2 / (W * H); n++; since++;
      } else if (since < 99) since++;
    } });
    return sum / n;
  }
  const lin = emergenceError(false), ph = emergenceError(true);
  out.P7 = { linearEmergenceErr: lin, phaseEmergenceErr: ph, ratio: ph / lin,
             pass: Math.abs(ph / lin - 1) <= 0.15, surprise: ph / lin < 0.85 };
}

console.log(JSON.stringify(out, (k, v) => (typeof v === 'number' ? +v.toPrecision(4) : v), 2));
