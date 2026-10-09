// node tests/core.test.cjs  — honest checks on the Perjantai core
const assert = require('assert');
const { Brain, synthFrame, makeRng } = require('../core.js');
const W = 64, H = 64;
let pass = 0;
function test(name, fn) { fn(); pass++; console.log('ok  ', name); }
function feed(b, T, opts, rng = makeRng(11), t0 = 0) {
  const f = new Float32Array(W * H * 3);
  for (let t = t0; t < t0 + T; t++) { synthFrame(t, W, H, f, rng, opts); b.step(f); }
  return f;
}

test('starts as copy-last-frame (skill 0 before learning)', () => {
  const b = new Brain(W, H); b.params.learn = false; feed(b, 200, {});
  assert(Math.abs(b.stats.skill) < 1e-6, 'skill ' + b.stats.skill);
});

test('still noisy room: beats copy-last-frame, near the ideal 0.5', () => {
  // copy-last error = 2 sigma^2, ideal denoised predictor = sigma^2 -> skill <= ~0.5
  const b = new Brain(W, H); feed(b, 400, { still: true });
  assert(b.stats.skill > 0.4 && b.stats.skill < 0.55, 'skill ' + b.stats.skill);
  assert(b.w[18] > 0.2, 'memory weight should carry the static scene, got ' + b.w[18]);
});

test('noiseless motion: learns motion continuation (difference taps positive)', () => {
  const b = new Brain(W, H); feed(b, 400, { noise: 0 });
  let dsum = 0; for (let k = 9; k < 18; k++) dsum += b.w[k];
  assert(b.stats.skill > 0.8, 'skill ' + b.stats.skill);
  assert(dsum > 0.3, 'difference taps sum ' + dsum);
});

test('moving noisy room: positive skill', () => {
  const b = new Brain(W, H); feed(b, 400, {});
  assert(b.stats.skill > 0.25, 'skill ' + b.stats.skill);
});

test('gaze follows the moving blob (mean distance far below chance)', () => {
  const b = new Brain(W, H); const rng = makeRng(11); const f = new Float32Array(W * H * 3);
  let dsum = 0, csum = 0, n = 0; const crng = makeRng(99);
  for (let t = 0; t < 400; t++) {
    const pos = synthFrame(t, W, H, f, rng, {}); b.step(f);
    if (t >= 150) {
      dsum += Math.hypot(b.gaze.u - pos.b1.u, b.gaze.v - pos.b1.v);
      csum += Math.hypot(crng() - pos.b1.u, crng() - pos.b1.v); n++;
    }
  }
  const d = dsum / n, chance = csum / n;
  console.log(`      gaze-to-blob ${d.toFixed(3)}  vs random point ${chance.toFixed(3)}`);
  assert(d < 0.5 * chance, `gaze ${d} chance ${chance}`);
});

test('dream (closed loop) loses fine detail, drifts, and does not learn; waking restores', () => {
  // Measured, not assumed: the learned filter keeps unit gain at DC, so the dream does
  // NOT collapse to grey. It smooths away fine detail and lets the layout drift.
  const b = new Brain(W, H); const rng = makeRng(5); feed(b, 300, {}, rng);
  const detailCam = Brain.detail(b.cam, W, H), w0 = Array.from(b.w);
  b.params.reality = 0; feed(b, 300, {}, rng, 300);
  const detailDream = Brain.detail(b.x, W, H);
  console.log(`      detail camera ${detailCam.toFixed(4)}  dream ${detailDream.toFixed(4)}  drift ${b.stats.drift.toExponential(2)}`);
  assert(detailDream < 0.5 * detailCam, `detail cam ${detailCam} dream ${detailDream}`);
  assert(b.stats.drift > 0.002, 'drift ' + b.stats.drift);
  assert.deepStrictEqual(Array.from(b.w), w0, 'weights must not learn from the dream');
  b.params.reality = 1; feed(b, 60, {}, rng, 600);
  assert(b.stats.drift < 0.002, 'drift after waking ' + b.stats.drift);
});

test('a still room dreams almost true: the memory tap anchors it', () => {
  const b = new Brain(W, H); const rng = makeRng(5); feed(b, 300, { still: true }, rng);
  b.params.reality = 0; feed(b, 300, { still: true }, rng, 300);
  assert(b.stats.drift < 0.002, 'drift ' + b.stats.drift);
});

test('k-step rollout loses detail with horizon (the further future is blurrier)', () => {
  const b = new Brain(W, H); const rng = makeRng(3); feed(b, 300, {}, rng);
  b.params.horizon = 1; feed(b, 1, {}, rng, 300); const s1 = Brain.detail(b.predK, W, H);
  b.params.horizon = 20; feed(b, 1, {}, rng, 301); const s20 = Brain.detail(b.predK, W, H);
  console.log(`      detail h1 ${s1.toFixed(4)}  h20 ${s20.toFixed(4)}`);
  assert(s20 < 0.7 * s1, `detail h1 ${s1} h20 ${s20}`);
});

test('attention-gated learning still learns and stays finite', () => {
  const b = new Brain(W, H); b.params.attentionGatesLearning = true; feed(b, 300, {});
  assert(b.w.every(Number.isFinite)); assert(b.stats.skill > 0.2, 'skill ' + b.stats.skill);
});

test('precision: the belief is much closer to the clean scene than the camera', () => {
  const b = new Brain(W, H); b.params.precision = true; const rng = makeRng(11), rc = makeRng(1);
  const f = new Float32Array(W * H * 3), clean = new Float32Array(W * H * 3); let a = 0, c = 0;
  for (let t = 0; t < 300; t++) {
    synthFrame(t, W, H, f, rng, { still: true }); b.step(f);
    if (t >= 250) { synthFrame(t, W, H, clean, rc, { still: true, noise: 0 });
      for (let q = 0; q < f.length; q++) { a += (b.x[q] - clean[q]) ** 2; c += (f[q] - clean[q]) ** 2; } }
  }
  assert(a < 0.3 * c, `posterior ${a} camera ${c}`);
});

test('precision: a sudden object breaks through in its first frame', () => {
  const b = new Brain(W, H); b.params.precision = true; const rng = makeRng(11);
  const f = new Float32Array(W * H * 3);
  for (let t = 0; t <= 250; t++) { synthFrame(t, W, H, f, rng, { still: true, newBlobAt: 250 }); b.step(f); }
  // centre of the new patch (u 0.7, v 0.3)
  const p = Math.floor(0.3 * H) * W + Math.floor(0.7 * W);
  assert(b.gain[p] > 0.8, 'gain at the new object ' + b.gain[p]);
  assert(Math.abs(b.x[p * 3] - f[p * 3]) < 0.08, 'belief vs eye ' + Math.abs(b.x[p * 3] - f[p * 3]));
});

test('phase moon runs, stays finite, and is used on fast translation', () => {
  const b = new Brain(W, H); b.params.phase = true; const rng = makeRng(11); const f = new Float32Array(W * H * 3);
  for (let t = 0; t < 200; t++) { synthFrame(t, W, H, f, rng, { scene: 'translate', vx: 4.5, noise: 0 }); b.step(f); }
  assert(b.w.every(Number.isFinite));
  assert(Math.abs(b.w[20]) + Math.abs(b.w[21]) > 1e-3, 'phase weights ' + b.w[20] + ' ' + b.w[21]);
});

console.log(`\n${pass} passed`);
