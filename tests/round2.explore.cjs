// EXPLORATORY (run after P5 was killed; not pre-registered): where, if anywhere, does phase win?
const { Brain, synthFrame, makeRng } = require('../core.js');
const W = 64, H = 64, L = W * H * 3;
function skill(opts, phase, T = 300) {
  const b = new Brain(W, H); b.params.phase = phase; const r = makeRng(11), f = new Float32Array(L);
  for (let t = 0; t < T; t++) { synthFrame(t, W, H, f, r, opts); b.step(f); }
  return b.stats.skill;
}
for (const noise of [0, 0.01])
  for (const vx of [1.5, 3, 4.5, 6]) {
    const o = { scene: 'translate', vx, noise };
    const a = skill(o, false), b = skill(o, true);
    console.log(`noise ${noise}  vx ${vx} px/fr   linear ${(100*a).toFixed(1)}%   phase ${(100*b).toFixed(1)}%   gain ${(100*(b-a)).toFixed(1)} pts`);
  }
