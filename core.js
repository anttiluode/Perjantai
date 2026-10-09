/*
 * Perjantai core — past, present, future at 64x64.
 *
 *   past     : mem_t   = slow running average of what was seen   (episode memory)
 *              w       = 20 learned weights                       (the older past)
 *   present  : x_t     = the frame that just arrived
 *   future   : p_{t+1} = w · phi(x_t, x_t - x_{t-1}, mem_t)       (past read through the present)
 *   surprise : e_t     = camera_t - p_t                           (the innovation; the only thing that teaches)
 *
 * The predictor is one 20-weight linear filter shared by every pixel and colour
 * channel, learned online by exact least squares with forgetting. It starts as
 * copy-last-frame and earns its keep only if its error falls below
 * copy-last-frame's. The page shows that ratio live.
 *
 * The gaze (the "future moon") goes where the running squared surprise is largest:
 * the place the model is least sure about. Same rule that won in Kuulustelu —
 * ask where you expect the answer to split.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.Perjantai = api;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // 9 (3x3 of x) + 9 (3x3 of x - xPrev) + 1 (memory) + 1 (bias) + 2 (phase continuation, two scales)
  const NF = 22;
  const CENTER = 4; // index of the centre tap in a 3x3

  function clamp01(v) { return v < 0 ? 0 : v > 1 ? 1 : v; }

  // ---------- the phase moon: complex Gabor phase-velocity memory ----------
  // z = A e^{i phi} per pixel, orientation and scale. Past = slow trace of phase
  // velocity V <- (1-b)V + b z_t conj(z_{t-1}); omega = arg V. The feature handed to
  // least squares is the change phase continuation predicts: sum_k Re(z_k (e^{i omega_k} - 1)).
  class PhaseScale {
    constructor(W, H, lambda) {
      this.W = W; this.H = H; this.N = W * H;
      const sigma = 0.5 * lambda, R = Math.ceil(2.5 * sigma), k0 = 2 * Math.PI / lambda;
      this.R = R;
      const gauss = new Float32Array(2 * R + 1);
      for (let t = -R; t <= R; t++) gauss[t + R] = Math.exp(-t * t / (2 * sigma * sigma));
      this.gauss = gauss;
      this.dcLeak = Math.exp(-k0 * k0 * sigma * sigma / 2); // Morlet zero-mean correction
      this.ori = [];
      for (let o = 0; o < 4; o++) {
        const th = o * Math.PI / 4, kx = k0 * Math.cos(th), ky = k0 * Math.sin(th);
        const xr = new Float32Array(2 * R + 1), xi = new Float32Array(2 * R + 1);
        const yr = new Float32Array(2 * R + 1), yi = new Float32Array(2 * R + 1);
        for (let t = -R; t <= R; t++) {
          xr[t + R] = gauss[t + R] * Math.cos(kx * t); xi[t + R] = gauss[t + R] * Math.sin(kx * t);
          yr[t + R] = gauss[t + R] * Math.cos(ky * t); yi[t + R] = gauss[t + R] * Math.sin(ky * t);
        }
        const N = this.N;
        this.ori.push({ xr, xi, yr, yi,
          zr: new Float32Array(N), zi: new Float32Array(N),       // current response
          pr: new Float32Array(N), pi: new Float32Array(N),       // previous response
          vr: new Float32Array(N), vi: new Float32Array(N),       // phase-velocity trace
          cr: new Float32Array(N), ci: new Float32Array(N) });    // unit rotation e^{i omega}
      }
      this.tr = new Float32Array(this.N); this.ti = new Float32Array(this.N);
      this.low = new Float32Array(this.N); this.lowT = new Float32Array(this.N);
      this.started = false;
    }
    // separable real blur (for the DC correction)
    _blur(Y) {
      const W = this.W, H = this.H, R = this.R, g = this.gauss, t = this.lowT, o = this.low;
      for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
        let s = 0; for (let d = -R; d <= R; d++) { let jj = j + d; jj = jj < 0 ? -jj : jj >= W ? 2 * W - 2 - jj : jj; s += g[d + R] * Y[i * W + jj]; }
        t[i * W + j] = s;
      }
      for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
        let s = 0; for (let d = -R; d <= R; d++) { let ii = i + d; ii = ii < 0 ? -ii : ii >= H ? 2 * H - 2 - ii : ii; s += g[d + R] * t[ii * W + j]; }
        o[i * W + j] = s;
      }
    }
    update(Y, beta) {
      const W = this.W, H = this.H, R = this.R, N = this.N, tr = this.tr, ti = this.ti;
      this._blur(Y);
      const leak = this.dcLeak, low = this.low;
      for (const O of this.ori) {
        O.pr.set(O.zr); O.pi.set(O.zi);
        // horizontal complex pass on the real image
        for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
          let sr = 0, si = 0;
          for (let d = -R; d <= R; d++) {
            let jj = j + d; jj = jj < 0 ? -jj : jj >= W ? 2 * W - 2 - jj : jj;
            const y = Y[i * W + jj]; sr += O.xr[d + R] * y; si += O.xi[d + R] * y;
          }
          tr[i * W + j] = sr; ti[i * W + j] = si;
        }
        // vertical complex pass
        for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
          let sr = 0, si = 0;
          for (let d = -R; d <= R; d++) {
            let ii = i + d; ii = ii < 0 ? -ii : ii >= H ? 2 * H - 2 - ii : ii;
            const a = tr[ii * W + j], b = ti[ii * W + j], cr = O.yr[d + R], ci = O.yi[d + R];
            sr += a * cr - b * ci; si += a * ci + b * cr;
          }
          const q = i * W + j;
          O.zr[q] = sr - leak * low[q]; O.zi[q] = si;
        }
        if (!this.started) { O.pr.set(O.zr); O.pi.set(O.zi); }
        for (let q = 0; q < N; q++) {
          // z_t * conj(z_{t-1})
          const ar = O.zr[q] * O.pr[q] + O.zi[q] * O.pi[q];
          const ai = O.zi[q] * O.pr[q] - O.zr[q] * O.pi[q];
          O.vr[q] += beta * (ar - O.vr[q]); O.vi[q] += beta * (ai - O.vi[q]);
          const m = Math.hypot(O.vr[q], O.vi[q]);
          if (m > 1e-12) { O.cr[q] = O.vr[q] / m; O.ci[q] = O.vi[q] / m; } else { O.cr[q] = 1; O.ci[q] = 0; }
        }
      }
      this.started = true;
    }
    // change predicted by phase continuation, s steps beyond the present:
    // Re(z e^{i s w} (e^{i w} - 1))
    feature(out, s) {
      const N = this.N; out.fill(0);
      for (const O of this.ori) for (let q = 0; q < N; q++) {
        let rr = 1, ri = 0;
        if (s > 0) { // e^{i s w} by repeated multiplication (s is small)
          for (let k = 0; k < s; k++) { const t = rr * O.cr[q] - ri * O.ci[q]; ri = rr * O.ci[q] + ri * O.cr[q]; rr = t; }
        }
        const zr = O.zr[q] * rr - O.zi[q] * ri, zi = O.zr[q] * ri + O.zi[q] * rr;
        // Re(z (c - 1)) = zr (cr - 1) - zi ci
        out[q] += zr * (O.cr[q] - 1) - zi * O.ci[q];
      }
    }
  }

  class PhaseMoon {
    constructor(W, H) {
      this.W = W; this.H = H; this.N = W * H;
      this.s1 = new PhaseScale(W, H, 8);
      this.W2 = W >> 1; this.H2 = H >> 1;
      this.s2 = new PhaseScale(this.W2, this.H2, 8); // lambda 16 px in full-res terms
      this.Y = new Float32Array(this.N); this.Y2 = new Float32Array(this.W2 * this.H2);
      this.f2 = new Float32Array(this.W2 * this.H2);
      this.beta = 0.4;
    }
    update(img) {
      const N = this.N, Y = this.Y;
      for (let p = 0; p < N; p++) Y[p] = 0.299 * img[p * 3] + 0.587 * img[p * 3 + 1] + 0.114 * img[p * 3 + 2];
      const W = this.W, W2 = this.W2, H2 = this.H2, Y2 = this.Y2;
      for (let i = 0; i < H2; i++) for (let j = 0; j < W2; j++) {
        const a = (2 * i) * W + 2 * j;
        Y2[i * W2 + j] = 0.25 * (Y[a] + Y[a + 1] + Y[a + W] + Y[a + W + 1]);
      }
      this.s1.update(Y, this.beta); this.s2.update(Y2, this.beta);
    }
    // out: Float32Array(2N) = [scale1 feature | scale2 feature upsampled]
    features(out, s) {
      const N = this.N, W = this.W, H = this.H, W2 = this.W2, H2 = this.H2;
      const a = out.subarray(0, N), b = out.subarray(N, 2 * N);
      this.s1.feature(a, s);
      this.s2.feature(this.f2, s);
      for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
        // bilinear upsample from the half-resolution grid
        const y = Math.min(H2 - 1, Math.max(0, (i - 0.5) / 2)), x = Math.min(W2 - 1, Math.max(0, (j - 0.5) / 2));
        const i0 = Math.floor(y), j0 = Math.floor(x), i1 = Math.min(H2 - 1, i0 + 1), j1 = Math.min(W2 - 1, j0 + 1);
        const fy = y - i0, fx = x - j0, f = this.f2;
        b[i * W + j] = (1 - fy) * ((1 - fx) * f[i0 * W2 + j0] + fx * f[i0 * W2 + j1]) + fy * ((1 - fx) * f[i1 * W2 + j0] + fx * f[i1 * W2 + j1]);
      }
    }
  }

  class Brain {
    constructor(W = 64, H = 64) {
      this.W = W; this.H = H; this.N = W * H;
      const L = this.N * 3;
      this.x = new Float32Array(L);       // present (input actually fed in)
      this.xPrev = new Float32Array(L);
      this.mem = new Float32Array(L);     // past (episode)
      this.pred = new Float32Array(L);    // 1-step future, made last tick for this tick
      this.predK = new Float32Array(L);   // k-step future (rollout)
      this.cam = new Float32Array(L);     // latest camera frame
      this.camPrev = new Float32Array(L);
      this.err = new Float32Array(this.N);     // |camera - prediction|, channel mean, latest
      this.unc = new Float32Array(this.N);     // running mean of squared surprise
      this.uncBlur = new Float32Array(this.N);
      this.attn = new Float32Array(this.N);    // gaze window used for gated learning
      this._tmp = new Float32Array(this.N);
      this._ra = new Float32Array(L); this._rb = new Float32Array(L);
      this._rm = new Float32Array(L); this._rn = new Float32Array(L);
      this._phi = new Float32Array(NF);
      this.w = new Float32Array(NF);
      this.phase = new PhaseMoon(W, H);
      this.phF = new Float32Array(2 * this.N);  // phase features of the current state
      this._phR = new Float32Array(2 * this.N); // phase features during rollout
      this.gain = new Float32Array(this.N);     // precision: per-pixel trust in the eye
      this._e2 = new Float32Array(this.N);
      this._sortBuf = new Float32Array(this.N);
      this.noiseVar = 0;
      this.gaze = { u: 0.5, v: 0.5 };
      this.params = {
        memRate: 0.04,      // lambda: how fast the past moon forgets
        lr: 0.05,           // plasticity: how fast the learned weights forget (per frame)
        learn: true,
        reality: 1.0,       // 1 = camera drives, 0 = closed-loop dream
        horizon: 1,         // frames the future moon looks ahead
        uncRate: 0.12,      // how fast the uncertainty map adapts
        attentionGatesLearning: false,
        precision: false,   // present = belief p + K (camera - p) instead of the raw frame
        phase: false,       // phase-velocity memory feeds two extra features
        gainMin: 0.05,
        noisePercentile: 0.2,
      };
      this.stats = { t: 0, mseModel: 0, mseCopy: 0, skill: 0, drift: 0, updates: 0, meanGain: 1, noiseVar: 0 };
      this.resetWeights();
      this.started = false;
    }

    resetWeights() {
      this.w.fill(0);
      this.w[CENTER] = 1; // start as copy-last-frame
      this.R = new Float64Array(NF * NF);
      this.r = new Float64Array(NF);
      this.ridge = 1e-4;
    }

    // phi for pixel (i,j), channel c, from frames A (now), B (previous), memory M, phase features P
    _features(A, B, M, P, i, j, c) {
      const W = this.W, H = this.H, phi = this._phi;
      let k = 0;
      for (let di = -1; di <= 1; di++) {
        let ii = i + di; if (ii < 0) ii = 0; else if (ii >= H) ii = H - 1;
        for (let dj = -1; dj <= 1; dj++) {
          let jj = j + dj; if (jj < 0) jj = 0; else if (jj >= W) jj = W - 1;
          const q = (ii * W + jj) * 3 + c;
          phi[k] = A[q];
          phi[k + 9] = A[q] - B[q];
          k++;
        }
      }
      phi[18] = M[(i * W + j) * 3 + c];
      phi[19] = 1;
      if (P && this.params.phase) { const p = i * W + j; phi[20] = P[p]; phi[21] = P[this.N + p]; }
      else { phi[20] = 0; phi[21] = 0; }
      return phi;
    }

    _predictInto(A, B, M, P, out) {
      const W = this.W, H = this.H, w = this.w;
      for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) for (let c = 0; c < 3; c++) {
        const phi = this._features(A, B, M, P, i, j, c);
        let s = 0; for (let f = 0; f < NF; f++) s += w[f] * phi[f];
        out[(i * W + j) * 3 + c] = clamp01(s);
      }
    }

    _attentionMap() {
      const W = this.W, H = this.H, a = this.attn, g = this.gaze;
      const s2 = 2 * 0.16 * 0.16;
      for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
        const du = (j + 0.5) / W - g.u, dv = (i + 0.5) / H - g.v;
        a[i * W + j] = 0.1 + 0.9 * Math.exp(-(du * du + dv * dv) / s2);
      }
    }

    // Exact online least squares with forgetting. The learned past is kept as its
    // sufficient statistics R = <phi phi^T>, r = <phi y>, decayed by (1 - plasticity)
    // each frame, and the weights are re-solved, with a small ridge pull toward
    // copy-last-frame:  (R + d I) w = r + d w0.
    _learn(target) {
      const W = this.W, H = this.H, gated = this.params.attentionGatesLearning;
      if (gated) this._attentionMap();
      const Rf = this._Rf || (this._Rf = new Float64Array(NF * NF));
      const rf = this._rf || (this._rf = new Float64Array(NF));
      Rf.fill(0); rf.fill(0); let n = 0;
      // features of the state that produced this.pred (x, xPrev, mem are still last tick's)
      for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
        const gate = gated ? this.attn[i * W + j] : 1;
        for (let c = 0; c < 3; c++) {
          const q = (i * W + j) * 3 + c;
          const phi = this._features(this.x, this.xPrev, this.mem, this.phF, i, j, c);
          for (let a = 0; a < NF; a++) {
            const pa = gate * phi[a];
            rf[a] += pa * target[q];
            for (let b = a; b < NF; b++) Rf[a * NF + b] += pa * phi[b];
          }
          n += gate;
        }
      }
      const rho = Math.min(0.5, Math.max(0.001, this.params.lr));
      const R = this.R, r = this.r;
      for (let a = 0; a < NF; a++) {
        r[a] += rho * (rf[a] / n - r[a]);
        for (let b = a; b < NF; b++) {
          const v = R[a * NF + b] + rho * (Rf[a * NF + b] / n - R[a * NF + b]);
          R[a * NF + b] = v; R[b * NF + a] = v;
        }
      }
      if (this._solve()) this.stats.updates++;
    }

    _solve() {
      const d = this.ridge, M = this._M || (this._M = new Float64Array(NF * (NF + 1)));
      for (let a = 0; a < NF; a++) {
        for (let b = 0; b < NF; b++) M[a * (NF + 1) + b] = this.R[a * NF + b] + (a === b ? d : 0);
        M[a * (NF + 1) + NF] = this.r[a] + d * (a === CENTER ? 1 : 0);
      }
      // Gaussian elimination with partial pivoting
      for (let k = 0; k < NF; k++) {
        let piv = k, best = Math.abs(M[k * (NF + 1) + k]);
        for (let a = k + 1; a < NF; a++) { const v = Math.abs(M[a * (NF + 1) + k]); if (v > best) { best = v; piv = a; } }
        if (best < 1e-12) return false;
        if (piv !== k) for (let b = 0; b <= NF; b++) {
          const t = M[k * (NF + 1) + b]; M[k * (NF + 1) + b] = M[piv * (NF + 1) + b]; M[piv * (NF + 1) + b] = t;
        }
        for (let a = k + 1; a < NF; a++) {
          const f = M[a * (NF + 1) + k] / M[k * (NF + 1) + k];
          if (f !== 0) for (let b = k; b <= NF; b++) M[a * (NF + 1) + b] -= f * M[k * (NF + 1) + b];
        }
      }
      const w = new Float64Array(NF);
      for (let a = NF - 1; a >= 0; a--) {
        let s = M[a * (NF + 1) + NF];
        for (let b = a + 1; b < NF; b++) s -= M[a * (NF + 1) + b] * w[b];
        w[a] = s / M[a * (NF + 1) + a];
      }
      for (let a = 0; a < NF; a++) if (!Number.isFinite(w[a])) return false;
      for (let a = 0; a < NF; a++) this.w[a] = w[a];
      return true;
    }

    _gazeUpdate() {
      const W = this.W, H = this.H, u = this.unc, t = this._tmp, b = this.uncBlur, R = 4;
      // separable box blur
      for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
        let s = 0, n = 0;
        for (let d = -R; d <= R; d++) { const jj = j + d; if (jj >= 0 && jj < W) { s += u[i * W + jj]; n++; } }
        t[i * W + j] = s / n;
      }
      let mx = 0, mi = 0, mj = 0;
      for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
        let s = 0, n = 0;
        for (let d = -R; d <= R; d++) { const ii = i + d; if (ii >= 0 && ii < H) { s += t[ii * W + j]; n++; } }
        const v = s / n; b[i * W + j] = v; if (v > mx) { mx = v; mi = i; mj = j; }
      }
      if (mx <= 1e-9) return;
      // local centroid around the peak: where the next answer is least predictable.
      // (a global centroid fails on a ring of motion: it lands in the quiet middle)
      const thr = 0.6 * mx, LR = 8; let su = 0, sv = 0, sw = 0;
      for (let i = Math.max(0, mi - LR); i <= Math.min(H - 1, mi + LR); i++)
        for (let j = Math.max(0, mj - LR); j <= Math.min(W - 1, mj + LR); j++) {
          const v = b[i * W + j] - thr;
          if (v > 0) { su += v * (j + 0.5) / W; sv += v * (i + 0.5) / H; sw += v; }
        }
      if (sw > 0) {
        const k = 0.25; // smooth the saccade a little
        this.gaze.u += k * (su / sw - this.gaze.u);
        this.gaze.v += k * (sv / sw - this.gaze.v);
      }
    }

    /**
     * Feed one camera frame (Float32Array, W*H*3, values 0..1, row 0 = top).
     */
    step(frame) {
      const P = this.params, L = this.N * 3, N = this.N;
      this.camPrev.set(this.cam); this.cam.set(frame);
      if (!this.started) {
        this.x.set(frame); this.xPrev.set(frame); this.mem.set(frame);
        this.pred.set(frame); this.predK.set(frame); this.camPrev.set(frame);
        this.gain.fill(1);
        this.phase.update(this.x); this.phase.features(this.phF, 0);
        this.started = true; this.stats.t = 1;
        return this.stats;
      }
      const alpha = P.reality;
      // the input actually fed to the brain this tick
      const input = this._ra;
      for (let q = 0; q < L; q++) input[q] = alpha * frame[q] + (1 - alpha) * this.pred[q];

      // learn only from reality: a dreaming model must not be taught by its own dream
      if (P.learn && alpha >= 0.999) this._learn(input);

      // surprise is always measured against the camera
      let sm = 0, sc = 0, sd = 0;
      const ur = P.uncRate;
      for (let p = 0; p < N; p++) {
        let e = 0, e2 = 0;
        for (let c = 0; c < 3; c++) {
          const q = p * 3 + c;
          const d = frame[q] - this.pred[q];
          const dc = frame[q] - this.camPrev[q];
          e += Math.abs(d); e2 += d * d; sc += dc * dc;
          const dd = this.x[q] - frame[q]; sd += dd * dd;
        }
        this.err[p] = e / 3; sm += e2; this._e2[p] = e2 / 3;
        this.unc[p] += ur * (e2 / 3 - this.unc[p]);
      }
      const a = this.stats.t < 30 ? 0.2 : 0.03;
      this.stats.mseModel += a * (sm / L - this.stats.mseModel);
      this.stats.mseCopy += a * (sc / L - this.stats.mseCopy);
      this.stats.drift = sd / L; // how far the present the brain lives in is from the camera (last tick)
      this.stats.skill = this.stats.mseCopy > 1e-9 ? 1 - this.stats.mseModel / this.stats.mseCopy : 0;

      // the present: either the raw input, or a belief that trusts the eye only where
      // the prediction is unsure (precision), x+ = p + K (input - p)
      this.xPrev.set(this.x);
      if (P.precision) this._posteriorInto(input, this.x);
      else { this.x.set(input); this.gain.fill(1); this.stats.meanGain = 1; }
      const lam = P.memRate;
      for (let q = 0; q < L; q++) this.mem[q] += lam * (this.x[q] - this.mem[q]);

      // the phase moon reads the new present
      if (P.phase) { this.phase.update(this.x); this.phase.features(this.phF, 0); }

      // the future: one step for surprise, k steps for the moon
      this._predictInto(this.x, this.xPrev, this.mem, this.phF, this.pred);
      this._rollout(Math.max(1, Math.min(30, P.horizon | 0)));

      this._gazeUpdate();
      this.stats.t++;
      return this.stats;
    }

    // Precision. K = clamp(1 - R / max(U, E), Kmin, 1): U is the running innovation
    // variance, E the 3x3 mean of this frame's squared innovation (so a sudden change
    // breaks through at once), R the sensor-noise variance (a low percentile of U).
    _posteriorInto(input, out) {
      const W = this.W, H = this.H, N = this.N, P = this.params, U = this.unc, e2 = this._e2;
      const sb = this._sortBuf; sb.set(U); sb.sort();
      const R = sb[Math.min(N - 1, Math.floor(P.noisePercentile * N))];
      this.noiseVar = R; this.stats.noiseVar = R;
      let gs = 0;
      for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
        let a = 0, n = 0;
        for (let di = -1; di <= 1; di++) { const ii = i + di; if (ii < 0 || ii >= H) continue;
          for (let dj = -1; dj <= 1; dj++) { const jj = j + dj; if (jj < 0 || jj >= W) continue; a += e2[ii * W + jj]; n++; } }
        const p = i * W + j;
        const S = Math.max(U[p], a / n, 1e-12);
        let K = 1 - R / S; if (K < P.gainMin) K = P.gainMin; else if (K > 1) K = 1;
        this.gain[p] = K; gs += K;
        for (let c = 0; c < 3; c++) {
          const q = p * 3 + c;
          out[q] = this.pred[q] + K * (input[q] - this.pred[q]);
        }
      }
      this.stats.meanGain = gs / N;
    }

    _rollout(k) {
      if (k === 1) { this.predK.set(this.pred); return; }
      let A = this._rb, B = this._rn, M = this._rm;
      A.set(this.pred); B.set(this.x); M.set(this.mem);
      const lam = this.params.memRate, L = this.N * 3;
      for (let q = 0; q < L; q++) M[q] += lam * (A[q] - M[q]);
      const usePhase = this.params.phase;
      for (let s = 1; s < k; s++) {
        // phase continues by rotation: the s-th step ahead uses z_t e^{i s w}
        if (usePhase) this.phase.features(this._phR, s);
        this._predictInto(A, B, M, usePhase ? this._phR : null, this.predK);
        const t = B; B = A; A = t; A.set(this.predK);
        for (let q = 0; q < L; q++) M[q] += lam * (A[q] - M[q]);
      }
    }

    // mean |Laplacian| over all channels: how much fine detail an image carries
    static detail(img, W, H) {
      let s = 0, n = 0;
      for (let i = 1; i < H - 1; i++) for (let j = 1; j < W - 1; j++) for (let c = 0; c < 3; c++) {
        const q = (i * W + j) * 3 + c;
        s += Math.abs(4 * img[q] - img[q - 3] - img[q + 3] - img[q - 3 * W] - img[q + 3 * W]); n++;
      }
      return s / n;
    }

    // spatial standard deviation of an image
    static spatialStd(img) {
      let s = 0, s2 = 0; const n = img.length;
      for (let q = 0; q < n; q++) { s += img[q]; s2 += img[q] * img[q]; }
      const m = s / n; return Math.sqrt(Math.max(0, s2 / n - m * m));
    }
  }

  // ---------- a synthetic world for when there is no camera ----------
  function hash(i, j) {
    let h = (i * 374761393 + j * 668265263) | 0;
    h = (h ^ (h >>> 13)) * 1274126177 | 0;
    return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
  }
  function makeRng(seed) {
    let s = seed >>> 0 || 1;
    return function () {
      s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0;
      return s / 4294967296;
    };
  }
  function gauss(rng) {
    let u = 0, v = 0; while (u === 0) u = rng(); v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  /**
   * Synthetic room: static textured background, two drifting coloured blobs,
   * sensor noise. opts: {blobs: true, noise: 0.03, still: false}
   */
  // smooth, band-limited texture for sub-pixel translation tests
  const TEX = (() => {
    const r = makeRng(4242), waves = [];
    for (let k = 0; k < 10; k++) {
      const lam = 5 + 15 * r(), th = Math.PI * r();
      waves.push({ kx: 2 * Math.PI / lam * Math.cos(th), ky: 2 * Math.PI / lam * Math.sin(th), ph: 2 * Math.PI * r(), a: 0.04 + 0.05 * r() });
    }
    return function (x, y) { // x, y in pixels
      let s = 0; for (const w of waves) s += w.a * Math.cos(w.kx * x + w.ky * y + w.ph);
      return s;
    };
  })();

  function synthFrame(t, W, H, out, rng, opts = {}) {
    const noise = opts.noise ?? 0.03, blobs = opts.blobs ?? true;
    if (opts.scene === 'translate') {
      // a band-limited texture sliding at (vx, vy) pixels per frame
      const vx = opts.vx ?? 1.5, vy = opts.vy ?? 0;
      for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
        const v = TEX(j - vx * t, i - vy * t), q = (i * W + j) * 3;
        out[q] = clamp01(0.45 + v + noise * gauss(rng));
        out[q + 1] = clamp01(0.42 + 0.8 * v + noise * gauss(rng));
        out[q + 2] = clamp01(0.40 + 0.6 * v + noise * gauss(rng));
      }
      return {};
    }
    if (opts.scene === 'occluder') {
      // a blob moving left to right, passing behind a static grey bar
      const speed = opts.speed ?? (1.5 / W), bar0 = 0.4, bar1 = 0.65, period = 1.5;
      const bu = ((t * speed) % period) - 0.25, bv = 0.5, sg = 0.04;
      for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
        const u = (j + 0.5) / W, v = (i + 0.5) / H, q = (i * W + j) * 3;
        const tex = 0.04 * Math.sin(u * 23) * Math.sin(v * 17);
        const k = Math.exp(-((u - bu) ** 2 + (v - bv) ** 2) / (2 * sg * sg));
        let r = 0.3 + tex + 0.6 * k, g = 0.32 + tex + 0.3 * k, b = 0.38 + tex + 0.05 * k;
        if (u >= bar0 && u < bar1) { r = 0.55; g = 0.55; b = 0.55; }
        out[q] = clamp01(r + noise * gauss(rng)); out[q + 1] = clamp01(g + noise * gauss(rng)); out[q + 2] = clamp01(b + noise * gauss(rng));
      }
      return { bu, bv, bar0, bar1, sg };
    }
    const tt = opts.still ? 0 : t;
    const b1u = 0.5 + 0.28 * Math.cos(tt * 0.045), b1v = 0.5 + 0.22 * Math.sin(tt * 0.06);
    const b2u = 0.3 + 0.12 * Math.cos(tt * 0.031 + 2), b2v = 0.65 + 0.1 * Math.sin(tt * 0.027);
    for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
      const u = (j + 0.5) / W, v = (i + 0.5) / H;
      const tex = 0.08 * (hash(i >> 2, j >> 2) - 0.5) + 0.05 * Math.sin(u * 18) * Math.sin(v * 11);
      let r = 0.25 + 0.3 * u + tex, g = 0.22 + 0.25 * (1 - v) + tex, b = 0.35 + 0.15 * v + tex;
      if (opts.newBlobAt != null && t >= opts.newBlobAt) {
        // a bright patch that appears suddenly (for the precision break-through test)
        const k = Math.exp(-((u - 0.7) ** 2 + (v - 0.3) ** 2) / (2 * 0.06 * 0.06));
        r += 0.5 * k; g += 0.5 * k; b += 0.5 * k;
      }
      if (blobs) {
        const d1 = ((u - b1u) ** 2 + (v - b1v) ** 2) / (2 * 0.07 * 0.07);
        const d2 = ((u - b2u) ** 2 + (v - b2v) ** 2) / (2 * 0.05 * 0.05);
        const k1 = Math.exp(-d1), k2 = Math.exp(-d2);
        r += 0.6 * k1 + 0.1 * k2; g += 0.35 * k1 + 0.5 * k2; b += 0.05 * k1 + 0.45 * k2;
      }
      const q = (i * W + j) * 3;
      out[q] = clamp01(r + noise * gauss(rng));
      out[q + 1] = clamp01(g + noise * gauss(rng));
      out[q + 2] = clamp01(b + noise * gauss(rng));
    }
    return { b1: { u: b1u, v: b1v }, b2: { u: b2u, v: b2v } };
  }

  return { Brain, synthFrame, makeRng, NF };
});
