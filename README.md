# Perjantai

**Live:** https://anttiluode.github.io/Perjantai/

A brain that lives at 64×64 pixels and keeps three moons in a room.

| moon | what it is | in a transformer it would be |
|---|---|---|
| **past** | a slow running average of everything seen, plus the 20 learned weights | the KV cache, plus the weights |
| **present** | the frame that just arrived | the query, computed from the current state |
| **future** | the past read through the present: its guess at the next frame | what the attention output is used to predict |

The walls light up only where the future was wrong. Sit still and the room learns you and goes dark. Wave a hand and the walls light up where you moved, and the future moon drifts over to look.

Open the page, press **Open the camera** (or use the synthetic room), and watch the skill meter.

## The equations

```
past      mem_t   = mem_{t-1} + λ (x_t − mem_{t-1})
present   x_t     = camera frame
future    p_{t+1} = w · φ(x_t, x_t − x_{t−1}, mem_t)        φ: 3×3 of x, 3×3 of motion, memory, bias  (20 numbers)
surprise  e_t     = camera_t − p_t                         the innovation: the only thing that teaches
learning  (R + δI) w = r + δ w₀,   R ← R + ρ(⟨φφᵀ⟩ − R),  r ← r + ρ(⟨φ y⟩ − r)
gaze      the future moon goes to the peak of the running mean of e²
```

`w` starts as copy-last-frame (`w₀`). The weights are exact least squares with forgetting: the learned past is kept as its sufficient statistics, decayed each frame, and re-solved.

The gaze rule is the one that won in [Kuulustelu](https://github.com/anttiluode/Kuulustelu): look where you expect the answer to split. Here it is programmed, not learned.

## Round 2: precision and phase

Predictions were frozen in [PROTOCOL.md](PROTOCOL.md) and pushed before any code or measurement. Run `node tests/round2.eval.cjs`.

| # | prediction | result | verdict |
|---|---|---|---|
| P1 | duplicated frames collapse skill below 10% | 40% → 24% | **failed**: duplicates hurt, but they don't explain a collapse on their own |
| P2 | the belief is ≤ 50% of the camera's error vs the clean scene | **13%** (7.5× cleaner) | pass |
| P3 | precision costs < 2 skill points | still +0.5, moving **+3.8** | pass |
| P4 | a sudden object passes through in its first frame (< 30% error) | **8%** | pass |
| P5 | phase gains ≥ 5 points on smooth 1.5 px/frame translation | **+0.0** (linear already 98.7%) | **killed** |
| P6 | phase on the moving noisy room (recorded) | +1.7 | — |
| P7 | phase gives no object permanence behind an occluder (within 15%) | ratio 0.94 | pass |

**Exploratory, after the kill** (`tests/round2.explore.cjs`): phase earns something only when motion is fast compared with the linear filter's reach, about +3 to +4 points at 4.5–6 px/frame, with or without noise. As built, two phase features beside a linear filter, it is marginal.

So **precision is on by default**: the present moon shows a belief, `p + K·(eye − p)`. **Phase is an opt-in experiment.**

**What P7 says about the moons.** The phase-velocity memory is attached to pixel *locations*. Behind an occluder those locations hold a static bar, so nothing carries the blob's motion across. Anticipating a hand that reappears needs a memory attached to the *thing*, not to the place.

**About Antti's live +0.2%.** The page used to feed each live-camera frame about twice, because `video.currentTime` advances continuously on a live stream. That's fixed: steps now follow `requestVideoFrameCallback`, with an exact-duplicate guard, and the meter shows how many duplicates were skipped. In simulation, duplicates cost 16 points, not 40, so something else about real cameras is also at work (likely compression keeping static regions identical, or very low sensor noise). That needs a rerun on the real camera.

## What the controls do

- **Past moon memory**: how long the past moon remembers. Long memory lets a still room be predicted from the denoised past instead of the noisy present.
- **Plasticity**: how fast the 20 weights forget old evidence.
- **Camera ↔ dream**: below 100% the brain is fed its own prediction instead of the camera. Learning pauses, because a dream must not teach itself.
- **Future moon horizon**: rolls the model k frames forward from the present. The further future is blurrier.
- **Attention gates learning**: the weights learn mostly from where the future moon is looking.
- **Precision** (on): the present is a belief that trusts the eye only where the prediction is unsure, `K = 1 − R / max(U, E)`. R is the sensor-noise floor (20th percentile of running surprise), U the running surprise, and E this frame's local surprise, so anything new breaks through immediately.
- **Phase moon** (off): a complex Gabor bank (2 scales, 4 orientations) remembers phase velocity and offers the predicted phase continuation to least squares. The rollout continues it by phase rotation.

## Honest ledger

Measured, in `node tests/core.test.cjs` (12 tests, all pass):

- **Still noisy room:** beats copy-last-frame by about 47%. The ideal for i.i.d. sensor noise is about 50%, since copy-last carries 2σ² and a perfectly denoised predictor carries σ². The memory tap carries the scene.
- **Smooth noiseless motion:** beats copy-last-frame by about 91%. The motion taps learn to continue the last change.
- **Moving noisy room:** about 40%.
- **Gaze:** tracks the moving blob at a mean distance of 0.11, against 0.42 for a random point.
- **Dream (closed loop):** I expected it to melt to grey. **It doesn't.** The learned filter keeps unit gain at DC (spatial taps plus memory sum to about 1.00), so the large-scale layout persists and slowly inflates while fine detail dissolves almost completely (mean |Laplacian| falls from 0.11 to 0.0005). A *still* room dreams almost true, because the memory tap anchors it. Back at 100% camera, the drift collapses within 60 frames.
- **Rollout:** fine detail at +20 frames is about 7× lower than at +1.

What this is not:

- It is not a deep video model. It is the smallest predictor that has to *beat copy-last-frame*, and the skill meter shows live whether it does. Real cameras vary: lighting flicker, autoexposure and compression all show up as surprise.
- The walls show surprise above the frame's own noise floor: a 3×3 average minus the frame mean. That is a display choice. The learning uses the raw error.
- The gaze does not decide what is learned unless you tick *attention gates learning*. Even then the rule is fixed, not learned.

## Where a bigger model plugs in

The future is one function: `_predictInto(present, previous, memory, out)` in `core.js`. A larger low-res video model (ONNX in the browser, or a GPU server) can replace it, and everything else (surprise walls, dream mode, horizon rollout, gaze) works unchanged. The skill meter stays the judge.

## Files

- `index.html`: the room (WebGL2, analytic ray tracing: three refracting moons, four bounces), controls and meters
- `core.js`: the brain. No dependencies; runs in the browser and in node.
- `tests/core.test.cjs`: the ledger above, as asserts.

Built with Claude (Opus 5.5), Friday 9 October 2026.
