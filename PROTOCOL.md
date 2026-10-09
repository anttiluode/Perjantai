# Perjantai — frozen predictions for round 2 (precision + phase)

Written and pushed **before** any of the code below exists or any number is measured.
Friday 9 October 2026, 16:30 Helsinki.

## What triggered this

Antti's live screenshot: skill +0.2%, model error = copy-last error, weights collapsed
to pure copy-last, step rate 60/s. Diagnosis (not yet verified): for a live
MediaStream `video.currentTime` advances continuously, so the "never step a
repeated frame" guard fails and each camera frame is fed about twice. On a duplicate,
copy-last is exact, so least squares learns copy-last.

## Changes to be built

1. **Frame dedup**: `requestVideoFrameCallback` on an in-DOM video, plus an
   exact-duplicate guard (identical 64×64 capture is skipped).
2. **Precision (posterior present)**: the present becomes a belief,
   `x⁺ = p + K ⊙ (c − p)`, per pixel `K = clamp(1 − R / max(U, E), K_min, 1)`, with
   `U` the running innovation variance, `E` the 3×3 local mean of the current squared
   innovation (so a sudden hand breaks through immediately), and `R` the sensor-noise
   variance estimated as a low percentile of `U`. The posterior feeds memory and the
   next prediction. Learning still targets the camera.
3. **Phase moon**: complex Gabor responses of luminance at 2 scales (λ = 8 px at
   64×64, and λ = 8 px at 32×32), 4 orientations. Past = a slow trace of phase
   velocity `V ← (1−β)V + β z_t z̄_{t−1}`, `ω = arg V`. Two new least-squares
   features, one per scale: the change phase continuation predicts,
   `Σ_k Re(z_k e^{iω_k}) − Re(z_k)`. Least squares decides whether to use them.
   The rollout continues them by rotating phase, `z_{t+s} = z_t e^{isω}`.

## Predictions (synthetic, measured after 400 frames unless stated, seed fixed)

| # | prediction | kill / fail condition |
|---|---|---|
| P1 | With duplicated frames (each frame fed twice), the learned model collapses toward copy-last: skill falls below 10% on the moving noisy room where it is ~40% without duplicates. This confirms the diagnosis. | skill with duplicates ≥ 25% → diagnosis wrong |
| P2 | Precision denoises the present: on the still noisy room (σ = 0.03), MSE of the posterior present vs the clean scene is ≤ 50% of the raw camera's. | > 75% → precision fails |
| P3 | Precision does not hurt prediction: 1-step skill with precision ≥ skill without, minus 2 points, on both still and moving noisy rooms. | drop > 2 points on either → fails |
| P4 | Precision passes a sudden change: when a new bright blob appears in a still room, the posterior present's error on the blob region in its first frame is < 30% of the blob's contrast. | ≥ 50% → K lags (Kalman lag not broken) |
| P5 | Phase helps on smooth translation: a noiseless texture translating 1.5 px/frame. Phase features get nonzero weight and skill rises ≥ 5 points over the linear model. | gain < 2 points → the phase moon did not earn its place |
| P6 | On the default moving noisy room, phase gains < 5 points (noise corrupts phase on soft blobs). Not a kill; recorded either way. | — |
| P7 | Phase does NOT give object permanence: a blob moving behind a static occluder bar. Error in the 5 frames after it re-emerges is within 15% of the linear model's. The phase-velocity memory is attached to pixel locations, and behind the bar those locations hold a static bar, so nothing carries the blob's motion across. | phase error < 85% of linear at re-emergence → my reasoning was wrong (interesting) |

Everything measured goes into the README ledger, including failures.
