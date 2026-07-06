---
name: MediaPipe dual-model OOM in Replit preview
description: Running @mediapipe/pose + @mediapipe/selfie_segmentation together crashes the Replit iframe sandbox with WebAssembly OOM; use Pose-only + landmark polygons for person-over-garment compositing.
---

## Rule
Never load two large MediaPipe WASM models simultaneously in a Replit preview iframe. The sandbox has insufficient WebAssembly memory for concurrent large models.

**Why:** The preview iframe is sandboxed with restricted memory. Both Pose (~50MB WASM) and SelfieSegmentation (~35MB WASM) simultaneously fail with `WebAssembly.instantiate(): Out of memory: Cannot allocate Wasm memory for new instance`. This also cascades into `gl_context.cc:867 Check failed: delegate_graph_context` abort floods.

**How to apply:** For "person in front of garment" compositing, use Pose-only + arm polygon clipping:
- Define a hexagonal arm polygon from shoulder→elbow→wrist landmarks
- Use `ctx.clip()` with that polygon
- Draw the raw video frame back in the clipped region
- This achieves identical visual effect (arm pixels over garment) without a second model
- Works in any memory-constrained environment; is also more precise than segmentation for arms

**OffscreenCanvas fallback:** `computeCropBounds()` must guard with `typeof OffscreenCanvas !== 'undefined'` and fall back to `document.createElement('canvas')` for Safari/iOS compatibility.
