/**
 * useTryOn — Virtual Try-On Hook  (Phase 2 + Phase 3)
 *
 * Phase 2 — Perspective-correct garment warp onto torso polygon.
 *   cv2.getPerspectiveTransform + warpPerspective equivalent:
 *   NxN sub-quad affine tiling (each sub-quad rendered as 2 affine triangles).
 *   Point ordering (matches getPerspectiveTransform convention):
 *     src: TL, TR, BL, BR  of the cropped garment image
 *     dst: TL=left-shoulder, TR=right-shoulder, BL=left-hip, BR=right-hip
 *
 * Phase 3 — Realistic overlay.
 *   GaussianBlur(kernel 11) → ctx.filter='blur(5px)' on OffscreenCanvas.
 *   Soft alpha blending, shadow, landmark smoothing, arm masking, size guard.
 *
 * Coordinate system: video drawn straight (no mirror transform).
 *   MediaPipe landmark coords (normalised [0,1]) map directly as:
 *     pixel = { x: lm.x * W, y: lm.y * H }
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import { Pose, Results as PoseResults, NormalizedLandmarkList } from '@mediapipe/pose';

// ── Landmark indices ──────────────────────────────────────────────────────────
const IDX_LEFT_SHOULDER  = 11;
const IDX_RIGHT_SHOULDER = 12;
const IDX_LEFT_ELBOW     = 13;
const IDX_RIGHT_ELBOW    = 14;
const IDX_LEFT_WRIST     = 15;
const IDX_RIGHT_WRIST    = 16;
const IDX_LEFT_HIP       = 23;
const IDX_RIGHT_HIP      = 24;

// ── Tuning ────────────────────────────────────────────────────────────────────
const VIS_THRESHOLD     = 0.6;  // spec: hide if shoulder/hip vis < 0.6
const HISTORY_SIZE      = 5;    // spec: smooth over last 5 frames
const WARP_N            = 10;   // NxN sub-quad divisions (10 = visually perfect for torso)
const ARM_HW_RATIO      = 0.12; // arm polygon half-width as fraction of shoulder span
const MAX_TORSO_H       = 0.88; // hide garment if torso height > this fraction of canvas H

// ── Types ─────────────────────────────────────────────────────────────────────
interface Pt { x: number; y: number }
interface CropBounds { sx: number; sy: number; sw: number; sh: number }
interface TorsoQuad { tl: Pt; tr: Pt; bl: Pt; br: Pt }

// ── Utilities ─────────────────────────────────────────────────────────────────
const lmToPx = (lm: { x: number; y: number }, W: number, H: number): Pt =>
  ({ x: lm.x * W, y: lm.y * H });

const isVis = (lm: { visibility?: number }): boolean =>
  (lm.visibility ?? 0) >= VIS_THRESHOLD;

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

// ── PIL getbbox() equivalent ──────────────────────────────────────────────────
// Scans all pixels and returns the tight bounding box of non-transparent pixels.
// Safari fallback: uses a regular <canvas> element when OffscreenCanvas is absent.
function computeCropBounds(img: HTMLImageElement): CropBounds {
  const iw = img.naturalWidth, ih = img.naturalHeight;
  if (!iw || !ih) return { sx: 0, sy: 0, sw: iw, sh: ih };

  let ctx2: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null;
  if (typeof OffscreenCanvas !== 'undefined') {
    const oc = new OffscreenCanvas(iw, ih);
    ctx2 = oc.getContext('2d');
  } else {
    const el = document.createElement('canvas');
    el.width = iw; el.height = ih;
    ctx2 = el.getContext('2d');
  }
  if (!ctx2) return { sx: 0, sy: 0, sw: iw, sh: ih };

  ctx2.drawImage(img, 0, 0);
  const data = ctx2.getImageData(0, 0, iw, ih).data;
  let x0 = iw, x1 = 0, y0 = ih, y1 = 0;
  for (let y = 0; y < ih; y++) {
    for (let x = 0; x < iw; x++) {
      if (data[(y * iw + x) * 4 + 3] > 8) {
        if (x < x0) x0 = x; if (x > x1) x1 = x;
        if (y < y0) y0 = y; if (y > y1) y1 = y;
      }
    }
  }
  if (x1 < x0) return { sx: 0, sy: 0, sw: iw, sh: ih };
  return { sx: x0, sy: y0, sw: x1 - x0 + 1, sh: y1 - y0 + 1 };
}

// ── Affine solve ──────────────────────────────────────────────────────────────
// Computes the 2D affine transform M that maps source triangle (s0,s1,s2)
// to destination triangle (d0,d1,d2).
//
// Canvas ctx.transform(a,b,c,d,e,f) formula:
//   screen_x = a * img_x + c * img_y + e
//   screen_y = b * img_x + d * img_y + f
//
// We solve for [a,c,e] and [b,d,f] using 3 point correspondences.
// This is the browser equivalent of cv2.getAffineTransform().
function solveAffine(
  s0: Pt, s1: Pt, s2: Pt,
  d0: Pt, d1: Pt, d2: Pt,
): [number, number, number, number, number, number] {
  const denom =
    (s1.x - s0.x) * (s2.y - s0.y) -
    (s2.x - s0.x) * (s1.y - s0.y);
  if (Math.abs(denom) < 1e-9) return [1, 0, 0, 1, 0, 0];
  const inv = 1 / denom;

  // x-row: a, c, e
  const a = inv * ((d1.x - d0.x) * (s2.y - s0.y) - (d2.x - d0.x) * (s1.y - s0.y));
  const c = inv * ((d2.x - d0.x) * (s1.x - s0.x) - (d1.x - d0.x) * (s2.x - s0.x));
  const e = d0.x - a * s0.x - c * s0.y;

  // y-row: b, d, f
  const b = inv * ((d1.y - d0.y) * (s2.y - s0.y) - (d2.y - d0.y) * (s1.y - s0.y));
  const d = inv * ((d2.y - d0.y) * (s1.x - s0.x) - (d1.y - d0.y) * (s2.x - s0.x));
  const f = d0.y - b * s0.x - d * s0.y;

  return [a, b, c, d, e, f];
}

// Clip canvas to dest triangle, apply affine transform, draw image.
type AnyCtx = CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
function drawAffineTriangle(
  ctx: AnyCtx,
  img: HTMLImageElement,
  s0: Pt, s1: Pt, s2: Pt,   // source corners (image pixel coords)
  d0: Pt, d1: Pt, d2: Pt,   // dest  corners (canvas pixel coords)
): void {
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(d0.x, d0.y); ctx.lineTo(d1.x, d1.y); ctx.lineTo(d2.x, d2.y);
  ctx.closePath();
  ctx.clip();
  const [a, b, cc, dd, e, f] = solveAffine(s0, s1, s2, d0, d1, d2);
  ctx.transform(a, b, cc, dd, e, f);
  ctx.drawImage(img, 0, 0);
  ctx.restore();
}

// ── Perspective warp ──────────────────────────────────────────────────────────
// Renders the garment with perspective-correct warping by subdividing the
// destination quad into WARP_N × WARP_N sub-quads, each rendered as two
// affine-warped triangles.  This approximates cv2.warpPerspective() —
// at N=10 the visual error is sub-pixel for any near-rectangular torso quad.
//
// Source ordering  (top-left, top-right, bottom-left, bottom-right):
//   TL = (sx,      sy     )   TR = (sx+sw, sy     )
//   BL = (sx,      sy+sh  )   BR = (sx+sw, sy+sh  )
//
// Destination ordering:
//   TL = left-shoulder   TR = right-shoulder
//   BL = left-hip        BR = right-hip
function renderWarpedGarment(
  ctx: AnyCtx,
  img: HTMLImageElement,
  crop: CropBounds,
  dst: TorsoQuad,
): void {
  const { sx, sy, sw, sh } = crop;
  const N = WARP_N;

  // Bilinear interpolation on the destination quad
  const bilerp = (s: number, t: number): Pt => ({
    x: lerp(lerp(dst.tl.x, dst.tr.x, s), lerp(dst.bl.x, dst.br.x, s), t),
    y: lerp(lerp(dst.tl.y, dst.tr.y, s), lerp(dst.bl.y, dst.br.y, s), t),
  });

  for (let row = 0; row < N; row++) {
    for (let col = 0; col < N; col++) {
      const t0 = row / N,       t1 = (row + 1) / N;
      const s0 = col / N,       s1 = (col + 1) / N;

      // Destination sub-quad corners
      const dtl = bilerp(s0, t0), dtr = bilerp(s1, t0);
      const dbl = bilerp(s0, t1), dbr = bilerp(s1, t1);

      // Source sub-quad corners in image pixel space
      const stl: Pt = { x: sx + sw * s0, y: sy + sh * t0 };
      const str: Pt = { x: sx + sw * s1, y: sy + sh * t0 };
      const sbl: Pt = { x: sx + sw * s0, y: sy + sh * t1 };
      const sbr: Pt = { x: sx + sw * s1, y: sy + sh * t1 };

      // Two triangles per sub-quad (CCW winding)
      drawAffineTriangle(ctx, img, stl, str, sbr, dtl, dtr, dbr);
      drawAffineTriangle(ctx, img, stl, sbr, sbl, dtl, dbr, dbl);
    }
  }
}

// ── Arm masking ───────────────────────────────────────────────────────────────
// Clips a hexagonal polygon along shoulder→elbow→wrist and redraws raw video
// pixels there, so the arm appears in front of the garment.
// Canvas equivalent of cv2.fillPoly arm mask + bitwise_and composite.
function paintArmMask(
  ctx: CanvasRenderingContext2D,
  videoSrc: unknown,
  shoulder: Pt, elbow: Pt, wrist: Pt,
  halfW: number, W: number, H: number,
): void {
  const a1 = Math.atan2(elbow.y - shoulder.y, elbow.x - shoulder.x);
  const a2 = Math.atan2(wrist.y  - elbow.y,   wrist.x  - elbow.x);
  const dx1 = Math.cos(a1 + Math.PI / 2) * halfW, dy1 = Math.sin(a1 + Math.PI / 2) * halfW;
  const dx2 = Math.cos(a2 + Math.PI / 2) * halfW, dy2 = Math.sin(a2 + Math.PI / 2) * halfW;

  ctx.save();
  ctx.beginPath();
  ctx.moveTo(shoulder.x + dx1, shoulder.y + dy1);
  ctx.lineTo(elbow.x   + dx1, elbow.y   + dy1);
  ctx.lineTo(wrist.x   + dx2, wrist.y   + dy2);
  ctx.lineTo(wrist.x   - dx2, wrist.y   - dy2);
  ctx.lineTo(elbow.x   - dx1, elbow.y   - dy1);
  ctx.lineTo(shoulder.x - dx1, shoulder.y - dy1);
  ctx.closePath();
  ctx.clip();
  ctx.drawImage(videoSrc as unknown as CanvasImageSource, 0, 0, W, H);
  ctx.restore();
}

// ── Hook ──────────────────────────────────────────────────────────────────────
export function useTryOn() {
  const videoRef  = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const poseRef   = useRef<Pose | null>(null);
  const animRef   = useRef<number | null>(null);
  // Pre-allocated OffscreenCanvas — garment is warped here once per frame,
  // then composited to the main canvas with blur / opacity passes.
  const garmentOCRef = useRef<OffscreenCanvas | null>(null);

  // ── State ─────────────────────────────────────────────────────────────────
  const [selectedGarment, setSelectedGarment]   = useState<string | null>(null);
  const [opacity,         setOpacity]           = useState<number>(75);
  const [poseStatus,      setPoseStatus]        = useState<string>('Initializing…');
  const [fps,             setFps]               = useState<number>(0);
  const [webcamError,     setWebcamError]       = useState<string | null>(null);
  // Debug toggles
  const [showTorsoPoints,  setShowTorsoPoints]  = useState<boolean>(false);
  const [showTorsoPolygon, setShowTorsoPolygon] = useState<boolean>(false);
  const [showWarpBox,      setShowWarpBox]      = useState<boolean>(false);

  // ── Refs (stale-closure-safe values for onResults callback) ───────────────
  const garmentImgRef      = useRef<HTMLImageElement | null>(null);
  const cropBoundsRef      = useRef<CropBounds | null>(null);
  const landmarkHistoryRef = useRef<NormalizedLandmarkList[]>([]);
  const lastFrameTimeRef   = useRef<number>(Date.now());
  const fpsHistoryRef      = useRef<number[]>([]);
  const opacityRef         = useRef<number>(75);
  const showPointsRef      = useRef<boolean>(false);
  const showPolygonRef     = useRef<boolean>(false);
  const showWarpBoxRef     = useRef<boolean>(false);

  useEffect(() => { opacityRef.current     = opacity;          }, [opacity]);
  useEffect(() => { showPointsRef.current  = showTorsoPoints;  }, [showTorsoPoints]);
  useEffect(() => { showPolygonRef.current = showTorsoPolygon; }, [showTorsoPolygon]);
  useEffect(() => { showWarpBoxRef.current = showWarpBox;      }, [showWarpBox]);

  // ── Garment load: decode PNG + compute crop bounds once ───────────────────
  useEffect(() => {
    if (!selectedGarment) {
      garmentImgRef.current = null;
      cropBoundsRef.current = null;
      return;
    }
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = selectedGarment;
    img.onload = () => {
      garmentImgRef.current = img;
      cropBoundsRef.current = computeCropBounds(img);
    };
  }, [selectedGarment]);

  // ── Core render callback ──────────────────────────────────────────────────
  const onResults = useCallback((results: PoseResults) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const W = canvas.width, H = canvas.height;

    // FPS — rolling 12-frame average
    const now = Date.now();
    fpsHistoryRef.current.push(1000 / Math.max(1, now - lastFrameTimeRef.current));
    lastFrameTimeRef.current = now;
    if (fpsHistoryRef.current.length > 12) fpsHistoryRef.current.shift();
    setFps(Math.round(fpsHistoryRef.current.reduce((a, b) => a + b, 0) / fpsHistoryRef.current.length));

    ctx.clearRect(0, 0, W, H);

    // ── 1. Background: raw video — NO mirror transform ─────────────────────
    // Landmark coords are in the same coordinate space as the raw frame.
    // Applying any additional flip would misalign them.
    ctx.drawImage(results.image as unknown as CanvasImageSource, 0, 0, W, H);

    const lms = results.poseLandmarks;
    if (!lms) {
      setPoseStatus('No pose — step back so full body is visible');
      return;
    }

    // ── 2. Landmark smoothing: 5-frame rolling average ────────────────────
    landmarkHistoryRef.current.push(lms);
    if (landmarkHistoryRef.current.length > HISTORY_SIZE) landmarkHistoryRef.current.shift();
    const smoothed = lms.map((_, i) => {
      let x = 0, y = 0, v = 0;
      for (const f of landmarkHistoryRef.current) { x += f[i].x; y += f[i].y; v += f[i].visibility ?? 0; }
      const n = landmarkHistoryRef.current.length;
      return { x: x / n, y: y / n, visibility: v / n };
    });

    const ls = smoothed[IDX_LEFT_SHOULDER];
    const rs = smoothed[IDX_RIGHT_SHOULDER];
    const lh = smoothed[IDX_LEFT_HIP];
    const rh = smoothed[IDX_RIGHT_HIP];
    const le = smoothed[IDX_LEFT_ELBOW];
    const re = smoothed[IDX_RIGHT_ELBOW];
    const lw = smoothed[IDX_LEFT_WRIST];
    const rw = smoothed[IDX_RIGHT_WRIST];

    // ── 3. Visibility check — spec: threshold = 0.6 ───────────────────────
    if (!isVis(ls) || !isVis(rs) || !isVis(lh) || !isVis(rh)) {
      setPoseStatus('Move back for full body view');
      return;
    }

    // Convert smoothed landmarks → pixel coords.
    // ORDER matches getPerspectiveTransform convention:
    //   TL = top-left,  TR = top-right,  BL = bottom-left,  BR = bottom-right
    const tl = lmToPx(ls, W, H); // TL = left shoulder
    const tr = lmToPx(rs, W, H); // TR = right shoulder
    const bl = lmToPx(lh, W, H); // BL = left hip
    const br = lmToPx(rh, W, H); // BR = right hip

    const dst: TorsoQuad = { tl, tr, bl, br };

    // ── 4. Size guard — prevent garment becoming huge when user is close ───
    const torsoH = Math.max(bl.y, br.y) - Math.min(tl.y, tr.y);
    const tooClose = torsoH > H * MAX_TORSO_H;

    // ── 5. Garment rendering (Phase 2 + 3) ────────────────────────────────
    const garment = garmentImgRef.current;
    const crop    = cropBoundsRef.current;

    if (garment && crop && !tooClose) {
      // Ensure OffscreenCanvas is allocated and sized correctly
      if (!garmentOCRef.current ||
          garmentOCRef.current.width  !== W ||
          garmentOCRef.current.height !== H) {
        garmentOCRef.current = new OffscreenCanvas(W, H);
      }
      const gOC  = garmentOCRef.current;
      const gCtx = gOC.getContext('2d')!;

      // Render perspective-warped garment into the OffscreenCanvas.
      // Source:  TL=(sx,sy) TR=(sx+sw,sy) BL=(sx,sy+sh) BR=(sx+sw,sy+sh)
      // Dest:    TL=leftShoulder  TR=rightShoulder  BL=leftHip  BR=rightHip
      // Garment is drawn upright — top maps to shoulders, bottom maps to hips.
      gCtx.clearRect(0, 0, W, H);
      renderWarpedGarment(gCtx, garment, crop, dst);

      // 5a. Drop shadow: blurred + offset dark version of the warped garment.
      //     Drawn first so it appears behind the garment.
      ctx.save();
      ctx.globalAlpha = 0.28;
      ctx.filter      = 'blur(10px)';
      ctx.drawImage(gOC, 5, 8);   // slight down-right offset
      ctx.filter      = 'none';
      ctx.restore();

      // 5b. Edge halo: blurred soft-border pass (GaussianBlur kernel 11 equiv).
      //     Approximates blurring the alpha channel for feathered edges.
      ctx.save();
      ctx.globalAlpha = 0.42 * (opacityRef.current / 100);
      ctx.filter      = 'blur(5px)';
      ctx.drawImage(gOC, 0, 0);
      ctx.filter      = 'none';
      ctx.restore();

      // 5c. Crisp garment at full slider opacity.
      ctx.save();
      ctx.globalAlpha = opacityRef.current / 100;
      ctx.drawImage(gOC, 0, 0);
      ctx.restore();

      // ── 6. Arm masking: draw video pixels over garment in arm regions ────
      // Gives the appearance that the person's arms are in front of the garment.
      // Equivalent to cv2.fillPoly arm mask + addWeighted composite.
      const shoulderSpan = Math.hypot(tr.x - tl.x, tr.y - tl.y);
      const armHW = shoulderSpan * ARM_HW_RATIO;

      if (isVis(le) && isVis(lw)) {
        paintArmMask(ctx, results.image, tl, lmToPx(le, W, H), lmToPx(lw, W, H), armHW, W, H);
      }
      if (isVis(re) && isVis(rw)) {
        paintArmMask(ctx, results.image, tr, lmToPx(re, W, H), lmToPx(rw, W, H), armHW, W, H);
      }
    }

    // Status
    if (tooClose) {
      setPoseStatus('Step back — too close to camera');
    } else {
      setPoseStatus(garment ? 'Tracking ✓' : 'Select a garment →');
    }

    // ── 7. Debug overlays (off by default, toggled via UI) ────────────────
    if (showPolygonRef.current) {
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(tl.x, tl.y); ctx.lineTo(tr.x, tr.y);
      ctx.lineTo(br.x, br.y); ctx.lineTo(bl.x, bl.y);
      ctx.closePath();
      ctx.strokeStyle = 'rgba(0,255,128,0.9)';
      ctx.lineWidth   = 2;
      ctx.stroke();
      ctx.fillStyle   = 'rgba(0,255,128,0.07)';
      ctx.fill();
      ctx.restore();
    }

    if (showWarpBoxRef.current) {
      // Expanded destination quad (garment bounding box) — gold dashed
      const cx = (tl.x + tr.x + bl.x + br.x) / 4;
      const cy = (tl.y + tr.y + bl.y + br.y) / 4;
      const EX = 1.14;
      const exp = (p: Pt): Pt => ({ x: cx + (p.x - cx) * EX, y: cy + (p.y - cy) * EX });
      const etl = exp(tl), etr = exp(tr), ebl = exp(bl), ebr = exp(br);
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(etl.x, etl.y); ctx.lineTo(etr.x, etr.y);
      ctx.lineTo(ebr.x, ebr.y); ctx.lineTo(ebl.x, ebl.y);
      ctx.closePath();
      ctx.strokeStyle = 'rgba(255,215,0,0.85)';
      ctx.lineWidth   = 2;
      ctx.setLineDash([6, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();
    }

    if (showPointsRef.current) {
      const corners = [
        { pt: tl, label: 'TL · L.Shoulder', color: '#00ff80' },
        { pt: tr, label: 'TR · R.Shoulder', color: '#00ff80' },
        { pt: bl, label: 'BL · L.Hip',      color: '#00aaff' },
        { pt: br, label: 'BR · R.Hip',      color: '#00aaff' },
      ];
      for (const { pt, label, color } of corners) {
        ctx.save();
        ctx.beginPath(); ctx.arc(pt.x, pt.y, 6, 0, Math.PI * 2);
        ctx.fillStyle = color; ctx.strokeStyle = '#000'; ctx.lineWidth = 2;
        ctx.fill(); ctx.stroke();
        ctx.font = 'bold 10px monospace'; ctx.textAlign = 'center';
        ctx.strokeStyle = '#000'; ctx.lineWidth = 3;
        ctx.strokeText(label, pt.x, pt.y - 11);
        ctx.fillStyle = '#fff'; ctx.fillText(label, pt.x, pt.y - 11);
        ctx.restore();
      }
    }
  }, []);

  // ── Camera init ───────────────────────────────────────────────────────────
  useEffect(() => {
    let active = true;
    navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
    }).then((stream) => {
      if (!active || !videoRef.current) return;
      videoRef.current.srcObject = stream;
      videoRef.current.play().catch(() => {});
    }).catch(() => {
      if (!active) return;
      setWebcamError('Camera access required — allow permissions and reload');
      setPoseStatus('Camera error');
    });
    return () => {
      active = false;
      if (videoRef.current?.srcObject) {
        (videoRef.current.srcObject as MediaStream).getTracks().forEach(t => t.stop());
      }
    };
  }, []);

  // ── MediaPipe Pose init ───────────────────────────────────────────────────
  useEffect(() => {
    const pose = new Pose({
      locateFile: (f) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5.1675469404/${f}`,
    });
    pose.setOptions({
      modelComplexity:        1,
      smoothLandmarks:        true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence:  0.5,
    });
    pose.onResults(onResults);
    poseRef.current = pose;
    return () => { poseRef.current = null; pose.close().catch(() => {}); };
  }, [onResults]);

  // ── Animation loop ────────────────────────────────────────────────────────
  useEffect(() => {
    async function loop() {
      const video = videoRef.current, canvas = canvasRef.current;
      if (video && video.readyState >= 2 && poseRef.current) {
        if (canvas && video.videoWidth > 0 && canvas.width !== video.videoWidth) {
          canvas.width  = video.videoWidth;
          canvas.height = video.videoHeight;
        }
        try { await poseRef.current.send({ image: video }); } catch { /* transient */ }
      }
      animRef.current = requestAnimationFrame(loop);
    }
    animRef.current = requestAnimationFrame(loop);
    return () => { if (animRef.current) cancelAnimationFrame(animRef.current); };
  }, []);

  // ── Capture ───────────────────────────────────────────────────────────────
  const captureLook = useCallback(() => {
    if (!selectedGarment) return false;
    canvasRef.current?.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = 'virtual-tryon.png'; a.click();
      URL.revokeObjectURL(url);
    }, 'image/png');
    return true;
  }, [selectedGarment]);

  return {
    videoRef, canvasRef,
    selectedGarment, setSelectedGarment,
    opacity, setOpacity,
    poseStatus, fps, webcamError,
    captureLook,
    showTorsoPoints,  setShowTorsoPoints,
    showTorsoPolygon, setShowTorsoPolygon,
    showWarpBox,      setShowWarpBox,
  };
}
