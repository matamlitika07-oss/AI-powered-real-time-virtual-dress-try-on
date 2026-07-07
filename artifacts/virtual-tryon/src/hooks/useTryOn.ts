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
import { Pose, Results as PoseResults } from '@mediapipe/pose';

// ── Landmark indices ──────────────────────────────────────────────────────────
const IDX_LEFT_SHOULDER  = 11;
const IDX_RIGHT_SHOULDER = 12;
const IDX_LEFT_ELBOW     = 13;
const IDX_RIGHT_ELBOW    = 14;
const IDX_LEFT_WRIST     = 15;
const IDX_RIGHT_WRIST    = 16;
const IDX_LEFT_HIP       = 23;
const IDX_RIGHT_HIP      = 24;
const IDX_LEFT_EYE       = 2;   // Ali-Kalsekar: mp_pose.PoseLandmark.LEFT_EYE
const IDX_RIGHT_EYE      = 5;   // Ali-Kalsekar: mp_pose.PoseLandmark.RIGHT_EYE

// ── Tuning ────────────────────────────────────────────────────────────────────
const VIS_THRESHOLD     = 0.6;  // spec: hide if shoulder/hip vis < 0.6
const EMA_ALPHA         = 0.65; // EMA from Ali-Kalsekar smooth_factor≈0.7; tuned for browser
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

// ── Scale quad ────────────────────────────────────────────────────────────────
// Expand / shrink each corner of the torso quad from its centroid.
// scale > 1 = bigger garment, scale < 1 = smaller.  Ali-Kalsekar's +/- keys.
function scaleQuad(q: TorsoQuad, factor: number): TorsoQuad {
  if (factor === 1) return q;
  const cx = (q.tl.x + q.tr.x + q.bl.x + q.br.x) / 4;
  const cy = (q.tl.y + q.tr.y + q.bl.y + q.br.y) / 4;
  const s = (p: Pt): Pt => ({ x: cx + (p.x - cx) * factor, y: cy + (p.y - cy) * factor });
  return { tl: s(q.tl), tr: s(q.tr), bl: s(q.bl), br: s(q.br) };
}

// ── Glasses overlay ───────────────────────────────────────────────────────────
// Ported from Ali-Kalsekar ClothingOverlay._overlay_glasses():
//   target_w  = eye_distance * 2.2 * scale_factor
//   top_left  = center - (w/2, h*0.45)
//   rotation  = atan2(right_eye - left_eye)
// Browser version: rotate the canvas context rather than warpAffine.
function renderGlasses(
  ctx: CanvasRenderingContext2D,
  img: HTMLImageElement,
  leftEye: Pt,
  rightEye: Pt,
  scale: number,
): void {
  const iw = img.naturalWidth  || (img as HTMLImageElement).width  || 300;
  const ih = img.naturalHeight || (img as HTMLImageElement).height || 80;
  if (!iw || !ih) return;

  const eyeSpan  = Math.hypot(rightEye.x - leftEye.x, rightEye.y - leftEye.y);
  const targetW  = Math.max(40, eyeSpan * 2.2 * scale);
  const targetH  = targetW * (ih / iw);
  const cx       = (leftEye.x + rightEye.x) / 2;
  const cy       = (leftEye.y + rightEye.y) / 2;
  const angle    = Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x);

  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(angle);
  ctx.globalAlpha = 0.92;
  ctx.drawImage(img, -targetW / 2, -targetH * 0.45, targetW, targetH); // Ali-Kalsekar anchor
  ctx.globalAlpha = 1;
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
  const [selectedGarment,  setSelectedGarment]  = useState<string | null>(null);
  const [garmentCategory,  setGarmentCategory]  = useState<'torso' | 'glasses' | null>(null);
  const [opacity,          setOpacity]          = useState<number>(75);
  const [scaleFactor,      setScaleFactor]      = useState<number>(1.0);
  const [poseStatus,       setPoseStatus]       = useState<string>('Initializing…');
  const [fps,              setFps]              = useState<number>(0);
  const [webcamError,      setWebcamError]      = useState<string | null>(null);
  // Debug toggles
  const [showTorsoPoints,  setShowTorsoPoints]  = useState<boolean>(false);
  const [showTorsoPolygon, setShowTorsoPolygon] = useState<boolean>(false);
  const [showWarpBox,      setShowWarpBox]      = useState<boolean>(false);

  // ── Refs (stale-closure-safe values for onResults callback) ───────────────
  const garmentImgRef       = useRef<HTMLImageElement | null>(null);
  const cropBoundsRef       = useRef<CropBounds | null>(null);
  // EMA state — replaces rolling-average history buffer (Ali-Kalsekar approach)
  const emaLandmarksRef     = useRef<Array<{ x: number; y: number; visibility: number }>>([]);
  const lastFrameTimeRef    = useRef<number>(Date.now());
  const fpsHistoryRef       = useRef<number[]>([]);
  const opacityRef          = useRef<number>(75);
  const scaleFactorRef      = useRef<number>(1.0);
  const garmentCategoryRef  = useRef<'torso' | 'glasses' | null>(null);
  const showPointsRef       = useRef<boolean>(false);
  const showPolygonRef      = useRef<boolean>(false);
  const showWarpBoxRef      = useRef<boolean>(false);

  useEffect(() => { opacityRef.current         = opacity;          }, [opacity]);
  useEffect(() => { scaleFactorRef.current     = scaleFactor;      }, [scaleFactor]);
  useEffect(() => { garmentCategoryRef.current = garmentCategory;  }, [garmentCategory]);
  useEffect(() => { showPointsRef.current      = showTorsoPoints;  }, [showTorsoPoints]);
  useEffect(() => { showPolygonRef.current     = showTorsoPolygon; }, [showTorsoPolygon]);
  useEffect(() => { showWarpBoxRef.current     = showWarpBox;      }, [showWarpBox]);

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

    // ── 2. Landmark smoothing: EMA (exponential moving average) ──────────
    // Ported from Ali-Kalsekar PoseDetector._smooth():
    //   smoothed = α * prev + (1−α) * current
    // α=0.65 is more responsive than their 0.7 while keeping jitter-free.
    const ema = emaLandmarksRef.current;
    if (ema.length !== lms.length) {
      // First frame — seed with raw values
      emaLandmarksRef.current = lms.map(lm => ({
        x: lm.x, y: lm.y, visibility: lm.visibility ?? 0,
      }));
    } else {
      for (let i = 0; i < lms.length; i++) {
        ema[i].x          = EMA_ALPHA * ema[i].x          + (1 - EMA_ALPHA) * lms[i].x;
        ema[i].y          = EMA_ALPHA * ema[i].y          + (1 - EMA_ALPHA) * lms[i].y;
        ema[i].visibility = EMA_ALPHA * ema[i].visibility + (1 - EMA_ALPHA) * (lms[i].visibility ?? 0);
      }
    }
    const smoothed = emaLandmarksRef.current;

    const garment  = garmentImgRef.current;
    const category = garmentCategoryRef.current;

    // ── 3. Branch on category — glasses needs ONLY eye landmarks ─────────
    // Ported from Ali-Kalsekar: glasses use eye-level algorithm, completely
    // independent of torso visibility.  Return early so torso guards don't
    // block eyewear when shoulders/hips aren't fully in frame.
    if (category === 'glasses') {
      const eyeL = smoothed[IDX_LEFT_EYE];
      const eyeR = smoothed[IDX_RIGHT_EYE];
      if (garment && isVis(eyeL) && isVis(eyeR)) {
        renderGlasses(ctx, garment, lmToPx(eyeL, W, H), lmToPx(eyeR, W, H), scaleFactorRef.current);
        setPoseStatus('Tracking ✓');
      } else {
        setPoseStatus(garment ? 'Look straight at the camera' : 'Select eyewear →');
      }
      // Debug overlays skipped in glasses mode (no torso polygon to show)
      return;
    }

    // ── 4. Torso path — require shoulder + hip visibility ─────────────────
    const ls = smoothed[IDX_LEFT_SHOULDER];
    const rs = smoothed[IDX_RIGHT_SHOULDER];
    const lh = smoothed[IDX_LEFT_HIP];
    const rh = smoothed[IDX_RIGHT_HIP];
    const elL = smoothed[IDX_LEFT_ELBOW];   // renamed: avoid shadowing eye vars
    const elR = smoothed[IDX_RIGHT_ELBOW];
    const wrL = smoothed[IDX_LEFT_WRIST];
    const wrR = smoothed[IDX_RIGHT_WRIST];

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

    // Apply user scale factor (Ali-Kalsekar +/- keys → slider)
    const dst = scaleQuad({ tl, tr, bl, br }, scaleFactorRef.current);

    // ── 5. Size guard — prevent garment becoming huge when user is close ───
    const torsoH = Math.max(bl.y, br.y) - Math.min(tl.y, tr.y);
    const tooClose = torsoH > H * MAX_TORSO_H;

    // ── 6. Torso overlay (perspective warp, Phase 2 + 3) ──────────────────
    const crop = cropBoundsRef.current;
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
      gCtx.clearRect(0, 0, W, H);
      renderWarpedGarment(gCtx, garment, crop, dst);

      // 6a. Drop shadow
      ctx.save();
      ctx.globalAlpha = 0.28;
      ctx.filter      = 'blur(10px)';
      ctx.drawImage(gOC, 5, 8);
      ctx.filter      = 'none';
      ctx.restore();

      // 6b. Edge halo (GaussianBlur kernel 11 equiv — soft alpha feather)
      ctx.save();
      ctx.globalAlpha = 0.42 * (opacityRef.current / 100);
      ctx.filter      = 'blur(5px)';
      ctx.drawImage(gOC, 0, 0);
      ctx.filter      = 'none';
      ctx.restore();

      // 6c. Crisp garment at full slider opacity
      ctx.save();
      ctx.globalAlpha = opacityRef.current / 100;
      ctx.drawImage(gOC, 0, 0);
      ctx.restore();

      // ── 7. Arm masking ────────────────────────────────────────────────
      const shoulderSpan = Math.hypot(tr.x - tl.x, tr.y - tl.y);
      const armHW = shoulderSpan * ARM_HW_RATIO;
      if (isVis(elL) && isVis(wrL)) {
        paintArmMask(ctx, results.image, tl, lmToPx(elL, W, H), lmToPx(wrL, W, H), armHW, W, H);
      }
      if (isVis(elR) && isVis(wrR)) {
        paintArmMask(ctx, results.image, tr, lmToPx(elR, W, H), lmToPx(wrR, W, H), armHW, W, H);
      }
    }

    // ── Single authoritative status decision per frame ────────────────────
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
    garmentCategory,  setGarmentCategory,
    opacity, setOpacity,
    scaleFactor, setScaleFactor,
    poseStatus, fps, webcamError,
    captureLook,
    showTorsoPoints,  setShowTorsoPoints,
    showTorsoPolygon, setShowTorsoPolygon,
    showWarpBox,      setShowWarpBox,
  };
}
