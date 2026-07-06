import { useEffect, useRef, useState, useCallback } from 'react';
import { Pose, Results as PoseResults, NormalizedLandmarkList } from '@mediapipe/pose';

// ─── Landmark indices ────────────────────────────────────────────────────────
const NOSE           = 0;
const LEFT_SHOULDER  = 11;
const RIGHT_SHOULDER = 12;
const LEFT_ELBOW     = 13;
const RIGHT_ELBOW    = 14;
const LEFT_WRIST     = 15;
const RIGHT_WRIST    = 16;
const LEFT_HIP       = 23;
const RIGHT_HIP      = 24;

// ─── Tuning constants ────────────────────────────────────────────────────────
const HISTORY_SIZE    = 5;
const VIS_MIN         = 0.5;
const MAX_GARMENT_W   = 0.75;
const MAX_GARMENT_H   = 0.87;
const GARMENT_W_SCALE = 1.35;
const EDGE_BLUR_PX    = 6;
const ARM_WIDTH_RATIO = 0.13; // arm half-width as fraction of shoulder width

// ─── Helpers ─────────────────────────────────────────────────────────────────
interface Pt { x: number; y: number }
interface CropBounds { sx: number; sy: number; sw: number; sh: number }

const mid = (a: Pt, b: Pt): Pt => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
const dist = (a: Pt, b: Pt): number => Math.hypot(b.x - a.x, b.y - a.y);
const toPx = (lm: { x: number; y: number }, w: number, h: number): Pt => ({ x: lm.x * w, y: lm.y * h });
const visible = (lm: { visibility?: number }) => (lm.visibility ?? 0) >= VIS_MIN;

/**
 * PIL getbbox() equivalent — bounding box of non-transparent pixels.
 * Runs once on garment load; eliminates blank padding that causes mis-alignment.
 */
function computeCropBounds(img: HTMLImageElement): CropBounds {
  const iw = img.naturalWidth;
  const ih = img.naturalHeight;
  if (!iw || !ih) return { sx: 0, sy: 0, sw: iw, sh: ih };
  // OffscreenCanvas is not available in some Safari/iOS versions — fall back to a
  // regular in-memory <canvas> element so garment preprocessing works cross-browser.
  let ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null;
  if (typeof OffscreenCanvas !== 'undefined') {
    const oc = new OffscreenCanvas(iw, ih);
    ctx = oc.getContext('2d');
  } else {
    const el = document.createElement('canvas');
    el.width = iw; el.height = ih;
    ctx = el.getContext('2d');
  }
  if (!ctx) return { sx: 0, sy: 0, sw: iw, sh: ih };
  ctx.drawImage(img, 0, 0);
  const data = ctx.getImageData(0, 0, iw, ih).data;
  let x0 = iw, x1 = 0, y0 = ih, y1 = 0;
  for (let y = 0; y < ih; y++) {
    for (let x = 0; x < iw; x++) {
      if (data[(y * iw + x) * 4 + 3] > 8) {
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
  }
  if (x1 < x0) return { sx: 0, sy: 0, sw: iw, sh: ih };
  return { sx: x0, sy: y0, sw: x1 - x0 + 1, sh: y1 - y0 + 1 };
}

/**
 * Arm foreground compositor.
 * Clips a polygon from shoulder → elbow → wrist and draws video pixels
 * into it — so the arm appears *in front of* the garment without needing
 * a separate segmentation model.
 *
 * cv2 equivalent: create an arm mask with fillPoly, then composite with
 * the original frame.
 */
function paintArmOverGarment(
  ctx: CanvasRenderingContext2D,
  image: unknown,
  shoulder: Pt,
  elbow: Pt,
  wrist: Pt,
  halfWidth: number,
  canvasW: number,
  canvasH: number,
) {
  ctx.save();

  // Upper-arm segment perpendicular
  const uaAngle = Math.atan2(elbow.y - shoulder.y, elbow.x - shoulder.x);
  const uaDx = Math.cos(uaAngle + Math.PI / 2) * halfWidth;
  const uaDy = Math.sin(uaAngle + Math.PI / 2) * halfWidth;

  // Forearm segment perpendicular
  const faAngle = Math.atan2(wrist.y - elbow.y, wrist.x - elbow.x);
  const faDx = Math.cos(faAngle + Math.PI / 2) * halfWidth;
  const faDy = Math.sin(faAngle + Math.PI / 2) * halfWidth;

  // Hexagonal arm polygon
  ctx.beginPath();
  ctx.moveTo(shoulder.x + uaDx, shoulder.y + uaDy);
  ctx.lineTo(elbow.x   + uaDx, elbow.y   + uaDy);
  ctx.lineTo(wrist.x   + faDx, wrist.y   + faDy);
  ctx.lineTo(wrist.x   - faDx, wrist.y   - faDy);
  ctx.lineTo(elbow.x   - uaDx, elbow.y   - uaDy);
  ctx.lineTo(shoulder.x - uaDx, shoulder.y - uaDy);
  ctx.closePath();
  ctx.clip();

  // Paint the raw video frame back inside the clip → arm texture on top of garment
  ctx.drawImage(image as unknown as CanvasImageSource, 0, 0, canvasW, canvasH);

  ctx.restore();
}

// ─── Hook ─────────────────────────────────────────────────────────────────────
export function useTryOn() {
  const videoRef  = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const poseRef   = useRef<Pose | null>(null);
  const animRef   = useRef<number | null>(null);

  const [selectedGarment, setSelectedGarment] = useState<string | null>(null);
  const [opacity, setOpacity]                 = useState<number>(75);
  const [poseStatus, setPoseStatus]           = useState<string>('Initializing camera...');
  const [fps, setFps]                         = useState<number>(0);
  const [webcamError, setWebcamError]         = useState<string | null>(null);

  const garmentImgRef       = useRef<HTMLImageElement | null>(null);
  const cropBoundsRef       = useRef<CropBounds | null>(null);
  const landmarksHistoryRef = useRef<NormalizedLandmarkList[]>([]);
  const lastFrameTimeRef    = useRef<number>(Date.now());
  const fpsHistoryRef       = useRef<number[]>([]);
  // opacity ref avoids stale closure inside onResults
  const opacityRef = useRef<number>(75);
  useEffect(() => { opacityRef.current = opacity; }, [opacity]);

  // ── Load garment + compute crop bounds once on selection ─────────────────
  useEffect(() => {
    if (!selectedGarment) { garmentImgRef.current = null; cropBoundsRef.current = null; return; }
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = selectedGarment;
    img.onload = () => {
      garmentImgRef.current = img;
      cropBoundsRef.current = computeCropBounds(img);
    };
  }, [selectedGarment]);

  // ── Core render callback ─────────────────────────────────────────────────
  const onResults = useCallback((results: PoseResults) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;

    // FPS
    const now = Date.now();
    fpsHistoryRef.current.push(1000 / Math.max(1, now - lastFrameTimeRef.current));
    lastFrameTimeRef.current = now;
    if (fpsHistoryRef.current.length > 10) fpsHistoryRef.current.shift();
    setFps(Math.round(fpsHistoryRef.current.reduce((a, b) => a + b, 0) / fpsHistoryRef.current.length));

    ctx.clearRect(0, 0, W, H);

    // ── Enter mirrored coordinate space ───────────────────────────────────
    // All drawing happens in "webcam space" (landmark coords × W/H map directly).
    // ctx.scale(-1,1) makes everything appear correctly flipped on screen.
    ctx.save();
    ctx.translate(W, 0);
    ctx.scale(-1, 1);

    // Step 1: background — raw mirrored video frame
    ctx.drawImage(results.image as unknown as CanvasImageSource, 0, 0, W, H);

    const lms = results.poseLandmarks;
    if (lms) {
      // Smooth landmarks over rolling window of N frames
      landmarksHistoryRef.current.push(lms);
      if (landmarksHistoryRef.current.length > HISTORY_SIZE) landmarksHistoryRef.current.shift();

      const smoothed = lms.map((_, i) => {
        const frames = landmarksHistoryRef.current;
        let x = 0, y = 0, v = 0;
        for (const f of frames) { x += f[i].x; y += f[i].y; v += (f[i].visibility ?? 0); }
        const n = frames.length;
        return { x: x / n, y: y / n, visibility: v / n };
      });

      const ls   = smoothed[LEFT_SHOULDER];
      const rs   = smoothed[RIGHT_SHOULDER];
      const lh   = smoothed[LEFT_HIP];
      const rh   = smoothed[RIGHT_HIP];
      const nose = smoothed[NOSE];
      const le   = smoothed[LEFT_ELBOW];
      const re   = smoothed[RIGHT_ELBOW];
      const lw   = smoothed[LEFT_WRIST];
      const rw   = smoothed[RIGHT_WRIST];

      const poseOk = visible(ls) && visible(rs) && visible(lh) && visible(rh);

      if (!poseOk) {
        setPoseStatus('Move back for full body view');
      } else {
        setPoseStatus('Pose detected');

        const garment = garmentImgRef.current;
        const crop    = cropBoundsRef.current;

        if (garment && crop) {
          // Landmark → pixel coords (in webcam/mirrored space)
          const lsPx   = toPx(ls,   W, H);
          const rsPx   = toPx(rs,   W, H);
          const nosePx = toPx(nose, W, H);
          const lePx   = toPx(le,   W, H);
          const rePx   = toPx(re,   W, H);
          const lwPx   = toPx(lw,   W, H);
          const rwPx   = toPx(rw,   W, H);

          const shoulderMid   = mid(lsPx, rsPx);
          const shoulderWidth = dist(lsPx, rsPx);

          // Garment dimensions — width from shoulder span, height from crop aspect ratio
          let gw = Math.min(shoulderWidth * GARMENT_W_SCALE, W * MAX_GARMENT_W);
          const cropAspect = crop.sw / crop.sh;
          let gh = gw / cropAspect;
          if (gh > H * MAX_GARMENT_H) { gh = H * MAX_GARMENT_H; gw = gh * cropAspect; }

          // Collar Y — 25 % of the way from shoulder toward nose (neck level, not chest)
          const neckY      = shoulderMid.y - (shoulderMid.y - nosePx.y) * 0.25;
          const collarDrop = shoulderMid.y - neckY;

          // Shoulder tilt → garment rotation (cv2.warpPerspective equivalent)
          const tiltAngle = Math.atan2(rsPx.y - lsPx.y, rsPx.x - lsPx.x);

          // ── Step 2: draw garment rotated to shoulder angle ───────────────
          ctx.save();
          ctx.translate(shoulderMid.x, shoulderMid.y);
          ctx.rotate(tiltAngle);

          const gx = -gw / 2;
          const gy = -collarDrop - gh * 0.02; // tiny upward nudge so collar sits right

          // Light drop shadow below garment hem (before garment so it's underneath)
          {
            const shadowTop  = gy + gh;
            const grad = ctx.createLinearGradient(gx, shadowTop, gx, shadowTop + 28);
            grad.addColorStop(0, 'rgba(0,0,0,0.28)');
            grad.addColorStop(1, 'rgba(0,0,0,0)');
            ctx.save();
            ctx.globalAlpha = 0.85;
            ctx.fillStyle = grad;
            ctx.fillRect(gx - 6, shadowTop, gw + 12, 28);
            ctx.restore();
          }

          // Edge halo — blurred oversize copy → GaussianBlur alpha-edge effect
          ctx.save();
          ctx.filter = `blur(${EDGE_BLUR_PX}px)`;
          ctx.globalAlpha = (opacityRef.current / 100) * 0.5;
          ctx.drawImage(garment, crop.sx, crop.sy, crop.sw, crop.sh, gx - 4, gy - 4, gw + 8, gh + 8);
          ctx.restore();

          // Crisp garment on top of halo
          ctx.globalAlpha = opacityRef.current / 100;
          ctx.drawImage(garment, crop.sx, crop.sy, crop.sw, crop.sh, gx, gy, gw, gh);
          ctx.globalAlpha = 1;

          ctx.restore(); // un-rotate

          // ── Step 3: arm foreground — landmark polygon clipped from video ──
          // Paints real arm pixels back over the garment so arms appear in front.
          // Same net effect as segmentation-based compositing but uses only the
          // Pose model (one WASM module, ~half the memory).
          const armHW = shoulderWidth * ARM_WIDTH_RATIO;

          if (visible(ls) && visible(le) && visible(lw)) {
            paintArmOverGarment(ctx, results.image, lsPx, lePx, lwPx, armHW, W, H);
          }
          if (visible(rs) && visible(re) && visible(rw)) {
            paintArmOverGarment(ctx, results.image, rsPx, rePx, rwPx, armHW, W, H);
          }

        }
      }
    } else {
      setPoseStatus('Stand back so we can see your full body');
    }

    ctx.restore(); // exit mirrored space
  }, []);

  // ── Camera init ───────────────────────────────────────────────────────────
  useEffect(() => {
    let active = true;
    navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } },
    }).then((stream) => {
      if (!active || !videoRef.current) return;
      videoRef.current.srcObject = stream;
      videoRef.current.play().catch(() => {});
    }).catch(() => {
      if (!active) return;
      setWebcamError('Camera access required — please allow camera permissions and reload');
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
      locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5.1675469404/${file}`,
    });
    pose.setOptions({
      modelComplexity:        1,
      smoothLandmarks:        true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence:  0.5,
    });
    pose.onResults(onResults);
    poseRef.current = pose;

    return () => {
      poseRef.current = null;
      pose.close().catch(() => {});
    };
  }, [onResults]);

  // ── Animation loop ────────────────────────────────────────────────────────
  useEffect(() => {
    async function loop() {
      const video  = videoRef.current;
      const canvas = canvasRef.current;
      if (video && video.readyState >= 2 && poseRef.current) {
        if (canvas && video.videoWidth > 0 && canvas.width !== video.videoWidth) {
          canvas.width  = video.videoWidth;
          canvas.height = video.videoHeight;
        }
        try { await poseRef.current.send({ image: video }); } catch { /* ignore */ }
      }
      animRef.current = requestAnimationFrame(loop);
    }
    animRef.current = requestAnimationFrame(loop);
    return () => { if (animRef.current) cancelAnimationFrame(animRef.current); };
  }, []);

  // ── Capture look ─────────────────────────────────────────────────────────
  const captureLook = useCallback(() => {
    if (!selectedGarment) return false;
    canvasRef.current?.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = 'virtual-tryon-capture.png'; a.click();
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
  };
}
