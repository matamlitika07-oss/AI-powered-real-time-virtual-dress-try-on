/**
 * useTryOn — Virtual Try-On Hook
 *
 * PHASE 1 (current): Torso detection + debug visualization only.
 *   - Video drawn straight (no mirror transform).
 *   - Landmark coords mapped directly: pixel = normalised * canvas dimension.
 *   - Draws 4 torso corner points, torso polygon, and garment bounding box.
 *   - No clothing rendered yet.
 *
 * PHASE 2 (next): Verify alignment, then apply perspective warp.
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import { Pose, Results as PoseResults } from '@mediapipe/pose';

// ─── MediaPipe landmark indices ───────────────────────────────────────────────
const IDX_LEFT_SHOULDER  = 11;
const IDX_RIGHT_SHOULDER = 12;
const IDX_LEFT_HIP       = 23;
const IDX_RIGHT_HIP      = 24;

// ─── Visibility threshold ─────────────────────────────────────────────────────
const VIS_THRESHOLD = 0.5;

// ─── Helpers ─────────────────────────────────────────────────────────────────
interface Pt { x: number; y: number }

/** Convert a normalised MediaPipe landmark to canvas pixel coordinates. */
function lmToPx(lm: { x: number; y: number }, W: number, H: number): Pt {
  return { x: lm.x * W, y: lm.y * H };
}

function isVisible(lm: { visibility?: number }): boolean {
  return (lm.visibility ?? 0) >= VIS_THRESHOLD;
}

// ─── Hook ─────────────────────────────────────────────────────────────────────
export function useTryOn() {
  const videoRef  = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const poseRef   = useRef<Pose | null>(null);
  const animRef   = useRef<number | null>(null);

  const [selectedGarment, setSelectedGarment] = useState<string | null>(null);
  const [opacity, setOpacity]                 = useState<number>(75);
  const [poseStatus, setPoseStatus]           = useState<string>('Initializing...');
  const [fps, setFps]                         = useState<number>(0);
  const [webcamError, setWebcamError]         = useState<string | null>(null);

  const lastFrameTimeRef = useRef<number>(Date.now());
  const fpsHistoryRef    = useRef<number[]>([]);

  // ── Phase 1 render ────────────────────────────────────────────────────────
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
    if (fpsHistoryRef.current.length > 12) fpsHistoryRef.current.shift();
    setFps(Math.round(fpsHistoryRef.current.reduce((a, b) => a + b, 0) / fpsHistoryRef.current.length));

    ctx.clearRect(0, 0, W, H);

    // ── Step 1: draw raw video frame — NO mirror transform.
    // Landmark coords are in the same coordinate space as the raw frame,
    // so drawing them directly on top of an un-flipped frame keeps everything
    // aligned. Previous bug: applying ctx.scale(-1,1) double-flipped landmarks.
    ctx.drawImage(results.image as unknown as CanvasImageSource, 0, 0, W, H);

    const lms = results.poseLandmarks;
    if (!lms) {
      setPoseStatus('No pose detected — stand back so your full body is visible');
      return;
    }

    const ls = lms[IDX_LEFT_SHOULDER];
    const rs = lms[IDX_RIGHT_SHOULDER];
    const lh = lms[IDX_LEFT_HIP];
    const rh = lms[IDX_RIGHT_HIP];

    const allVisible = isVisible(ls) && isVisible(rs) && isVisible(lh) && isVisible(rh);
    if (!allVisible) {
      setPoseStatus('Move back — need to see shoulders and hips');
      return;
    }

    // Convert all 4 torso corners to pixel positions.
    // ORDER (matches cv2.getPerspectiveTransform convention):
    //   tl = left-shoulder,  tr = right-shoulder
    //   bl = left-hip,       br = right-hip
    const tl = lmToPx(ls, W, H); // top-left  (left shoulder)
    const tr = lmToPx(rs, W, H); // top-right (right shoulder)
    const bl = lmToPx(lh, W, H); // bottom-left  (left hip)
    const br = lmToPx(rh, W, H); // bottom-right (right hip)

    // ── Step 2: draw torso polygon ───────────────────────────────────────
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(tl.x, tl.y);
    ctx.lineTo(tr.x, tr.y);
    ctx.lineTo(br.x, br.y);
    ctx.lineTo(bl.x, bl.y);
    ctx.closePath();
    ctx.strokeStyle = 'rgba(0, 255, 128, 0.9)';
    ctx.lineWidth   = 2;
    ctx.stroke();
    ctx.fillStyle   = 'rgba(0, 255, 128, 0.08)';
    ctx.fill();
    ctx.restore();

    // ── Step 3: draw garment bounding box (same quad — expanded outward) ──
    // This is the destination quad that the garment will be warped into
    // once Phase 2 confirms alignment is correct.
    // Expand each point slightly outward from the torso centroid.
    const cx = (tl.x + tr.x + bl.x + br.x) / 4;
    const cy = (tl.y + tr.y + bl.y + br.y) / 4;
    const EXPAND = 1.15; // 15% outward expansion

    function expand(pt: Pt): Pt {
      return { x: cx + (pt.x - cx) * EXPAND, y: cy + (pt.y - cy) * EXPAND };
    }

    const dtl = expand(tl);
    const dtr = expand(tr);
    const dbl = expand(bl);
    const dbr = expand(br);

    ctx.save();
    ctx.beginPath();
    ctx.moveTo(dtl.x, dtl.y);
    ctx.lineTo(dtr.x, dtr.y);
    ctx.lineTo(dbr.x, dbr.y);
    ctx.lineTo(dbl.x, dbl.y);
    ctx.closePath();
    ctx.strokeStyle = 'rgba(255, 215, 0, 0.85)'; // gold = future garment destination
    ctx.lineWidth   = 2;
    ctx.setLineDash([6, 4]);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();

    // ── Step 4: draw labeled corner dots ─────────────────────────────────
    const corners: { pt: Pt; label: string; color: string }[] = [
      { pt: tl,  label: 'TL (L.Shoulder)', color: '#00ff80' },
      { pt: tr,  label: 'TR (R.Shoulder)', color: '#00ff80' },
      { pt: bl,  label: 'BL (L.Hip)',      color: '#00aaff' },
      { pt: br,  label: 'BR (R.Hip)',      color: '#00aaff' },
    ];

    for (const { pt, label, color } of corners) {
      // Dot
      ctx.save();
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 7, 0, Math.PI * 2);
      ctx.fillStyle   = color;
      ctx.strokeStyle = '#000';
      ctx.lineWidth   = 2;
      ctx.fill();
      ctx.stroke();
      ctx.restore();

      // Label
      ctx.save();
      ctx.font         = 'bold 11px monospace';
      ctx.fillStyle    = '#fff';
      ctx.strokeStyle  = '#000';
      ctx.lineWidth    = 3;
      ctx.textAlign    = 'center';
      const labelX = pt.x;
      const labelY = pt.y - 12;
      ctx.strokeText(label, labelX, labelY);
      ctx.fillText(label, labelX, labelY);
      ctx.restore();
    }

    // ── Step 5: diagonal cross-check lines (sanity — TL↔BR and TR↔BL) ────
    ctx.save();
    ctx.setLineDash([3, 6]);
    ctx.strokeStyle = 'rgba(255,255,255,0.2)';
    ctx.lineWidth   = 1;
    ctx.beginPath(); ctx.moveTo(tl.x, tl.y); ctx.lineTo(br.x, br.y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(tr.x, tr.y); ctx.lineTo(bl.x, bl.y); ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();

    // ── Phase label overlay ───────────────────────────────────────────────
    const phaseText = 'PHASE 1 — TORSO DETECTION';
    ctx.save();
    ctx.font      = 'bold 12px monospace';
    ctx.fillStyle = 'rgba(0,0,0,0.55)';
    ctx.fillRect(8, 8, ctx.measureText(phaseText).width + 16, 24);
    ctx.fillStyle = '#00ff80';
    ctx.fillText(phaseText, 16, 25);
    ctx.restore();

    setPoseStatus('Phase 1 — torso polygon drawn. Verify alignment.');
  }, []);

  // ── Camera ────────────────────────────────────────────────────────────────
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

  // ── MediaPipe Pose ────────────────────────────────────────────────────────
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
        // Sync canvas size to video once
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
      a.href = url; a.download = 'virtual-tryon-phase1.png'; a.click();
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
