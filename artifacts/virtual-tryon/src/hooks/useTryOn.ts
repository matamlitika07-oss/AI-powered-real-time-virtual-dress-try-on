import { useEffect, useRef, useState, useCallback } from 'react';
import { Pose, Results, NormalizedLandmarkList } from '@mediapipe/pose';

interface Point {
  x: number;
  y: number;
}

const LEFT_SHOULDER = 11;
const RIGHT_SHOULDER = 12;
const LEFT_ELBOW = 13;
const RIGHT_ELBOW = 14;
const LEFT_WRIST = 15;
const RIGHT_WRIST = 16;
const LEFT_HIP = 23;
const RIGHT_HIP = 24;

const HISTORY_SIZE = 5;

function midpoint(p1: Point, p2: Point): Point {
  return { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 };
}

function distance(p1: Point, p2: Point): number {
  return Math.hypot(p2.x - p1.x, p2.y - p1.y);
}

export function useTryOn() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const poseRef = useRef<Pose | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  
  const [selectedGarment, setSelectedGarment] = useState<string | null>(null);
  const [opacity, setOpacity] = useState<number>(75);
  const [poseStatus, setPoseStatus] = useState<string>("Initializing camera...");
  const [fps, setFps] = useState<number>(0);
  const [webcamError, setWebcamError] = useState<string | null>(null);

  const garmentImgRef = useRef<HTMLImageElement | null>(null);
  const landmarksHistoryRef = useRef<NormalizedLandmarkList[]>([]);
  
  const lastFrameTimeRef = useRef<number>(Date.now());
  const fpsHistoryRef = useRef<number[]>([]);

  // Load garment image when selected
  useEffect(() => {
    if (!selectedGarment) {
      garmentImgRef.current = null;
      return;
    }
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = selectedGarment;
    img.onload = () => {
      garmentImgRef.current = img;
    };
  }, [selectedGarment]);

  const onResults = useCallback((results: Results) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Calculate FPS
    const now = Date.now();
    const currentFps = 1000 / (now - lastFrameTimeRef.current);
    lastFrameTimeRef.current = now;
    
    fpsHistoryRef.current.push(currentFps);
    if (fpsHistoryRef.current.length > 10) fpsHistoryRef.current.shift();
    const avgFps = fpsHistoryRef.current.reduce((a, b) => a + b, 0) / fpsHistoryRef.current.length;
    setFps(Math.round(avgFps));

    // Clear and draw mirrored video frame
    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Mirror horizontally
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    
    // Draw raw frame
    ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

    if (results.poseLandmarks) {
      // Smooth landmarks
      landmarksHistoryRef.current.push(results.poseLandmarks);
      if (landmarksHistoryRef.current.length > HISTORY_SIZE) {
        landmarksHistoryRef.current.shift();
      }

      // Average landmarks
      const smoothedLandmarks = results.poseLandmarks.map((_, i) => {
        let x = 0, y = 0, z = 0, visibility = 0;
        const validFrames = landmarksHistoryRef.current.filter(lms => lms[i]);
        if (validFrames.length === 0) return results.poseLandmarks[i];
        
        for (const lms of validFrames) {
          x += lms[i].x;
          y += lms[i].y;
          z += lms[i].z;
          visibility += lms[i].visibility || 0;
        }
        const len = validFrames.length;
        return { x: x / len, y: y / len, z: z / len, visibility: visibility / len };
      });

      const ls = smoothedLandmarks[LEFT_SHOULDER];
      const rs = smoothedLandmarks[RIGHT_SHOULDER];
      const lh = smoothedLandmarks[LEFT_HIP];
      const rh = smoothedLandmarks[RIGHT_HIP];

      // Because canvas is mirrored, x coords need to be flipped relative to image width
      // results.image.width vs canvas.width
      // Mediapip returns normalized [0,1]. When we draw we already have ctx flipped!
      // Wait, if ctx is flipped, we can draw the garment at its mirrored x coordinate,
      // or we can restore ctx, flip the landmarks conceptually, and draw normally.
      // Let's draw while ctx is still flipped: the normalized coordinates map directly onto the flipped video image.
      
      const vT = 0.5; // visibility threshold
      if ((ls.visibility || 0) < vT || (rs.visibility || 0) < vT || (lh.visibility || 0) < vT || (rh.visibility || 0) < vT) {
        setPoseStatus("Move back for full body view");
      } else {
        setPoseStatus("Pose detected");

        if (garmentImgRef.current) {
          const lSp = { x: ls.x * canvas.width, y: ls.y * canvas.height };
          const rSp = { x: rs.x * canvas.width, y: rs.y * canvas.height };
          const lHp = { x: lh.x * canvas.width, y: lh.y * canvas.height };
          const rHp = { x: rh.x * canvas.width, y: rh.y * canvas.height };

          const shoulderMid = midpoint(lSp, rSp);
          const hipMid = midpoint(lHp, rHp);
          const shoulderWidth = distance(lSp, rSp);

          let garmentWidth = shoulderWidth * 1.5;
          let garmentHeight = distance(shoulderMid, hipMid) * 1.5;

          garmentWidth = Math.min(garmentWidth, canvas.width * 0.8);
          garmentHeight = Math.min(garmentHeight, canvas.height * 0.9);

          const garmentX = shoulderMid.x - garmentWidth / 2;
          const garmentY = shoulderMid.y - garmentHeight * 0.12;

          // Draw garment with opacity
          ctx.globalAlpha = opacity / 100;
          ctx.shadowBlur = 20;
          ctx.shadowColor = 'rgba(0,0,0,0.5)';
          
          ctx.drawImage(garmentImgRef.current, garmentX, garmentY, garmentWidth, garmentHeight);
          
          // Reset shadow and alpha
          ctx.shadowBlur = 0;
          ctx.globalAlpha = 1.0;

          // Arm masking (source-over flesh colored ellipses over forearms)
          const maskArm = (elbowIdx: number, wristIdx: number) => {
            const elbow = smoothedLandmarks[elbowIdx];
            const wrist = smoothedLandmarks[wristIdx];
            if ((elbow.visibility || 0) > 0.4 && (wrist.visibility || 0) > 0.4) {
              const e = { x: elbow.x * canvas.width, y: elbow.y * canvas.height };
              const w = { x: wrist.x * canvas.width, y: wrist.y * canvas.height };
              
              const dist = distance(e, w);
              const angle = Math.atan2(w.y - e.y, w.x - e.x);
              const mid = midpoint(e, w);
              
              ctx.save();
              ctx.translate(mid.x, mid.y);
              ctx.rotate(angle);
              // We don't have user's skin tone, but we can do a composite operation
              // Wait, globalCompositeOperation = 'destination-out' to erase the garment where the arm is?
              // That would erase the garment and show the video underneath! 
              // Wait, if we use destination-out, it will erase everything including the video, leaving transparent canvas.
              // We would need to draw the video again underneath.
              // Actually, drawing the arm again from the original video frame would be perfect, but complex.
              // The instructions said: "draw flesh-colored ellipses over the forearm positions... using ctx.globalCompositeOperation = 'source-over'"
              ctx.globalCompositeOperation = 'source-over';
              ctx.fillStyle = 'rgba(235, 196, 159, 0.6)'; // generic flesh tone semi-transparent
              ctx.beginPath();
              ctx.ellipse(0, 0, dist / 2, dist * 0.25, 0, 0, Math.PI * 2);
              ctx.fill();
              ctx.restore();
            }
          };

          maskArm(LEFT_ELBOW, LEFT_WRIST);
          maskArm(RIGHT_ELBOW, RIGHT_WRIST);
        }
      }
    } else {
      setPoseStatus("Stand back so we can see your full body");
    }

    ctx.restore();
  }, [opacity]);

  useEffect(() => {
    let active = true;

    async function initCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'environment', width: 640, height: 480 }
        });
        if (!active) return;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play();
        }
      } catch (err: any) {
        if (!active) return;
        setWebcamError("Camera access required — please allow camera permissions");
        setPoseStatus("Camera error");
      }
    }

    initCamera();

    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    let active = true;

    const pose = new Pose({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5.1675469404/${file}`
    });

    pose.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });

    pose.onResults(onResults);
    poseRef.current = pose;

    return () => {
      active = false;
      if (poseRef.current) {
        poseRef.current.close();
      }
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [onResults]);

  useEffect(() => {
    let video = videoRef.current;
    
    async function loop() {
      if (!video) return;
      if (video.readyState >= 2 && poseRef.current) {
        // resize canvas to match video
        if (canvasRef.current) {
          if (canvasRef.current.width !== video.videoWidth) {
            canvasRef.current.width = video.videoWidth;
            canvasRef.current.height = video.videoHeight;
          }
        }
        await poseRef.current.send({ image: video });
      }
      animationFrameRef.current = requestAnimationFrame(loop);
    }

    animationFrameRef.current = requestAnimationFrame(loop);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  const captureLook = useCallback(() => {
    if (!selectedGarment) {
      return false; // Handled by UI to show toast
    }
    const canvas = canvasRef.current;
    if (canvas) {
      canvas.toBlob(blob => {
        if (blob) {
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = 'capture.png';
          a.click();
          URL.revokeObjectURL(url);
        }
      }, 'image/png');
    }
    return true;
  }, [selectedGarment]);

  return {
    videoRef,
    canvasRef,
    selectedGarment,
    setSelectedGarment,
    opacity,
    setOpacity,
    poseStatus,
    fps,
    webcamError,
    captureLook
  };
}
