import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { FilesetResolver, GestureRecognizer } from "@mediapipe/tasks-vision";

const assetUrl = (path) => `${import.meta.env.BASE_URL}${path.replace(/^\/+/, "")}`;

const REVEAL_IMAGES = [
  assetUrl("demo/show_1.jpg"),
  assetUrl("demo/show_2.jpg"),
  assetUrl("demo/show_3.jpg"),
  assetUrl("demo/show_4.jpg"),
  assetUrl("demo/show_5.jpg"),
  assetUrl("demo/show_6.jpg"),
  assetUrl("demo/show_7.jpg"),
  assetUrl("demo/show_8.png"),
  assetUrl("demo/show_9.jpg"),
  assetUrl("demo/show_10.jpg"),
];

const FINAL_IMAGE = assetUrl("demo/final.png");
const PANELS_PER_ROUND = 5;
const ROUND_COUNT = Math.ceil(REVEAL_IMAGES.length / PANELS_PER_ROUND);

const ROUND_PANEL_LAYOUT = [
  { left: 4.2, top: 8.5, width: 19, height: 34 },
  { left: 25.5, top: 28, width: 13.5, height: 39 },
  { left: 41.8, top: 10.5, width: 20.2, height: 35 },
  { left: 64.7, top: 30.5, width: 13.8, height: 38 },
  { left: 80.2, top: 9.5, width: 16.2, height: 35 },
];

const MODEL_URL = assetUrl("models/gesture_recognizer.task");
const WASM_ROOT = assetUrl("wasm");
const FIREWORK_SPARKS = 12;
const FINAL_FIREWORKS = [
  { x: 8, y: 18, delay: 0, color: "#f9e36d" },
  { x: 91, y: 17, delay: 130, color: "#9cf7ee" },
  { x: 17, y: 76, delay: 240, color: "#ffadc8" },
  { x: 84, y: 78, delay: 360, color: "#c7ff8a" },
  { x: 50, y: 7, delay: 470, color: "#ffffff" },
  { x: 7, y: 50, delay: 620, color: "#9fc7ff" },
  { x: 94, y: 48, delay: 760, color: "#ffd39c" },
  { x: 50, y: 91, delay: 890, color: "#d8b8ff" },
];

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function createPanels(roundIndex = 0, images = REVEAL_IMAGES) {
  const imageOffset = roundIndex * PANELS_PER_ROUND;

  return ROUND_PANEL_LAYOUT.map((panel, index) => ({
    ...panel,
    id: imageOffset + index + 1,
    round: roundIndex,
    image: images[(imageOffset + index) % images.length],
    progress: 0,
    pullX: 0,
    pullY: 0,
    grabX: 50,
    grabY: 50,
    angle: 0,
    curl: 0,
    trailAngle: 180,
    tailX: 50,
    tailY: 50,
    seamX: 50,
    seamY: 50,
    revealed: false,
    armedAt: 0,
  }));
}

function pointInsidePanel(point, panel) {
  return (
    point.x >= panel.left &&
    point.x <= panel.left + panel.width &&
    point.y >= panel.top &&
    point.y <= panel.top + panel.height
  );
}

function getPalmPoint(landmarks, mirrored) {
  const palmIndexes = [0, 5, 9, 13, 17];
  const palm = palmIndexes.reduce(
    (acc, index) => {
      acc.x += landmarks[index].x;
      acc.y += landmarks[index].y;
      return acc;
    },
    { x: 0, y: 0 },
  );

  const x = palm.x / palmIndexes.length;
  const y = palm.y / palmIndexes.length;

  return {
    x: (mirrored ? 1 - x : x) * 100,
    y: y * 100,
  };
}

function landmarkToStagePoint(landmark, mirrored) {
  return {
    x: (mirrored ? 1 - landmark.x : landmark.x) * 100,
    y: landmark.y * 100,
  };
}

function getPinchInfo(landmarks, mirrored) {
  const thumb = landmarks[4];
  const index = landmarks[8];
  const thumbPoint = landmarkToStagePoint(thumb, mirrored);
  const indexPoint = landmarkToStagePoint(index, mirrored);

  return {
    distance: Math.hypot(thumb.x - index.x, thumb.y - index.y),
    point: {
      x: (thumbPoint.x + indexPoint.x) / 2,
      y: (thumbPoint.y + indexPoint.y) / 2,
    },
  };
}

function getLocalPoint(point, panel) {
  return {
    x: clamp(((point.x - panel.left) / panel.width) * 100, 0, 100),
    y: clamp(((point.y - panel.top) / panel.height) * 100, 0, 100),
  };
}

function getPullMetrics(panel, drag, point) {
  const dx = point.x - drag.startX;
  const dy = point.y - drag.startY;
  const pullX = clamp((dx / panel.width) * 100, -135, 135);
  const pullY = clamp((dy / panel.height) * 100, -135, 135);
  const normalizedDistance = Math.hypot(dx / panel.width, dy / panel.height);
  const progress = clamp(normalizedDistance / 0.78, 0, 1);
  const angle = clamp(pullX * 0.035 + pullY * 0.025, -9, 9);
  const direction = Number.isFinite(Math.atan2(dy, dx))
    ? Math.atan2(dy, dx) * (180 / Math.PI)
    : 0;
  const isHorizontal = Math.abs(pullX) >= Math.abs(pullY);
  const tailX = isHorizontal ? (pullX >= 0 ? 0 : 100) : panel.grabX;
  const tailY = isHorizontal ? panel.grabY : pullY >= 0 ? 0 : 100;
  const seamX = isHorizontal
    ? clamp(pullX >= 0 ? pullX : 100 + pullX, 0, 100)
    : panel.grabX;
  const seamY = isHorizontal
    ? panel.grabY
    : clamp(pullY >= 0 ? pullY : 100 + pullY, 0, 100);

  return {
    progress,
    pullX,
    pullY,
    angle,
    curl: clamp(normalizedDistance * 1.15, 0, 1),
    trailAngle: direction + 180,
    tailX,
    tailY,
    seamX,
    seamY,
  };
}

function formatSvgNumber(value) {
  return Number(value.toFixed(2));
}

function svgPoint(point) {
  return `${formatSvgNumber(point.x)} ${formatSvgNumber(point.y)}`;
}

function lerpPoint(start, end, amount) {
  return {
    x: start.x + (end.x - start.x) * amount,
    y: start.y + (end.y - start.y) * amount,
  };
}

function addVector(point, vector, amount) {
  return {
    x: point.x + vector.x * amount,
    y: point.y + vector.y * amount,
  };
}

function curveBetween(start, end, normal, amount) {
  const first = addVector(lerpPoint(start, end, 0.32), normal, amount);
  const second = addVector(lerpPoint(start, end, 0.68), normal, -amount * 0.72);

  return `M ${svgPoint(start)} C ${svgPoint(first)} ${svgPoint(second)} ${svgPoint(end)}`;
}

function getCurtainGeometry(panel) {
  if (panel.revealed) {
    const hiddenPath = "M 220 220 L 221 220 L 221 221 L 220 221 Z";

    return {
      bodyPath: hiddenPath,
      edgePath: "M 220 220 L 221 221",
      foldPaths: [],
      threadPaths: [],
      gradient: { x1: 0, y1: 0, x2: 100, y2: 100 },
    };
  }

  const dx = panel.pullX;
  const dy = panel.pullY;
  const distance = Math.max(Math.hypot(dx, dy), 0.001);
  const direction = {
    x: dx / distance,
    y: dy / distance,
  };
  const normal = {
    x: -direction.y,
    y: direction.x,
  };
  const isHorizontalPull = Math.abs(dx) >= Math.abs(dy);
  const pullForward = isHorizontalPull ? dx >= 0 : dy >= 0;
  const progress = panel.progress;
  const curl = panel.curl;
  const trailingSlip = 0.06 + progress * 0.72;

  const leadAmount = (point) => {
    if (isHorizontalPull) {
      return pullForward ? point.x / 100 : 1 - point.x / 100;
    }

    return pullForward ? point.y / 100 : 1 - point.y / 100;
  };

  const deformPoint = (point) => {
    const lead = leadAmount(point);
    const moveRatio = trailingSlip + lead * (1.08 - trailingSlip);
    const moved = {
      x: point.x + dx * moveRatio,
      y: point.y + dy * moveRatio,
    };
    const cross = isHorizontalPull
      ? (point.y - panel.grabY) / 100
      : (point.x - panel.grabX) / 100;
    const foldTowardGrip = -cross * curl * 19 * (0.25 + lead * 0.75);
    const ripple =
      Math.sin((lead * 2.6 + point.x * 0.017 + point.y * 0.011 + progress) * Math.PI) *
      curl *
      4.6 *
      (0.35 + lead * 0.65);

    return addVector(addVector(moved, normal, foldTowardGrip), normal, ripple);
  };

  const topLeft = deformPoint({ x: 0, y: 0 });
  const topRight = deformPoint({ x: 100, y: 0 });
  const bottomRight = deformPoint({ x: 100, y: 100 });
  const bottomLeft = deformPoint({ x: 0, y: 100 });
  const topBend = curl * 8.8 + progress * 2.4;
  const sideBend = curl * 6.2 + progress * 1.8;
  const topC1 = addVector(lerpPoint(topLeft, topRight, 0.3), normal, topBend);
  const topC2 = addVector(lerpPoint(topLeft, topRight, 0.68), normal, -topBend * 0.7);
  const rightC1 = addVector(lerpPoint(topRight, bottomRight, 0.3), direction, sideBend);
  const rightC2 = addVector(lerpPoint(topRight, bottomRight, 0.72), direction, -sideBend * 0.55);
  const bottomC1 = addVector(lerpPoint(bottomRight, bottomLeft, 0.32), normal, -topBend);
  const bottomC2 = addVector(lerpPoint(bottomRight, bottomLeft, 0.7), normal, topBend * 0.72);
  const leftC1 = addVector(lerpPoint(bottomLeft, topLeft, 0.3), direction, -sideBend);
  const leftC2 = addVector(lerpPoint(bottomLeft, topLeft, 0.68), direction, sideBend * 0.55);
  const bodyPath = [
    `M ${svgPoint(topLeft)}`,
    `C ${svgPoint(topC1)} ${svgPoint(topC2)} ${svgPoint(topRight)}`,
    `C ${svgPoint(rightC1)} ${svgPoint(rightC2)} ${svgPoint(bottomRight)}`,
    `C ${svgPoint(bottomC1)} ${svgPoint(bottomC2)} ${svgPoint(bottomLeft)}`,
    `C ${svgPoint(leftC1)} ${svgPoint(leftC2)} ${svgPoint(topLeft)}`,
    "Z",
  ].join(" ");

  const topEdgePoint = (amount) =>
    addVector(
      lerpPoint(topLeft, topRight, amount),
      normal,
      Math.sin(amount * Math.PI * 2 + progress * 3.5) * curl * 3.4,
    );
  const bottomEdgePoint = (amount) =>
    addVector(
      lerpPoint(bottomLeft, bottomRight, amount),
      normal,
      Math.sin(amount * Math.PI * 2.2 + progress * 3.5 + 1.2) * curl * -3.2,
    );
  const leftEdgePoint = (amount) =>
    addVector(
      lerpPoint(topLeft, bottomLeft, amount),
      direction,
      Math.sin(amount * Math.PI * 2 + progress * 3.2) * curl * 2.8,
    );
  const rightEdgePoint = (amount) =>
    addVector(
      lerpPoint(topRight, bottomRight, amount),
      direction,
      Math.sin(amount * Math.PI * 2.2 + progress * 3.2 + 0.7) * curl * -2.8,
    );

  const foldPaths = Array.from({ length: 7 }, (_, index) => {
    const amount = (index + 1) / 8;
    const wave = Math.sin(index * 1.7 + progress * 7) * curl * 7;

    if (isHorizontalPull) {
      const start = topEdgePoint(amount);
      const end = bottomEdgePoint(amount);
      const first = addVector(lerpPoint(start, end, 0.34), direction, wave);
      const second = addVector(lerpPoint(start, end, 0.7), direction, -wave * 0.74);
      return `M ${svgPoint(start)} C ${svgPoint(first)} ${svgPoint(second)} ${svgPoint(end)}`;
    }

    const start = leftEdgePoint(amount);
    const end = rightEdgePoint(amount);
    const first = addVector(lerpPoint(start, end, 0.34), direction, wave);
    const second = addVector(lerpPoint(start, end, 0.7), direction, -wave * 0.74);
    return `M ${svgPoint(start)} C ${svgPoint(first)} ${svgPoint(second)} ${svgPoint(end)}`;
  });

  let edgeStart;
  let edgeEnd;

  if (isHorizontalPull) {
    edgeStart = pullForward ? topRight : topLeft;
    edgeEnd = pullForward ? bottomRight : bottomLeft;
  } else {
    edgeStart = pullForward ? bottomLeft : topLeft;
    edgeEnd = pullForward ? bottomRight : topRight;
  }

  const edgePath = curveBetween(edgeStart, edgeEnd, direction, sideBend * 1.2);
  const threadPaths = Array.from({ length: 5 }, (_, index) => {
    const amount = (index + 1) / 6;
    const base = lerpPoint(edgeStart, edgeEnd, amount);
    const looseness = (0.6 + index * 0.18) * curl * 16;
    const end = addVector(
      addVector(base, direction, 12 + curl * 34),
      normal,
      Math.sin(index * 1.35 + progress * 5) * looseness,
    );
    const first = addVector(lerpPoint(base, end, 0.32), normal, looseness * 0.58);
    const second = addVector(lerpPoint(base, end, 0.72), normal, -looseness * 0.46);

    return `M ${svgPoint(base)} C ${svgPoint(first)} ${svgPoint(second)} ${svgPoint(end)}`;
  });

  return {
    bodyPath,
    edgePath,
    foldPaths,
    threadPaths,
    gradient: {
      x1: formatSvgNumber(50 - direction.x * 52),
      y1: formatSvgNumber(50 - direction.y * 52),
      x2: formatSvgNumber(50 + direction.x * 52),
      y2: formatSvgNumber(50 + direction.y * 52),
    },
  };
}

function getStagePoint(event, stage) {
  if (!stage) return { x: 0, y: 0 };

  const rect = stage.getBoundingClientRect();
  return {
    x: clamp(((event.clientX - rect.left) / rect.width) * 100, 0, 100),
    y: clamp(((event.clientY - rect.top) / rect.height) * 100, 0, 100),
  };
}

function normalizeGestureName(name) {
  if (!name || name === "None") return "No gesture";
  return name.replaceAll("_", " ");
}

function getCameraErrorMessage(error) {
  if (!window.isSecureContext) {
    return {
      title: "需要 HTTPS 才能使用摄像头",
      hint: "请确认正在使用 https:// 链接访问页面。",
    };
  }

  switch (error?.name) {
    case "NotAllowedError":
    case "SecurityError":
      return {
        title: "摄像头权限被拒绝",
        hint: "请在浏览器或系统设置中允许摄像头；微信、QQ、部分国产浏览器可能会拦截，建议用 Chrome 或 Safari 打开。也可以直接用手指拖动幕布体验。",
      };
    case "NotFoundError":
    case "DevicesNotFoundError":
      return {
        title: "没有找到摄像头",
        hint: "请确认设备有可用摄像头，或直接用手指拖动幕布体验。",
      };
    case "NotReadableError":
    case "TrackStartError":
      return {
        title: "摄像头被占用",
        hint: "请关闭正在使用摄像头的其他应用后再试。",
      };
    default:
      return {
        title: error?.name ? `摄像头启动失败：${error.name}` : "摄像头启动失败",
        hint: "请检查浏览器摄像头权限，或换用 Chrome/Safari 打开。",
      };
  }
}

export default function App() {
  const stageRef = useRef(null);
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const recognizerRef = useRef(null);
  const rafRef = useRef(0);
  const panelsRef = useRef(createPanels(0));
  const dragRef = useRef(null);
  const lastVideoTimeRef = useRef(-1);
  const lastResultAtRef = useRef(0);

  const [panels, setPanels] = useState(() => panelsRef.current);
  const [roundIndex, setRoundIndex] = useState(0);
  const [showFinal, setShowFinal] = useState(false);
  const [cameraState, setCameraState] = useState("idle");
  const [modelState, setModelState] = useState("idle");
  const [statusText, setStatusText] = useState("System idle");
  const [statusHint, setStatusHint] = useState("");
  const [mirrored, setMirrored] = useState(true);
  const [zoom, setZoom] = useState(1);
  const [activeHand, setActiveHand] = useState(null);

  const revealedCount = useMemo(
    () => panels.filter((panel) => panel.revealed).length,
    [panels],
  );

  const syncPanels = useCallback((nextPanels) => {
    panelsRef.current = nextPanels;
    setPanels(nextPanels.map((panel) => ({ ...panel })));
  }, []);

  const resetPanels = useCallback(() => {
    dragRef.current = null;
    setRoundIndex(0);
    setShowFinal(false);
    syncPanels(createPanels(0));
    setStatusText(cameraState === "ready" ? "System active" : "System idle");
    setStatusHint("");
  }, [cameraState, syncPanels]);

  const updatePanelPull = useCallback(
    (panelId, nextState) => {
      syncPanels(
        panelsRef.current.map((panel) =>
          panel.id === panelId && !panel.revealed
            ? {
                ...panel,
                ...nextState,
                progress: Math.max(panel.progress, nextState.progress ?? 0),
                armedAt: 0,
              }
            : panel,
        ),
      );
    },
    [syncPanels],
  );

  const releasePanel = useCallback(
    (panelId, shouldReveal) => {
      syncPanels(
        panelsRef.current.map((panel) => {
          if (panel.id !== panelId || panel.revealed) return panel;

          if (shouldReveal) {
            return {
              ...panel,
              progress: 1,
              pullX: panel.pullX,
              pullY: panel.pullY,
              revealed: true,
              armedAt: 0,
            };
          }

          return {
            ...panel,
            progress: 0,
            pullX: 0,
            pullY: 0,
            angle: 0,
            curl: 0,
            trailAngle: 180,
            tailX: 50,
            tailY: 50,
            seamX: 50,
            seamY: 50,
            armedAt: 0,
          };
        }),
      );
    },
    [syncPanels],
  );

  const updatePanelsFromHand = useCallback(
    (handPoint, pinchInfo, gestureName) => {
      if (dragRef.current?.source === "pointer") return;

      const isPinching = pinchInfo?.distance < 0.065;
      const isGrabbing = gestureName === "Closed_Fist" || isPinching;
      const gripPoint = isPinching ? pinchInfo.point : handPoint;
      const activeDrag = dragRef.current?.source === "gesture" ? dragRef.current : null;

      if (activeDrag) {
        const activePanel = panelsRef.current.find((panel) => panel.id === activeDrag.panelId);

        if (!activePanel || activePanel.revealed) {
          dragRef.current = null;
          return;
        }

        if (!isGrabbing) {
          releasePanel(activePanel.id, activePanel.progress > 0.54);
          dragRef.current = null;
          setStatusText(activePanel.progress > 0.54 ? "Panel revealed" : "System active");
          return;
        }

        const metrics = getPullMetrics(activePanel, activeDrag, gripPoint);
        updatePanelPull(activePanel.id, metrics);
        setStatusText("Pinch and pull");

        if (metrics.progress >= 0.96) {
          releasePanel(activePanel.id, true);
          dragRef.current = null;
          setStatusText("Panel revealed");
        }
        return;
      }

      const target = panelsRef.current.find(
        (panel) => !panel.revealed && pointInsidePanel(gripPoint, panel),
      );

      if (!target) {
        return;
      }

      if (isGrabbing) {
        const localGrab = getLocalPoint(gripPoint, target);
        dragRef.current = {
          source: "gesture",
          panelId: target.id,
          startX: gripPoint.x,
          startY: gripPoint.y,
        };

        updatePanelPull(target.id, {
          progress: 0.06,
          pullX: 0,
          pullY: 0,
          grabX: localGrab.x,
          grabY: localGrab.y,
          angle: 0,
          curl: 0.08,
        });
        setStatusText("Pinch and pull");
      }
    },
    [releasePanel, updatePanelPull],
  );

  const predictFrame = useCallback(() => {
    const video = videoRef.current;
    const recognizer = recognizerRef.current;

    if (!video || !recognizer || video.readyState < 2) {
      rafRef.current = requestAnimationFrame(predictFrame);
      return;
    }

    const now = performance.now();
    if (video.currentTime !== lastVideoTimeRef.current) {
      try {
        const result = recognizer.recognizeForVideo(video, now);
        lastVideoTimeRef.current = video.currentTime;

        const landmarks = result.landmarks?.[0];
        const gesture = result.gestures?.[0]?.[0];

        if (landmarks) {
          const handPoint = getPalmPoint(landmarks, mirrored);
          const pinchInfo = getPinchInfo(landmarks, mirrored);
          const gestureName = gesture?.categoryName ?? "None";
          const isPinching = pinchInfo.distance < 0.065;
          const cursorPoint = isPinching ? pinchInfo.point : handPoint;

          lastResultAtRef.current = now;
          setActiveHand({
            x: cursorPoint.x,
            y: cursorPoint.y,
            gesture: isPinching ? "Pinch" : normalizeGestureName(gestureName),
            score: gesture?.score ?? 0,
          });
          if (!showFinal) {
            setStatusText("Hand tracking");
            updatePanelsFromHand(handPoint, pinchInfo, gestureName);
          }
        } else if (now - lastResultAtRef.current > 450) {
          setActiveHand(null);
          if (!showFinal) {
            setStatusText("System active");
          }
        }
      } catch (error) {
        console.error(error);
        setStatusText("Tracking error");
        setStatusHint("识别过程出错，可以点击 Stop 后重新 Start。");
      }
    }

    rafRef.current = requestAnimationFrame(predictFrame);
  }, [mirrored, showFinal, updatePanelsFromHand]);

  const startCamera = useCallback(async () => {
    if (cameraState === "starting" || cameraState === "ready") return;

    if (!navigator.mediaDevices?.getUserMedia || !window.isSecureContext) {
      const message = getCameraErrorMessage();
      setCameraState("error");
      setStatusText(message.title);
      setStatusHint(message.hint);
      return;
    }

    try {
      setCameraState("starting");
      setStatusText("Camera starting");
      setStatusHint("");

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });

      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();

      if (!recognizerRef.current) {
        setModelState("loading");
        setStatusText("Model loading");

        try {
          const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);
          recognizerRef.current = await GestureRecognizer.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath: MODEL_URL,
              delegate: "CPU",
            },
            runningMode: "VIDEO",
            numHands: 1,
            minHandDetectionConfidence: 0.45,
            minHandPresenceConfidence: 0.45,
            minTrackingConfidence: 0.45,
          });
          setModelState("ready");
        } catch (error) {
          console.error(error);
          setModelState("error");
          setCameraState("ready");
          setStatusText("Model unavailable");
          setStatusHint("手势模型加载失败，但仍可用手指拖动幕布。");
          return;
        }
      }

      setCameraState("ready");
      setStatusText("System active");
      setStatusHint("");
      cancelAnimationFrame(rafRef.current);
      rafRef.current = requestAnimationFrame(predictFrame);
    } catch (error) {
      console.error(error);
      const message = getCameraErrorMessage(error);
      setCameraState("error");
      setModelState((state) => (state === "loading" ? "idle" : state));
      setStatusText(message.title);
      setStatusHint(message.hint);
    }
  }, [cameraState, predictFrame]);

  const stopCamera = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraState("idle");
    setActiveHand(null);
    setStatusText("System idle");
    setStatusHint("");
  }, []);

  const beginPointerReveal = useCallback(
    (event, panel) => {
      if (panel.revealed) return;

      event.preventDefault();
      event.currentTarget.setPointerCapture?.(event.pointerId);

      const point = getStagePoint(event, stageRef.current);
      dragRef.current = {
        source: "pointer",
        panelId: panel.id,
        pointerId: event.pointerId,
        startX: point.x,
        startY: point.y,
        moved: false,
      };

      const localGrab = getLocalPoint(point, panel);
      updatePanelPull(panel.id, {
        progress: 0.08,
        pullX: 0,
        pullY: 0,
        grabX: localGrab.x,
        grabY: localGrab.y,
        angle: 0,
        curl: 0.08,
        trailAngle: 180,
        tailX: localGrab.x,
        tailY: localGrab.y,
        seamX: localGrab.x,
        seamY: localGrab.y,
      });
      setStatusText("Drag the curtain");
    },
    [updatePanelPull],
  );

  const updatePointerReveal = useCallback(
    (event, panelId) => {
      const drag = dragRef.current;
      if (!drag || drag.source !== "pointer" || drag.panelId !== panelId) return;
      if (drag.pointerId !== event.pointerId) return;

      const point = getStagePoint(event, stageRef.current);
      const panel = panelsRef.current.find((item) => item.id === panelId);
      if (!panel || panel.revealed) return;

      const movement = Math.hypot(point.x - drag.startX, point.y - drag.startY);
      const metrics = getPullMetrics(panel, drag, point);

      drag.moved = drag.moved || movement > 1.8;
      updatePanelPull(panelId, metrics);

      if (metrics.progress >= 0.96) {
        releasePanel(panelId, true);
        dragRef.current = null;
        setStatusText("Panel revealed");
      }
    },
    [releasePanel, updatePanelPull],
  );

  const endPointerReveal = useCallback(
    (event, panelId) => {
      const drag = dragRef.current;
      if (!drag || drag.source !== "pointer" || drag.panelId !== panelId) return;

      event.currentTarget.releasePointerCapture?.(event.pointerId);

      const panel = panelsRef.current.find((item) => item.id === panelId);
      if (!panel || panel.revealed) {
        dragRef.current = null;
        return;
      }

      if (panel.progress > 0.54) {
        releasePanel(panelId, true);
        setStatusText("Panel revealed");
      } else {
        releasePanel(panelId, false);
        setStatusText(cameraState === "ready" ? "System active" : "System idle");
      }

      dragRef.current = null;
    },
    [cameraState, releasePanel],
  );

  useEffect(() => {
    if (showFinal || panels.length === 0 || revealedCount < panels.length) return undefined;

    const timer = window.setTimeout(() => {
      dragRef.current = null;

      if (roundIndex < ROUND_COUNT - 1) {
        const nextRound = roundIndex + 1;

        setRoundIndex(nextRound);
        syncPanels(createPanels(nextRound));
        setStatusText(`Round ${nextRound + 1}`);
        return;
      }

      setShowFinal(true);
      setStatusText("Final revealed");
    }, roundIndex < ROUND_COUNT - 1 ? 950 : 720);

    return () => window.clearTimeout(timer);
  }, [panels.length, revealedCount, roundIndex, showFinal, syncPanels]);

  useEffect(() => {
    return () => {
      cancelAnimationFrame(rafRef.current);
      streamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  const isStarting = cameraState === "starting" || modelState === "loading";

  return (
    <main className="app-shell">
      <section className="workbench" aria-label="gesture curtain stage">
        <div ref={stageRef} className="stage">
          <video
            ref={videoRef}
            className="camera-feed"
            style={{
              transform: `scaleX(${mirrored ? -1 : 1}) scale(${zoom})`,
            }}
            playsInline
            muted
          />

          {cameraState !== "ready" && (
            <div className="camera-placeholder">
              <div className={isStarting ? "loader-dot is-loading" : "loader-dot"} />
              <p>{statusText}</p>
              {statusHint && <small>{statusHint}</small>}
            </div>
          )}

          <div className={`panel-layer ${showFinal ? "has-final" : ""}`}>
            {panels.map((panel) => {
              const maskId = `curtain-mask-${panel.id}`;
              const clipId = `curtain-clip-${panel.id}`;
              const sheetGradientId = `curtain-sheet-gradient-${panel.id}`;
              const shineGradientId = `curtain-shine-gradient-${panel.id}`;
              const weavePatternId = `curtain-weave-pattern-${panel.id}`;
              const silk = getCurtainGeometry(panel);
              const isHorizontalPull = Math.abs(panel.pullX) >= Math.abs(panel.pullY);

              return (
                <article
                  key={panel.id}
                  className={`reveal-panel ${isHorizontalPull ? "pull-horizontal" : "pull-vertical"} ${
                    panel.progress > 0 ? "is-pulled" : ""
                  } ${panel.revealed ? "is-revealed" : ""
                  }`}
                  style={{
                    left: `${panel.left}%`,
                    top: `${panel.top}%`,
                    width: `${panel.width}%`,
                    height: `${panel.height}%`,
                    "--progress": panel.progress,
                    "--pull-x": panel.pullX,
                    "--pull-y": panel.pullY,
                    "--grab-x": panel.grabX,
                    "--grab-y": panel.grabY,
                    "--angle": panel.angle,
                    "--curl": panel.curl,
                    "--trail-angle": panel.trailAngle,
                    "--tail-x": panel.tailX,
                    "--tail-y": panel.tailY,
                    "--seam-x": panel.seamX,
                    "--seam-y": panel.seamY,
                  }}
                  onPointerDown={(event) => beginPointerReveal(event, panel)}
                  onPointerMove={(event) => updatePointerReveal(event, panel.id)}
                  onPointerUp={(event) => endPointerReveal(event, panel.id)}
                  onPointerCancel={(event) => endPointerReveal(event, panel.id)}
                >
                  <svg
                    className="image-reveal-mask"
                    viewBox="0 0 100 100"
                    preserveAspectRatio="none"
                    aria-hidden="true"
                  >
                    <defs>
                      <mask
                        id={maskId}
                        maskUnits="userSpaceOnUse"
                        x="-180"
                        y="-180"
                        width="460"
                        height="460"
                      >
                        <rect x="-180" y="-180" width="460" height="460" fill="white" />
                        <path d={silk.bodyPath} fill="black" />
                      </mask>
                    </defs>
                    <image
                      className="revealed-image"
                      href={panel.image}
                      x="0"
                      y="0"
                      width="100"
                      height="100"
                      preserveAspectRatio="xMidYMid meet"
                      mask={`url(#${maskId})`}
                    />
                  </svg>
                  <svg
                    className="curtain-svg"
                    viewBox="0 0 100 100"
                    preserveAspectRatio="none"
                    aria-hidden="true"
                    style={{
                      "--progress": panel.progress,
                      "--pull-x": panel.pullX,
                      "--pull-y": panel.pullY,
                      "--grab-x": panel.grabX,
                      "--grab-y": panel.grabY,
                      "--angle": panel.angle,
                      "--curl": panel.curl,
                      "--trail-angle": panel.trailAngle,
                      "--tail-x": panel.tailX,
                      "--tail-y": panel.tailY,
                      "--seam-x": panel.seamX,
                      "--seam-y": panel.seamY,
                      "--curtain-opacity": Math.max(
                        0,
                        panel.revealed ? 0 : 0.78 - panel.progress * 0.14,
                      ),
                    }}
                  >
                    <defs>
                      <linearGradient
                        id={sheetGradientId}
                        gradientUnits="userSpaceOnUse"
                        x1={silk.gradient.x1}
                        y1={silk.gradient.y1}
                        x2={silk.gradient.x2}
                        y2={silk.gradient.y2}
                      >
                        <stop offset="0%" stopColor="#ffffff" stopOpacity="0.36" />
                        <stop offset="34%" stopColor="#eefdf8" stopOpacity="0.18" />
                        <stop offset="68%" stopColor="#b5e8d9" stopOpacity="0.09" />
                        <stop offset="100%" stopColor="#ffffff" stopOpacity="0.28" />
                      </linearGradient>
                      <linearGradient
                        id={shineGradientId}
                        gradientUnits="userSpaceOnUse"
                        x1={silk.gradient.x2}
                        y1={silk.gradient.y1}
                        x2={silk.gradient.x1}
                        y2={silk.gradient.y2}
                      >
                        <stop offset="0%" stopColor="#ffffff" stopOpacity="0" />
                        <stop offset="45%" stopColor="#ffffff" stopOpacity="0.5" />
                        <stop offset="100%" stopColor="#ffffff" stopOpacity="0" />
                      </linearGradient>
                      <pattern
                        id={weavePatternId}
                        width="9"
                        height="9"
                        patternUnits="userSpaceOnUse"
                      >
                        <path d="M 0 1 H 9 M 1 0 V 9" stroke="#ffffff" strokeOpacity="0.24" strokeWidth="0.45" />
                        <path d="M 0 5 H 9" stroke="#aee7d6" strokeOpacity="0.12" strokeWidth="0.5" />
                      </pattern>
                      <clipPath id={clipId}>
                        <path d={silk.bodyPath} />
                      </clipPath>
                    </defs>

                    <path className="curtain-shadow" d={silk.bodyPath} />
                    <path
                      className="curtain-sheet"
                      d={silk.bodyPath}
                      fill={`url(#${sheetGradientId})`}
                    />
                    <g clipPath={`url(#${clipId})`}>
                      <path
                        className="curtain-weave-svg"
                        d={silk.bodyPath}
                        fill={`url(#${weavePatternId})`}
                      />
                      <path
                        className="curtain-shine"
                        d={silk.bodyPath}
                        fill={`url(#${shineGradientId})`}
                      />
                      {silk.foldPaths.map((path, index) => (
                        <path
                          key={path}
                          className="curtain-fold-svg"
                          data-fold={index}
                          d={path}
                        />
                      ))}
                    </g>
                    <path className="curtain-live-edge" d={silk.edgePath} />
                    {silk.threadPaths.map((path, index) => (
                      <path
                        key={path}
                        className="curtain-thread-svg"
                        data-thread={index}
                        d={path}
                      />
                    ))}
                  </svg>
                  <div className="panel-frame" />
                </article>
              );
            })}
          </div>

          {showFinal && (
            <div className="final-reveal" aria-label="final photo">
              <div className="final-bloom">
                <div className="final-fireworks" aria-hidden="true">
                  {FINAL_FIREWORKS.map((firework, index) => (
                    <span
                      className="firework-burst"
                      key={`${firework.x}-${firework.y}`}
                      style={{
                        "--burst-x": `${firework.x}%`,
                        "--burst-y": `${firework.y}%`,
                        "--burst-delay": `${firework.delay}ms`,
                        "--burst-color": firework.color,
                      }}
                    >
                      {Array.from({ length: FIREWORK_SPARKS }, (_, sparkIndex) => (
                        <i
                          key={sparkIndex}
                          style={{
                            "--spark-angle": `${(360 / FIREWORK_SPARKS) * sparkIndex}deg`,
                            "--spark-distance": `${44 + ((sparkIndex + index) % 3) * 10}px`,
                          }}
                        />
                      ))}
                    </span>
                  ))}
                </div>
                <div className="final-petals" aria-hidden="true">
                  {Array.from({ length: 14 }, (_, index) => (
                    <span
                      key={index}
                      style={{
                        "--petal-angle": `${(360 / 14) * index}deg`,
                        "--petal-delay": `${index * 34}ms`,
                      }}
                    />
                  ))}
                </div>
                <img src={FINAL_IMAGE} alt="" />
              </div>
            </div>
          )}

          {activeHand && (
            <div
              className="hand-cursor"
              style={{
                left: `${activeHand.x}%`,
                top: `${activeHand.y}%`,
              }}
            >
              <span />
            </div>
          )}

          <div className="hud top-left">
            <strong>{statusText}</strong>
          </div>

          <div className="hud top-right">
            <strong>
              {showFinal
                ? "Final"
                : `Round ${roundIndex + 1}/${ROUND_COUNT} ${revealedCount}/${panels.length}`}
            </strong>
          </div>

          {activeHand && (
            <div className="gesture-chip">
              <span>{activeHand.gesture}</span>
            </div>
          )}
        </div>

        <aside className="control-rail" aria-label="controls">
          <button
            type="button"
            className="primary-action"
            onClick={cameraState === "ready" ? stopCamera : startCamera}
            disabled={isStarting}
          >
            {cameraState === "ready" ? "Stop" : isStarting ? "Loading" : "Start"}
          </button>

          <button type="button" onClick={resetPanels}>
            Reset
          </button>

          <button type="button" onClick={() => setMirrored((value) => !value)}>
            Mirror
          </button>

          <label className="zoom-control">
            <span>Zoom</span>
            <input
              type="range"
              min="1"
              max="1.35"
              step="0.01"
              value={zoom}
              onChange={(event) => setZoom(Number(event.target.value))}
            />
          </label>
        </aside>
      </section>
    </main>
  );
}
