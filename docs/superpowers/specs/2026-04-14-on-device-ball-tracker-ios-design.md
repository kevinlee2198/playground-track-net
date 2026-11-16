# On-Device Ball Tracker — iOS App (Slice 1) — Design Specification

**Date:** 2026-04-14
**Status:** Revised post agent-team review (R1/R2/R3 resolved inline; R4/R5 remain open)
**Parent product:** SwingVision-like racquet-sport analysis app (tennis v1, badminton/pickleball later)
**Preceding spec:** [2026-03-21-tracknet-ball-tracking-design.md](2026-03-21-tracknet-ball-tracking-design.md)

---

## 1. Context

The parent product is a SwingVision-like racquet-sport analysis app. A full SwingVision-equivalent decomposes into ~7 independently shippable slices:

| # | Slice | Status |
|---|---|---|
| **1** | **On-device ball tracker + iOS app shell** | **this spec** |
| 2 | Court keypoint detection + homography | future |
| 3 | Bounce detection + in/out line calls | future |
| 4 | Player detection + pose estimation | future |
| 5 | Shot classification (FH/BH/serve/volley) | future |
| 6 | Rally segmentation + auto-scoring | future |
| 7 | Highlights + stats dashboards | future |

Slice 1 is the architectural foundation every later slice sits on. Its job is to prove TrackNet can be deployed, inference can be driven from a live camera feed on an iPhone within a 30fps budget, and that an app shell exists to host future models. Every later slice plugs into the infrastructure established here.

---

## 2. Goal

Ship an iOS app that:
- Records tennis video from the rear camera at 30fps
- Runs TrackNet on-device in realtime and draws the detected ball's trajectory as an overlay on the live viewfinder
- Persists the recorded video (with metadata) for later review on a history screen
- Establishes the frame pipeline, inference loop, and model-loading infrastructure that slices 2–7 will extend

Explicit **non-goals** for slice 1: court detection, line calls, player detection, pose, shot classification, scoring, highlights, Android support, sports other than tennis, post-match analysis beyond basic video playback, and sharing/export beyond native iOS share sheet.

---

## 3. Locked Decisions

These are outcomes from the brainstorming session that precedes this spec. They are inputs to the design, not open questions.

| Decision | Choice | Rationale |
|---|---|---|
| Platform | iOS only | Core ML + Apple Neural Engine is the most mature on-device ML stack; SwingVision is the on-device precedent (per Apple "Behind the Design" interview: *"not possible without Neural Engine"*). PB Vision is a *partial* precedent — they ship live court validation on-device today but full video processing still runs on "beefy cloud servers" per their public roadmap. |
| Minimum iOS | **iOS 18+** | Required for multifunction `.mlpackage` (see §5 / R2 resolution) and stateful Core ML models. An iPhone 15 (A16) can upgrade to iOS 18. |
| Minimum device | iPhone 15 / A16+ | Adequate Neural Engine headroom (~17 TOPS) *for a single multifunction model*. See §9 R1/R4 for thermal caveats — A17 Pro / A18 Pro vapor-chamber designs have materially better sustained headroom, so a "lite mode" fallback may be needed on A16 if slices 2/3 push utilization over the thermal knee. |
| UX mode | Live overlay during capture | Matches SwingVision's headline UX; the "magical" demo that differentiates from post-upload tools like Mojjo |
| Sport (v1) | Tennis | Best pretrained checkpoint availability; largest consumer market; simplest to film personally |
| Model weights (v1) | Public pretrained TrackNetV2 or V3 tennis checkpoint | Decouples slice 1 from the separate training project; swap-in path later |
| Model architecture | TBD — decided by Slice 1a spike (V2 vs V5) | See §4 |
| Model packaging | **Multifunction `.mlpackage`** with shared backbone and one output function per head (ball heatmap today; court keypoints + bounce classifier added in slices 2/3 without shipping additional model files) | See §5.2 and §9 R2. Forced by ANE serialization of separate MLModel instances. |
| Frame pipeline | AVFoundation → direct Core ML (multifunction) → Core Animation overlay. Vision/VNCoreMLRequest rejected: incompatible with 9-channel input *and* with multifunction inference. | See §5 |
| Concurrency | Camera delegate on serial `DispatchQueue`; inference on a dedicated serial queue or a `@globalActor`-isolated inference worker; state kept in a Swift `actor`. CMSampleBuffer is not Sendable, so we extract CVImageBuffer inside the delegate queue before crossing actor boundaries. | See §5.4 |
| Post-match cloud re-analysis | Non-goal for slice 1, but **allowed in the architecture** for later slices (highlights, stats aggregation, match sharing). Matches SwingVision's actual pattern: live path is 100% on-device; cloud is for storage and sharing, not inference. | |

---

## 4. Architecture Overview

Slice 1 divides into two sub-slices with a decision gate between them:

```
Slice 1a: Core ML Conversion Spike (V2 vs V5 bake-off)
  - Port V2 architecture to Core ML (random weights OK)
  - Port V5 architecture to Core ML (random weights OK)
  - Benchmark both on A16+ hardware
  - Measure: latency, Neural Engine utilization, thermals
  - Decision gate -> pick V2 or V5 for Slice 1b

Slice 1b: iOS App + Live Overlay
  - iOS app shell (SwiftUI + UIKit bridging)
  - AVFoundation camera pipeline
  - Direct Core ML inference with chosen model
  - 3-frame ring buffer + 9-channel MLMultiArray assembly
  - Core Animation trajectory overlay
  - Video + trajectory persistence
  - History / playback screen
```

### 4.1 Slice 1a — Core ML Conversion Spike

**Why:** An earlier assumption was that V5's factorized spatio-temporal attention would fall back to CPU/GPU under Core ML and tank performance. That's a heuristic, not measured truth. Since training is a separate project and both architectures are viable training targets, the deployment decision should drive the training decision — not vice versa.

**Deliverables:**
- `models/export_coreml.py` — conversion script that loads a V2 or V5 PyTorch module, traces via `torch.export` (or TorchScript fallback), emits a `.mlpackage` via `coremltools` targeting `ct.target.iOS18`. Supports both architectures behind a CLI flag.
- `models/export_multifunction.py` — builds a multifunction `.mlpackage` from a backbone + one or more heads using `ct.utils.save_multifunction` and `MultiFunctionDescriptor`. For slice 1a this packs at least a dummy "ball" head and a dummy "keypoint" head to prove the pattern works on ANE before slices 2/3 need it for real.
- `tools/benchmark_ios.swift` — minimal Swift CLI / XCTest target that loads a `.mlpackage`, runs 1000 inferences with synthetic input, and reports p50/p95/p99 latency plus compute-unit utilization.
- **Measurement targets (on A16 or later):**
  - p95 single-head latency ≤ 25ms (leaves 8ms for camera + preprocessing + rendering in the 33ms budget)
  - p95 *multifunction* latency (ball head only) within ~2ms of single-function — proves the multifunction wrapper doesn't add meaningful overhead
  - ≥ 95% of ops execute on Neural Engine across *both* heads — confirms multifunction models stay ANE-resident (not documented by Apple; must be measured)
  - Sustained 10-minute run at 30fps on iPhone 15 (A16, no vapor chamber) with < 15% throughput degradation
- **Decision rule:**
  - If V5 meets all targets AND multifunction works on ANE → ship V5 with multifunction packaging
  - If V5 fails on ANE multifunction only → ship V5 but with single-function (accept serialized future models as technical debt)
  - If V5 fails latency/thermals → ship V2 with multifunction
  - If A16 fails sustained-thermal target regardless of model → revisit §3 minimum-device decision (bump to A17 Pro / iPhone 15 Pro+)

**Optional investigation (time permitting):** stateful Core ML models (iOS 18+) — can the 3-frame ring buffer move *inside* the model as persistent state, so we pass one frame per inference instead of assembling a 9-channel MLMultiArray each call? Expected wins: less CPU work per frame, simpler Swift code. Risks: ANE support for `MLState` is not publicly documented; may fall back to GPU.

**Out of scope for 1a:** real weights, real accuracy measurement, camera integration, UI. Random weights produce valid latency numbers because inference cost depends on architecture, not weight values.

### 4.2 Slice 1b — iOS App Shell + Live Overlay

Built on whichever model won 1a. The rest of this spec describes 1b in detail.

---

## 5. Frame Pipeline Design (Slice 1b Hot Path)

The hardest engineering problem in slice 1b is staying inside the 33ms-per-frame budget at 30fps sustained. This section describes the exact data flow.

### 5.1 Budget breakdown (target)

| Stage | Target budget | Notes |
|---|---|---|
| Camera → `CVPixelBuffer` | 0ms (hardware) | AVFoundation delivers frames asynchronously; this is concurrent with inference |
| Downscale 1080p/4K → 512×288 | ≤ 2ms | via `vImage` or `CIImage` + Metal |
| 3-frame ring buffer + 9-channel `MLMultiArray` assembly | ≤ 2ms | simple memcpy into a pre-allocated buffer |
| Core ML inference | ≤ 25ms | from slice 1a measurement |
| Heatmap → centroid (CPU post-processing) | ≤ 2ms | threshold + connected components (existing code in `inference/postprocess.py` reference) |
| Overlay render (CAShapeLayer update) | ≤ 2ms | a few line segments and a dot |
| **Total** | **≤ 33ms** | fits 30fps budget with headroom |

### 5.2 Pipeline stages

```
AVCaptureSession (.hd1920x1080, 30fps, landscape)
  -> AVCaptureVideoDataOutput (delegate, serial dispatch queue)
  -> CVPixelBuffer (BGRA, 1920x1080)
      |
      +-- forwarded to AVCaptureVideoPreviewLayer (native, zero-copy, direct to GPU)
      |
      +-- resized to 512x288 (vImage -> planar RGB float32)
           -> FrameRingBuffer (capacity 3)
                -> when full: assemble MLMultiArray (1, 9, 288, 512)
                     -> MLModel.prediction(from: featureProvider)
                          -> heatmap MLMultiArray (1, 3, 288, 512) -- one heatmap per input frame
                               -> threshold + centroid (3rd/newest frame's heatmap)
                                    -> TrajectoryStore (ring buffer, N ~ 90 = 3s at 30fps)
                                         -> OverlayLayer (CAShapeLayer) draws trail over preview
```

### 5.3 Why direct Core ML with a multifunction `.mlpackage`, not Vision

Two constraints drive this choice together — Vision rejection alone is not sufficient justification.

**Constraint 1 — multifunction inference:** Core ML `MLComputeUnits` provides only on-device options (`.all`, `.cpuAndGPU`, `.cpuOnly`, `.cpuAndNeuralEngine`). Crucially, **the Neural Engine serializes distinct MLModel instances** — only one model executes on ANE at a time; additional models queue. Slices 2 and 3 add a court keypoint head and a bounce classifier. Shipping them as separate `.mlpackage` files would serialize three inferences per frame and blow the 33ms budget. The only way to run multiple heads efficiently is a single **multifunction `.mlpackage`** (iOS 18+, authored via `ct.utils.save_multifunction`) with a shared backbone and deduplicated weights. Slice 1 must reserve this architecture from day one, or the later slices force a re-architecture.

**Constraint 2 — 9-channel input:** Vision's `VNCoreMLRequest` is designed around single-image models that accept a standard `CVPixelBuffer`. TrackNet's 9-channel concatenated-frame input has no native `CVPixelBuffer` format (OS formats top out at 4 channels). Workarounds:

- **Spatial tiling (3 frames side-by-side as one wide image):** requires retraining the model with a new input layout, incompatible with public pretrained checkpoints.
- **Vision for preprocessing only, direct Core ML for inference:** adds a dependency and hides control of the hot path.

Vision also has no first-class story for multifunction model inference — the `MLModel.prediction(from: featureProvider, options:)` path exposes the `function_name` selector directly; Vision does not. Direct Core ML is therefore the natural fit, preserves the ability to load any community-standard TrackNet checkpoint, and makes the multifunction story clean.

### 5.4 Concurrency model

Swift 6 concurrency is supported but the AVFoundation camera APIs remain callback-based and `CMSampleBuffer` is not `Sendable`. Recommended pattern:

- **Camera delegate:** a serial `DispatchQueue` with QoS `.userInitiated`. `AVCaptureVideoDataOutputSampleBufferDelegate` callbacks fire here. Extract `CVImageBuffer` and frame metadata inside this callback before crossing any actor boundary — do not pass `CMSampleBuffer` onward.
- **Inference worker:** a Swift `actor` (or a dedicated `@globalActor`-isolated type) that owns the multifunction `MLModel`, the 3-frame ring buffer (or its stateful-model equivalent), and the `TrajectoryStore`. The delegate posts `CVImageBuffer + timestamp` into this actor via `Task { await inferenceWorker.ingest(...) }`.
- **Main actor:** owns the overlay layer. Inference worker publishes trajectory updates via `@MainActor` async calls (cheap — just a couple of `CGPoint`s per frame).

This keeps the hot path (camera → inference) off the main thread while still giving us Swift 6's data-race guarantees, and the actor boundary makes state ownership obvious.

### 5.5 Thermal and overflow handling

- **If inference overruns the budget for N consecutive frames** (N = 10 suggested), drop the overlay for the frame (preview continues uninterrupted) and log the event.
- **If `ProcessInfo.thermalState` rises to `.serious` or `.critical`,** downshift inference to every-other-frame (effectively 15fps overlay while preview stays at 30fps). Also reduce the multifunction model to the ball head only (skip court/bounce heads on throttled frames once slices 2/3 land).
- **Cold thermal observation:** published benchmarks show A16-class phones lose material throughput within minutes of entering "hot" thermal state. Slice 1a's 10-minute sustained test on iPhone 15 (A16) is the authoritative check. If A16 fails, the §3 minimum-device decision bumps — most decisively to iPhone 17 Pro, which introduced Apple's vapor-chamber thermal design and is the first iPhone with clearly demonstrated sustained-Neural-Engine headroom in public benchmarks. An intermediate bump to iPhone 15 Pro / A17 Pro may be sufficient but is less certain.

---

## 6. App UX Shape

### 6.1 Screens (v1)

Three screens. SwiftUI for navigation and non-realtime UI; UIKit (`UIViewControllerRepresentable`) for the capture screen because `AVCaptureVideoPreviewLayer` is UIKit-native.

1. **Home / History** — list of past recordings as a grid of thumbnails, tap to open Playback. "Record" button (prominent) → opens Capture.
2. **Capture** — fullscreen camera preview, landscape-locked, with the trajectory overlay and a record/stop button. Shows thermal state indicator (text "cooling" if throttled).
3. **Playback** — native `AVPlayer` view replaying a recorded clip with the trajectory overlay redrawn from stored trajectory metadata.

### 6.2 Persistence

For each recorded session:

- `session-{uuid}/video.mov` — H.264, 1080p30, landscape, as-captured (Apple-standard)
- `session-{uuid}/trajectory.json` — per-frame detections: `[{"frame": int, "timestamp_ms": int, "x": float, "y": float, "confidence": float} | null]`, with `null` for frames where no ball was detected

Metadata is redrawn at playback time rather than re-running inference. Keeps playback cheap and lets future slices (court, bounces) add their own sidecar files without re-inferring the ball track.

### 6.3 Orientation & camera config

- Landscape-locked (a tennis court fills a landscape frame; portrait is never useful for this app).
- Capture at **1920×1080, 30fps, BGRA**. Downscale to 512×288 on the GPU for inference. Store the original 1080p video (users will want to rewatch in quality).
- Autofocus and auto-exposure on (AVCaptureDevice defaults), but lock to center-weighted metering to prevent exposure swings when the ball moves across bright/dark regions.

---

## 7. Module Layout (iOS project)

```
TrackNetApp/
  App/
    TrackNetApp.swift           // @main, app entry
    ContentView.swift           // root navigation
  Capture/
    CaptureView.swift           // SwiftUI wrapper
    CaptureViewController.swift // UIKit view controller, AVCaptureSession owner
    FramePipeline.swift         // downscale, ring buffer, MLMultiArray assembly
    InferenceEngine.swift       // MLModel wrapper, serial queue
    OverlayRenderer.swift       // CAShapeLayer management
    TrajectoryStore.swift       // ring buffer of detections
  Playback/
    PlaybackView.swift
    OverlayReplayer.swift       // redraws trajectory from JSON
  History/
    HistoryView.swift
    SessionStore.swift          // filesystem-backed list of sessions
  Model/
    Detector.mlpackage          // multifunction Core ML model (ball head today; court + bounce heads added in slices 2 & 3)
    ModelLoader.swift           // MLModel loading with caching, function selection via `MLModelConfiguration.functionName`
```

**Module boundaries:**
- `FramePipeline`, `InferenceEngine`, `OverlayRenderer`, `TrajectoryStore` each have one clear responsibility and can be unit-tested independently with synthetic inputs.
- `CaptureViewController` wires them together; it has no inference logic of its own.
- `Model/Detector.mlpackage` is swappable — replacing with a newer checkpoint requires no code changes as long as input/output shapes match. Adding future heads (court keypoints, bounce classifier) ships as a new `Detector.mlpackage` build with additional functions; the app selects functions at inference time via `MLModelConfiguration.functionName`.

---

## 8. Testing Strategy

### 8.1 Unit tests (XCTest)
- `FramePipelineTests`: feed synthetic `CVPixelBuffer`s, verify 3-frame ring buffer assembly produces the expected `MLMultiArray` layout.
- `OverlayRendererTests`: given a known trajectory, verify path geometry.
- `TrajectoryStoreTests`: ring buffer semantics (capacity, oldest-evicted, thread safety).
- `ModelLoaderTests`: verify the bundled `.mlpackage` loads and accepts the expected input shape.

### 8.2 Integration tests
- **Synthetic video test:** generate a 5-second fake video with a known moving dot; run the full pipeline; assert ≥ 90% of frames produce a detection within N pixels of ground truth.
- **Perf test:** measure sustained p95 latency over 10 minutes on a real device (CI can run the latency portion on a simulator; thermal testing requires a physical device and runs manually before release).

### 8.3 Manual QA checklist
- Record 10-minute tennis session; playback trajectory looks sensible.
- Record indoors and outdoors (different lighting).
- Record starting from cold phone and from already-warm phone.
- Background the app mid-recording; foreground; verify state.
- Low-storage handling.

---

## 9. Risks — Resolved and Open

### R1 — RESOLVED. On-device is correct. Live overlay cannot be cloud-assisted.
5G mobile uplink p99 jitter (not median) disqualifies any cloud-assisted live path — even best-case WebRTC + mobile networks produce frame-by-frame latency far above the 33ms budget, with a long tail that would break the UX on >5% of frames. Cloud gaming (the most aggressively optimized real-time stream-up/render-down workload) achieves 25–45ms only on fiber/Wi-Fi 6. The relevant disqualifier is **uplink variance**, not mean bandwidth.

SwingVision runs 100% on-device per Apple's "Behind the Design" interview. PB Vision has partial on-device (live court validation) but full video processing remains cloud-today per their public roadmap; it's referenced here as a *partial* precedent, not a full one.

**Residual risk:** A16 thermal headroom for sustained multi-head inference is unverified. The Slice 1a 10-minute test on iPhone 15 (A16, no vapor chamber) is the authoritative check; if it fails, §3's minimum-device decision bumps to A17 Pro. The fallback plan (§5.5) reduces inference to every-other-frame when thermal state rises.

### R2 — RESOLVED. Direct Core ML with multifunction `.mlpackage`. iOS 18 minimum.
The single most important finding from the architecture review: **Core ML serializes MLModel instances on the Neural Engine.** The earlier plan to run three separate models (ball + court keypoints + bounce classifier) would have serialized three inferences per frame, busting the 33ms budget. The resolution is a **multifunction `.mlpackage`** (iOS 18+): one model, shared backbone, weights deduplicated, multiple output functions selectable at inference time without reloading. Authored via `coremltools`' `ct.utils.save_multifunction` + `MultiFunctionDescriptor`. Slice 1a proves the pattern end-to-end with a dummy second head before slices 2/3 rely on it.

Vision framework remains rejected (9-channel input mismatch + no first-class multifunction support). `CAShapeLayer` over `AVCaptureVideoPreviewLayer` remains the right overlay mechanism (trivial geometry, bounded to ~90 points). Swift 6 concurrency pattern: serial `DispatchQueue` for delegate + inference worker `actor` for state; extract `CVImageBuffer` inside the delegate before crossing the actor boundary because `CMSampleBuffer` is not `Sendable`.

Stateful Core ML models (iOS 18+) are an optional investigation in Slice 1a — pushing the 3-frame ring buffer into the model as persistent state would simplify the app code, but ANE support for `MLState` is not publicly documented and must be measured.

### R3 — RESOLVED. Keypoint detector + line-refinement; visual bounce classifier + physics fit. Shared backbone via multifunction.
State of the art for tennis court geometry is a keypoint model (14–22 keypoints, heatmap-based) initialized from a yastrebksv/TennisCourtDetector-style head, then refined via PnLCalib-style non-linear optimization against detected court lines (arXiv:2404.08401). Pure segmentation+Hough works on clean pro courts but is brittle on amateur courts with shadows, occlusion, and surface variation. Raw keypoints alone hit ~pixel accuracy; line-refinement is what gets us the sub-pixel homography needed for cm-accurate line calls.

Bounce detection must combine (a) a small visual classifier on a ~9-frame stack centered on a detected velocity peak and (b) a physics parabola fit in court-meter coordinates for sub-frame bounce localization. Trajectory-only peak detection yields ~85% AUC; SwingVision's published 97% @ 10cm accuracy is consistent with this combined approach. At 30fps a ball moving 20 m/s travels ~66cm per frame, so the bounce lands *between* sampled frames — pure velocity-flip gives a frame index, not a cm location. Lines are ~5cm wide; errors inside that band will always be ambiguous, which is why "95%+ on balls ≥5cm from a line, graceful fuzz inside 5cm" is the realistic accuracy target for single-camera 1080p30.

Compute budget: TrackNet (~15–20 GFLOPs/frame × 30fps) + court keypoint head (~10 GFLOPs × 1 Hz) + bounce classifier (<2 GFLOPs, event-triggered) ≈ 500 GOP/s, against A16's ~17 TOPS INT8 (~3% of raw throughput). FLOPs are not the bottleneck; memory bandwidth for U-Net feature maps and thermal dissipation are. The shared-backbone multifunction packaging (R2 resolution) amortizes the expensive backbone pass across all three heads.

### R4 — OPEN. V5 vs V2 beyond latency (for Slice 1a to resolve).
Slice 1a decides on latency + ANE utilization + multifunction compatibility. Non-latency factors that remain open: (i) V5's attention layers and their INT8 quantization behavior (the separate training-project will need a quantization-aware-training pass if INT8 is on the table), (ii) V5's 5-frame temporal context vs V2's 3-frame — more context means more pipeline fragility to dropped frames, (iii) whether V5's factorized spatio-temporal attention stays ANE-resident under Core ML Tools' conversion. The 1a spike measures all three.

**Update 2026-11-15:** the training project has deferred V5. Its arXiv results are unreproduced, and both that model and our implementation use 3 frames, not 5 (the 5-frame model is a different paper also named TrackNetV5, from NYCU). The bake-off candidates are now V2 with a median background frame, WASB-HRNet, and a width-scaled MBConv U-Net. See [the pre-training review](../../2026-11-15-pre-training-review.md) §3 and §5.

### R5 — OPEN. Pretrained checkpoint availability.
We assume a public TrackNetV2 or V3 tennis checkpoint exists and is redistributable. The candidates to verify (and their licenses) are yastrebksv/TennisCourtDetector's linked weights, the original TrackNetV3 repo's checkpoints, and any community forks. This must be confirmed before Slice 1b begins — if no redistributable weights exist, slice 1b blocks on the separate training project delivering first. Flagged as a scheduling risk, not an architectural one.

**Update 2026-11-15:** partly answered. The official TrackNetV3 repo (qaz812345/TrackNetV3) now states that MIT covers its pretrained checkpoints as well as its code, but those are trained on badminton. RacketVision publishes a TrackNetV3 checkpoint trained on tennis, badminton and table tennis, in a model repo with no licence file. WASB's tennis weights are public, with no licence stated for the weights. No tennis checkpoint with clear redistribution terms has been confirmed, and our own training project has not produced weights yet; see [the pre-training review](../../2026-11-15-pre-training-review.md) §4.

---

## 10. Success Criteria for Slice 1

Slice 1 is done when:

1. An iOS app installs on an iPhone 15 (or later) running iOS 18+ from a development build.
2. User can record a 5-minute tennis session outdoors and see the ball's trajectory drawn on the live preview, with visible trail that updates in real time.
3. The shipped `Detector.mlpackage` is authored as a **multifunction** model and exposes at least one non-ball dummy function (e.g., a passthrough "keypoint" head returning zeros), proving the multifunction architecture runs end-to-end on ANE before slices 2/3 rely on it for real.
4. p95 inference latency ≤ 25ms on A16+ during sustained capture when the ball head is invoked.
5. Sustained-10-minute thermal test on iPhone 15 (A16) shows < 15% throughput degradation, OR the spec's minimum-device decision has been formally re-opened (no silent acceptance of throttling).
6. Recorded video + trajectory metadata persist across app restarts and play back correctly.
7. Code passes `ruff`, `ty`, and the full XCTest suite.

---

## 11. Timeline Estimate (rough)

- Slice 1a (Core ML conversion spike + multifunction benchmark + optional stateful spike): **5–7 days** (up from the original 3–5 because the multifunction + optional stateful investigations add scope)
- Slice 1b (iOS app + live overlay on the chosen model): **3–4 weeks**
- R5 verification (pretrained checkpoint license hunt): **1 day, runs in parallel with 1a**

Total to ship slice 1: **~5–6 weeks** of focused work, assuming a redistributable pretrained checkpoint is located and the developer has shipped at least one prior iOS app.
