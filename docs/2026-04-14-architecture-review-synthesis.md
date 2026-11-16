# Architecture Review — Team Synthesis

**Date:** 2026-04-14
**Status:** Team research complete; synthesis awaiting user decisions. **Corrected 2026-11-15:** several claims in §3 and §4.1 about WASB and TrackNetV5 did not hold up; see [§13](#13-corrections-2026-11-15).
**Scope:** Full-product architecture (all 7 slices), not just slice 1
**Related:** [2026-04-14-on-device-ball-tracker-ios-design.md](superpowers/specs/2026-04-14-on-device-ball-tracker-ios-design.md) — slice-1 design spec. Tier-1 changes in §4 should be folded into the spec in a follow-up step.

---

## 1. Purpose

This document captures the output of a 4-person agent research team (`tennis-architecture`) that investigated the question: **"What is the best possible architecture for a SwingVision-like tennis iPhone app with ball tracking, line detection, and automated scoring?"**

The team's mandate was to pressure-test the current slice-1 spec and expand to a full-product architecture view (slices 2–7). Each researcher owned one domain, with explicit cross-dependencies forcing them to DM each other via team messaging. They had access to WebSearch, WebFetch, and context7 for coremltools documentation.

This doc is the synthesis: consensus findings, recommended spec changes, and open strategic decisions. It does **not** itself modify the slice-1 spec — that happens in a follow-up step once the user has reviewed these recommendations.

---

## 2. Team composition

| Teammate | Domain |
|---|---|
| `platform-architect` | iOS surface: Core ML internals, runtime alternatives, frame pipeline, thermals, app shell |
| `ml-architect` | Full model stack: ball / court / bounce / shot / player — plus shared-backbone strategy |
| `scoring-architect` | Automated scoring pipeline: rally segmentation, serve detection, state machine, UX |
| `competitive-analyst` | Market reality: what SwingVision, In/Out, PB Vision, Mojjo, Wingfield, Hawk-Eye actually ship |

All four delivered final verdicts. During cross-team coordination the team produced material corrections to their own earlier reports — those corrections are reflected in this synthesis.

---

## 3. Consensus verdict (TL;DR)

**The slice-1 spec is ~80% right directionally.** The core architectural bets — on-device inference, iOS-only, Core ML + Apple Neural Engine, multifunction `.mlpackage` with shared backbone — all survive adversarial review. **Execution risk, not direction risk, is what matters.**

A dozen concrete changes are indicated, split into three tiers (§4–§6). The three most important findings are:

1. **Ball head: replace TrackNet V2/V5 with WASB-HRNet** (§4.1). Ball tracking has moved past U-Net; WASB is 2024 SOTA at 5× fewer params. *(Corrected 2026-11-15: not settled; WASB is one of three on-device candidates. See §13.)*
2. **Full-auto scoring is not achievable at >99%** with current SOTA (§4.7). The product shape must be "AI-assisted scoring with tap-to-correct," not silent live scoreboard. This matches SwingVision's actual pattern: their auto-scoring is **post-session, Apple-Watch-gated, and user-anchored**, not real-time.
3. **SwingVision has a granted patent (US11893808B2) on learned 2D→3D property extraction** (§4.8). Our planned 2D-heatmap-ball + classical-homography path is architecturally outside its claim scope — the patent actually protects their specific approach, not ours. But Slice 3 (line calling) still needs a freedom-to-operate review before shipping, and there's a separate Infinity Cube patent suit (3:22-cv-547) in the same space.

---

## 4. Tier 1 — Clear wins (fold into slice-1 spec)

These changes have no meaningful downside and strong evidence behind them.

### 4.1 Replace ball head: TrackNet V2/V5 → WASB-HRNet

> **Corrected 2026-11-15.** Four statements below are wrong: the expansion of "WASB", the parameter count, the comparison against TrackNetV3, and the licence of the weights. The recommendation is downgraded from "replace" to "include in the bake-off". Details in [§13](#13-corrections-2026-11-15).

- **What:** WASB (Weakly-supervised Active Sampling Ball tracker) at arxiv 2311.05237 (BMVC 2023). HRNet backbone, heatmap output, 3-frame input, ~3M params. Beats TrackNetV3 across all 5 sports in the paper's benchmark, including tennis. MIT-licensed tennis weights are already public in the WASB model zoo.
- **Why:** TrackNetV5 (arxiv 2512.02789, Dec 2025) has 14.77M params / 117 GFLOPs — too heavy once ANE is shared with court, bounce, shot, and player heads. WASB is ~5× smaller at higher accuracy.
- **Slice 1a impact:** bake-off becomes **V2 vs WASB**, not V2 vs V5. Measure the same p95/ANE-utilization/thermal targets.
- **Open risk:** HRNet has many branches/fusion ops — does `coremltools` convert them to ANE-resident kernels, or fall back to CPU? Must be measured in Slice 1a.
- **Source:** ml-architect verdict, confirmed via Papers-with-Code rankings for tennis ball tracking 2024.

### 4.2 Use Apple Vision `VNDetectHumanBodyPose3D` for shot classification
- **What:** Apple's Vision framework ships a 3D human-pose detector (17 joints) starting iOS 17. It's mostly ANE-resident with some GPU fallback. Free, works on A12+.
- **Why:** Shot classification (FH/BH/serve/volley) is a pose-sequence problem — the signal is in wrist/elbow/racket geometry. Feeding a 1.5s window of joints into a small temporal head (MViTv2-S-tiny or 1D-conv-over-time, ~3M params) beats raw-video models at 30× lower FLOPs.
- **Scheduling note:** pose is not free on ANE time. Schedule at ~10 Hz (not every frame) or only when a shot is imminent, so it doesn't serialize against the multifunction model.
- **Source:** ml-architect, confirmed scheduling detail by platform-architect.

### 4.3 Player detection is the 5th function in the multifunction `.mlpackage`
- **What:** Add a small YOLO-style detection head to the shared backbone (instead of shipping a separate player-detection model). ByteTrack for cross-frame association runs on CPU — no model weights, no ANE cost.
- **Why:** Same argument as §R2 in the slice-1 spec — the Neural Engine serializes distinct MLModel instances, so sharing a backbone across all heads is mandatory for 30fps budget.
- **Source:** ml-architect.

### 4.4 Shared-backbone choice: MobileOne-S4 or FastViT-MA24
- **What:** Both are Apple-authored architectures (CVPR 2023 and ICCV 2023 respectively). Proven to run ANE-resident. Sub-millisecond latency on A12+.
- **Why:** Spec currently leaves the backbone unspecified, saying "shared backbone." Picking an Apple-designed backbone maximizes the chance that `coremltools` conversion produces a clean ANE-resident graph.
- **Quantization:** 6-bit palette FP16 is the team's converged recommendation. Keeps weights ~35–40 MB across all heads while staying ANE-compatible.
- **Source:** ml-architect + platform-architect.

### 4.5 Clarify court-detection method vs citation
- **What:** Replace spec's citation of "PnLCalib for court keypoint + line-refinement" with "PnLCalib *method* (keypoint heatmaps + DLT + line-refinement), trained on tennis data from yastrebksv/TennisCourtDetector + amateur-court techniques from arxiv 2404.06977."
- **Why:** PnLCalib (arxiv 2404.08401, CVPR 2024) is trained on *soccer* data; the method transfers to tennis, but the weights don't. Spec currently implies drop-in use.
- **Source:** ml-architect. Spec §9 R3 needs the clarification.

### 4.6 Add audio pipeline to slice 1's architecture
- **What:** Wire `CMSampleBuffer` audio alongside video from `AVCaptureSession`. Do not add any audio models to slice 1 — just have the pipeline capture and persist the audio track so later slices can use it.
- **Why:** Rally segmentation and keyword spotting ("out," "let," "fault") need audio signals. SwingVision's scoring complaints map directly to missed acoustic signals, and their Max tier's "Audio Line Calls" feature (added in the April 2026 pricing refresh) confirms audio is a signal they now use. Retrofitting audio capture after slice 1 ships means migration pain for every persisted session.
- **Cost:** Very low — AVCapture delivers audio natively.
- **Source:** scoring-architect, confirmed by platform-architect.

### 4.7 Split slice 6 into 6a (post-session auto-score + correction UX) and 6b (live scoreboard overlay)
- **What:** Slice 6 was "rally segmentation + auto-scoring" as a single ambiguous bucket. Split it:
  - **6a (must-have):** post-session auto-scoring with a correction UX. Match setup screen before capture (format, server, players). Post-session scrubbable timeline with tap-to-edit per point. Audio keyword spotter via `SFSpeechRecognizer` on-device.
  - **6b (stretch-stretch):** live scoreboard overlay during capture. Silent by default — the current assumption that live scoring is required was implicit and wrong.
- **Why (math):** scoring-architect's error budget shows full-auto scoring needs ≥99.45% per-point correctness to expect ≤1 error in a 180-point match. Current SOTA (rally seg 98% × winner classifier 97% × serve/fault 99%) = 94.1% per-point = **~10.6 errors per match**, which matches SwingVision's top-complaint user reviews precisely.
- **Why (market):** SwingVision's own docs (`swingvision.zendesk.com/hc/en-us/articles/23556754246427`) state: *"Auto-Scoring does not keep score for you in real time."* Their scoring is post-session, Apple-Watch-gated (Game-by-Game mode), and anchored on user confirmation of the first point and match winner. **Live auto-scoring is an open market opportunity nobody has confirmed shipping.**
- **Product-shape change:** the product is **AI-assisted scoring with tap-to-correct**, not silent full-auto. Success criterion becomes "≤5 taps to correct a full match, ≥70% of points auto-classified correctly."
- **Source:** scoring-architect, confirmed and corrected by competitive-analyst.

### 4.8 Patent risks — add as new spec section
Two patents to flag:

**SwingVision's own patent: US11893808B2 — "Learning-based 3D property extraction"**
- Filed Nov 30, 2020; granted Feb 6, 2024; assigned to SwingVision Inc.
- Inventors: Sahai, Hsu, Balamurugan, Ramachandran. Sahai is ex-Tesla Autopilot (3D object tracking).
- **What it claims:** a neural network that extracts 3D properties (location, orientation, velocity) from 2D images, trained against calibrated radar/Lidar/multi-cam ground truth, including "out-of-bounds" determinations. Explicitly sport-generic (tennis, pickleball, badminton, squash, soccer, etc.).
- **Our architectural position:** our planned path is **2D pixel heatmap (WASB) + classical homography (cv2.findHomography) + 2D bounce classifier**. No learned 2D→3D lift. This puts us outside the claim scope.
- **Slice 3 hedge (architectural):** keep the 2D→court-meter translation as **classical homography** (no neural net). Bounce classifier outputs 2D pixel coords only; court-meter translation is a pure post-processing step.
- **Slice 3 hedge (product):** do NOT market this as a "line call system." Frame as "ball landing visualization" or "bounce overlay." The patent's framing is automated-scoring / line-calling; our product framing should be descriptive of what the stack actually does.
- **Required:** freedom-to-operate legal review before Slice 3 ships. Non-negotiable.

**Infinity Cube v. Mangolytics (SDCA 3:22-cv-547)**
- Separate patent-infringement suit against SwingVision (pre-renaming from Mangolytics) over electronic line calling. Infinity Cube's eyes3 product is the only ITF-PAT-approved portable electronic line-calling system. Specific patent number not yet located by the team.
- **Implication:** line-calling space has at least one active patent dispute. Our IP review should enumerate both.

### 4.9 App bundling strategy — On-Demand Resources
- **What:** Commit to Apple's [On-Demand Resources](https://developer.apple.com/documentation/foundation/on_demand_resources) from slice 1b.
- **Why:** SwingVision ships a 876.9 MB bundle while enforcing a 200 MB OTA cellular cap on initial install — they almost certainly use ODR to fetch most of the bundle on first Wi-Fi. Our v1 model stack is ~35-40 MB, so our initial install comfortably fits under 200 MB, but the ODR infrastructure matters for slices 4+ when per-tier model variants and tutorial content arrive. Cheap to wire in slice 1b.
- **Split:**
  - **Base (cellular install ≤200 MB):** app code, primary multifunction `Detector.mlpackage` (ball head), minimum UI, capture+playback paths.
  - **ODR (first Wi-Fi):** onboarding tutorial videos, demo matches, optional advanced models, slice-specific heads not needed for slice 1.
- **Source:** platform-architect, prompted by competitive-analyst's App Store findings.

### 4.10 Drop stateful Core ML from slice 1 critical path
- **What:** Keep `MLState` as an optional slice-1a investigation, but don't design slice 1b around it.
- **Why:** `MLState` ANE residency is undocumented. Apple's WWDC24 session 10161 demonstrated large speedups for LLM KV-cache (1.6× on M3 Max / Mistral-7B), but for a 9-channel frame ring buffer the win is marginal (~1-2 ms of memcpy saved per frame) and the downside risk — silent CPU/GPU fallback in a real-time loop — is unacceptable.
- **Source:** platform-architect.

### 4.11 Runtime alternatives disqualified (no spec change needed — confirms existing direction)
- **MLX Swift:** no Neural Engine path (Metal GPU only); would cook the phone at 30fps sustained. No shipping iOS apps use it for production CV as of April 2026.
- **ExecuTorch / ONNX Runtime Mobile:** no ANE path; Core ML is 5–10× more power-efficient per TOPS for sustained 30fps workloads.
- **Conclusion:** direct Core ML + multifunction `.mlpackage` remains the only right runtime.
- **Source:** platform-architect.

---

## 5. Tier 2 — Strategic, open decisions (for user)

These require a judgment call that the team flagged but did not resolve unilaterally.

### 5.1 Minimum device: A16 (iPhone 15) or A17 Pro (iPhone 15 Pro)?
- **platform-architect argues A17 Pro:** A16 is marginal on sustained thermals (no vapor chamber, 81% vs 94% throughput retention at 30min per Argmax benchmarks). A16 also lacks native INT8-INT8 compute path — INT8 weights on A16 still use FP16 activations, losing effective TOPS.
- **competitive-analyst counterpoint:** SwingVision ships on **A12 Bionic** minimum (iPhone XS, 2018). Our A16 floor is already 5 chip generations more conservative. SwingVision almost certainly **gates live inference by device tier** (post-capture on older hardware, live overlay on newer).
- **Reconciled rationale:** our A16 floor is a **live-overlay-with-multifunction** requirement, not a ball-tracking-in-general requirement. Explicit language for §3:
  > *iPhone 15 / A16+ required for live overlay at 30fps with multifunction inference. Older devices (A12+) may be supported in slice 7+ for post-capture processing, matching SwingVision's pattern. Slice 1 deliberately targets only devices that can run the live path end-to-end.*
- **Decision needed:** keep A16 as floor + run Slice 1a thermal test, OR pre-emptively bump to A17 Pro. Recommendation: keep A16, add a "lite mode" throttle fallback that reduces to ball-head-only when `ProcessInfo.thermalState` rises.

### 5.2 Apple Watch integration — TWO product-critical use cases
The competitive analysis produced two independent reasons Apple Watch matters:

1. **Auto-scoring anchoring (scoring-architect, slice 6a):** SwingVision's auto-scoring explicitly requires an Apple Watch in "Game by Game" mode during the match. The Watch provides shot-detection signals that anchor rally segmentation. Without a Watch we lose a major input.
2. **Shot-type classification (competitive-analyst, slice 5):** SwingVision uses Apple Watch CoreMotion (pronation, wrist orientation) for shot type — NOT vision pose. The corollary: **shipping pose-based shot classification without requiring a Watch is a differentiator**. Our users wouldn't need a Watch for shot-type accuracy; SwingVision's do.
- **Decision needed:** treat Watch as an *optional enhancement* (improves scoring accuracy when paired) vs a *hard dependency* (gates scoring feature entirely). Recommendation: optional enhancement; the pose-based differentiation (case 2) matters more than matching SwingVision's Watch requirement (case 1).

### 5.3 Promote Slice 5 (shot classification) from "future" to v1 or v1.5
- **Why:** Apple Vision pose is free compute. SwingVision does not ship vision-based pose — their shot classification is Watch-only. Ours can ship without a Watch requirement.
- **Cost:** ~6 MB shot classifier on top of the multifunction model; ~10 Hz pose scheduling so it doesn't starve the ball head on ANE.
- **Confirmed gap:** arxiv 2511.04126 (Nov 2025 SwingVision-clone paper) explicitly lists pose as "future work."
- **Decision needed:** accept the slice reordering. Slice 5 becomes either part of v1 or the first release after v1, not a distant slice.

### 5.4 Imported-video post-processing path?
- **What:** SwingVision accepts GoPro / Android video uploads for post-hoc analysis (no live line calls on imported video, but stats + highlights work).
- **Cost:** moderate — a second input pipeline (video file → frame reader → same inference pipeline as live) and a new UX surface.
- **Benefit:** expands TAM to users who own better cameras than their iPhone, and to users with legacy match recordings.
- **Decision needed:** defer to a dedicated slice between 6 and 7 OR include in v1. Not blocking.

### 5.5 Data moat plan
- **SwingVision has 5+ years of Encord-annotated match data** plus (per US11893808B2) a training rig of **radar + Lidar + multi-camera ground truth** at calibrated events. This is a cumulative, physical-capex moat.
- **Consequence:** our out-of-the-box accuracy ceiling will be below SwingVision's, especially on amateur courts / unusual lighting / non-standard camera angles. Replicating their training rig is not feasible for us.
- **Options:**
  - (a) User-contributed data pipeline — opt-in "help improve the AI" that uploads anonymized frames with user-corrected labels.
  - (b) Strategic partnership with a tennis facility / USTA section for training data.
  - (c) Self-supervised / weakly-supervised techniques (research, not guaranteed).
  - (d) Focus on a sport where SwingVision is weak (badminton, or pickleball where they're "mid 80s" accurate).
  - (e) Differentiate on features that don't depend on 3D ground truth (pose, stroke biomechanics, coaching-focused analysis) rather than line calls.
- **Decision needed:** product strategy discussion. Architecture needs to *allow* (a) — an anonymization + upload-consent pipeline is design work that affects slice 1's persistence layer.

---

## 6. Tier 3 — Rejected / out of scope

Decisions the team made to *not* do something.

- **MLX Swift runtime** — disqualified (§4.11).
- **ExecuTorch / ONNX Runtime Mobile** — disqualified (§4.11).
- **Doubles support in v1** — scoring-architect recommends singles-only for v1; doubles in v2. Doubles rally scoring tracks "which side served," which adds another state-machine dimension.
- **Full-auto live scoring with no correction UX** — mathematically infeasible (§4.7).
- **Segmentation + Hough for court detection** — brittle on amateur courts with shadows/occlusion per ml-architect.
- **SAM2 / DINOv3 / diffusion-based trackers for the ball head** — explicit paper findings that these are weak on small objects.
- **Per-frame shot classification** — event-triggered (only when racket contact detected) is cheaper and equally accurate.
- **Marketing the product as "line call system"** — patent-claim-proximity risk (§4.8). Use "ball landing visualization" or similar.
- **Matching SwingVision's Apple Watch hard dependency for shot classification** — we can do better by using free Apple Vision pose (§5.2).

---

## 7. Revised 7-slice product roadmap

Folding Tier-1 changes and pose-promotion (§5.3) into the slice structure:

| # | Slice | Headline changes from pre-review plan |
|---|---|---|
| 1 | On-device ball tracker + iOS app shell | Ball head is WASB not TrackNet; audio pipeline captured from day 1; multifunction `.mlpackage` with dummy second head reserved; ODR wired in |
| 2 | Court keypoint detection + homography | Use PnLCalib *method* trained on tennis; line-refinement step added explicitly |
| 3 | Bounce visualization (NOT marketed as "line calls") | Visual classifier (9-frame stack at velocity peak) + classical homography + 2D bounce classifier; FTO review gate before ship |
| 4 | Player detection + pose | YOLO head on shared backbone (not separate model); Apple Vision `VNDetectHumanBodyPose3D` for pose at ~10 Hz |
| **5** | **Shot classification — PROMOTED to v1/v1.5** | Pose-sequence temporal head (~3-6 MB); **no Apple Watch required** — this is a differentiator vs SwingVision |
| **6a** | **Post-session auto-scoring + correction UX** | Match setup screen, rally segmentation, serve detection, state machine, tap-to-edit UX, audio keyword spotter |
| **6b** | **Live scoreboard overlay (stretch)** | Silent by default; enable only when correction-UX confidence is high. **Open market opportunity — nobody ships this today.** |
| 7 | Highlights + stats dashboards + imported-video path (§5.4) | Unchanged; imported-video may slot in here |

**Dependency reordering:** slice 6a needs slices 2 + 3 + 4 (court, bounce, player-ID) but does **not** need slice 5 (shot classifier is for stats, not scoring). 6a can ship before 5 if we prioritize scoring over shot stats.

---

## 8. New risks raised by the review

1. **WASB → Core ML conversion is unverified** — HRNet fusion ops may not lower cleanly to ANE kernels. Measure in Slice 1a.
2. **Multifunction `.mlpackage` ANE-residency across functions is undocumented** by Apple. Measure in Slice 1a.
3. **Apple Vision pose is not free of ANE time** — mostly ANE-resident with some GPU fallback. Schedule at ~10 Hz or only when a shot is imminent.
4. **INT8 QAT for attention layers** (if V5-variant or MViT shot head ships) is a training-side 2–3 week investment.
5. **Patent risk — SwingVision US11893808B2** (§4.8). FTO review needed before Slice 3.
6. **Patent risk — Infinity Cube v. Mangolytics** (§4.8). Separate FTO consideration.
7. **Data moat vs SwingVision** — physical training rig (radar + Lidar + multi-cam) is capex-moat; our accuracy ceiling will lag unless we ship (5.5a-e).
8. **Encord dependency for annotation** — SwingVision annotates with Encord; we have no annotation pipeline. Non-trivial build if we pursue §5.5(a).
9. **SFSpeechRecognizer short-token reliability in noise** — "out," "let," "fault" in outdoor/amateur-court audio is unproven.
10. **Pose model serialization with multifunction model on ANE** — two separate MLModel instances serialize; 10-Hz scheduling mitigates but doesn't eliminate.

---

## 9. Open questions requiring a prototype

1. Does multifunction `.mlpackage` stay ANE-resident across function calls? (Slice 1a)
2. Does WASB-HRNet convert cleanly to an ANE-resident Core ML graph? (Slice 1a)
3. iPhone 15 (A16) 10-minute sustained at 30fps: pass or fail thermal target? (Slice 1a)
4. `MPSImageScale` vs `vImage` for 1920×1080→512×288 downscale on A16 — head-to-head? (Slice 1a)
5. Apple Vision pose at 10 Hz scheduling — does it free enough ANE for the multifunction model? (Slice 5)
6. Bounce classifier feature reuse from ball-head intermediates — authorable in `coremltools` multifunction descriptor? (Slice 3)
7. Multi-task training gradient conflict on shared backbone — empirically stable with `UncertaintyWeighting`? (Slice 2/3/4 training)
8. SFSpeechRecognizer accuracy on outdoor amateur-court audio for "out"/"let"/"fault" custom vocab? (Slice 6a)

---

## 10. Watch list (3- and 6-month re-check)

**3 months:**
- PB Vision on-device roadmap ETA (roadmap.pb.vision)
- WWDC 2026 — multifunction or MLX signals from Apple
- Infinity Cube v. Mangolytics ruling/settlement (PACER 3:22-cv-547)
- Apple Sports app — does Apple ship first-party tennis on Vision Pro?
- SwingVision post-pricing-refresh scorekeeping flows (Max tier's "Audio Line Calls" implementation)

**6 months:**
- Hawk-Eye / HawkAR consumer partnerships
- TrackNetV3/V4 academic refreshes
- Badminton competitive landscape (Smashspeed as incumbent)
- yastrebksv/TrackNet + TennisCourtDetector weights refreshes
- MLX production CV adoption on iOS
- PB Vision partnerships beyond DUPR
- SwingVision's additional patent filings (USPTO inventor search returned 403 during review)

---

## 11. Sources cited by the team

**Platform:**
- Apple "Behind the Design" SwingVision interview
- coremltools multifunction + stateful model docs
- Argmax iPhone 17 on-device inference benchmarks
- WWDC24 session 10161 (stateful models)
- Apple Developer MLComputeUnits
- Apple On-Demand Resources docs

**ML:**
- WASB (arxiv 2311.05237, BMVC 2023)
- BlurBall (arxiv 2509.18387, Sep 2025)
- TrackNetV5 (arxiv 2512.02789, Dec 2025)
- TrackNetV4 (arxiv 2409.14543, ICASSP 2025)
- No Bells Just Whistles / PnLCalib (arxiv 2404.08401, CVPR 2024)
- Amateur Tennis Line Detection (arxiv 2404.06977)
- TTNet (CVPRW 2020)
- MobileOne (Apple CVPR 2023)
- FastViT (Apple ICCV 2023)
- yastrebksv TennisCourtDetector + TennisProject repos

**Scoring:**
- TenniSet, TTNet
- ITF/USTA scoring rules
- SwingVision Zendesk auto-scoring docs (swingvision.zendesk.com/hc/en-us/articles/23556754246427)
- Tennis acoustic pilot studies

**Competitive:**
- Apple Newsroom SwingVision profile (2022)
- SwingVision App Store listing (A12 min, iOS 18, 876.9 MB)
- SwingVision iOS Engineer job posting (AVFoundation, Core ML, SwiftUI, UIKit)
- SwingVision Encord webinar
- **SwingVision patent US11893808B2** (patents.google.com/patent/US11893808B2/en)
- Infinity Cube v. Mangolytics (SDCA 3:22-cv-547, Bloomberg Law)
- PB Vision public roadmap
- ITF PAT approved list
- Baseline Vision, Wingfield, HomeCourt public pages
- pickleball Union SwingVision pickleball accuracy (Feb 2024)
- arxiv 2511.04126 (Nov 2025 SwingVision-clone paper, pose listed as future work)

---

## 12. Next steps

1. **User reviews this synthesis doc.** Decisions needed on Tier-2 items (§5) before spec revision:
   - 5.1 A16 floor vs A17 Pro
   - 5.2 Apple Watch integration policy
   - 5.3 Slice 5 pose promotion to v1 or v1.5
   - 5.4 Imported-video path timing
   - 5.5 Data moat strategy direction
2. **Fold Tier-1 changes into [2026-04-14-on-device-ball-tracker-ios-design.md](superpowers/specs/2026-04-14-on-device-ball-tracker-ios-design.md).** Spec changes should cross-reference this synthesis.
3. **Write or update product roadmap doc** reflecting the revised slice structure (§7) — possibly a separate `2026-04-14-product-roadmap.md`.
4. **Kick off FTO legal review** for line-calling slices before implementation begins (US11893808B2 + Infinity Cube).
5. **Commit slice-1 spec and synthesis together** once user approves Tier-1 changes.
6. **Invoke writing-plans skill** for the implementation plan of the revised slice-1.

---

## 13. Corrections (2026-11-15)

Source: [2026-11-15-pre-training-review.md](2026-11-15-pre-training-review.md). The rest of this document is unchanged from April.

### 13.1 WASB (§3 item 1, §4.1)

| April claim | Correction |
|---|---|
| "WASB (Weakly-supervised Active Sampling Ball tracker)" | WASB stands for **Widely Applicable Strong Baseline**. It does no weak supervision or pseudo-labelling. Its training trick is hard-to-localise sample mining from epoch 20, worth +0.3 to +0.6 F1 |
| "~3M params" | **1.5M** parameters. The count understates cost: the stem runs at full 288x512 resolution with 64 channels, about 28 GMACs per 3-frame pass by estimate, roughly a quarter of V2 |
| "Beats TrackNetV3 across all 5 sports in the paper's benchmark" | The WASB paper never compares against TrackNetV3. In later third-party comparisons WASB beats V2 in four of five and is roughly level with V3, sometimes below it |
| "MIT-licensed tennis weights are already public" | The code is MIT. The model zoo page states no licence for the weights |
| "2024 SOTA", "confirmed via Papers-with-Code rankings" | Not settled. BlurBall (WASB plus an SE block and blur-aware labels) beats it on table tennis; RacketVision's TrackNetV3 with a background frame beats it on tennis precision and localisation error |

**Effect on §4.1:** WASB stays an on-device candidate, but the bake-off should have three entries: V2 with a median background frame, WASB-HRNet, and a width-scaled MBConv U-Net in the style of the NYCU TrackNetV5-Lite (2.58M parameters, weights released). Whether HRNet's branches stay on the Neural Engine after Core ML conversion is still unmeasured (§8 risk 1, §9 question 2). The only device number found is an unreviewed repo reporting 3–5 ms per frame on an A17 Pro.

### 13.2 TrackNetV5 (§4.1, §11)

- The arXiv paper (2512.02789) now has a self-described official SDK under study-only terms, with no weights. Two independent retrains reached F1 .915 and .773, well below the paper's .986, which was measured on a 7:3 split with no held-out test set.
- A second, unrelated paper also named TrackNetV5 (NYCU, ICIP 2026) exists. Citations must say which one.
- The iOS spec's R4 says V5 has a 5-frame temporal context. The arXiv V5 and our implementation use 3 frames; it is the NYCU model that uses 5.

### 13.3 Training project status (§5.5, iOS spec R5)

- The first training run is V2 on tennis, after code fixes listed in the pre-training review §4. V5 is deferred.
- RacketVision was released on 2026-04-11, but its footage rights for commercial use are unresolved, as are the licences of most public ball-tracking datasets. This strengthens §5.5: public data alone may not be usable in a shipped model.
- Pretrained checkpoints for R5: TrackNetV3's repo now states MIT for both code and checkpoints (badminton-trained). RacketVision ships a TrackNetV3 ball-tracking checkpoint trained on its tennis, badminton and table tennis data, in a model repo with no licence file.
