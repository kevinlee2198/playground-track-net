# Training Data Sources

Last verified: **2026-11-15** (previous check: 2026-04-07). See the [verification log](#verification-log) for what was actually loaded versus inferred.

Label format we consume: one row per frame with frame index, visibility flag, and ball centre x, y in the source video's pixel coordinates.

## Summary

| Sport | Dataset | Labelled frames | Camera | Licence | Use |
|---|---|---|---|---|---|
| Tennis | TrackNet tennis | ~20K | Broadcast | None stated | First training run |
| Badminton | CoachAI shuttlecock (TrackNetV2) | 78,200 | Broadcast + 3 amateur matches | None stated | Core |
| Tennis / badminton / table tennis | RacketVision | 64,042 (of 435,179 frames) | Broadcast | MIT; footage rights unresolved | Pre-training, needs loss mask |
| Table tennis | BlurBall | ~64K | Fixed static cameras | Code MIT; data licence unstated | Pre-training candidate |
| Badminton | ETH one-shot shuttle detection | 20,510 | Stationary, non-broadcast | Unstated (code AGPL-3.0) | Candidate, check frame continuity |
| Pickleball | TrackNet-Pickleball | 12,810 | Single match | None | Labels only; raw video not published |

Pickleball remains the gap: nothing new and downloadable exists, so it has to be filled with our own recordings.

## Tennis — TrackNet dataset (~20K frames)

| Source | Link | Notes |
|--------|------|-------|
| NYCU SharePoint (official) | https://nycu1-my.sharepoint.com/:u:/g/personal/tik_m365_nycu_edu_tw/ETCr6-M0e1VDhGCdMbvljcsBu31AJTO5xa_1cW8pHa7niA | `Dataset.zip`, 2.56 GB, game1..game10. The source WASB-SBDT points to |
| Kaggle | https://www.kaggle.com/datasets/sofuskonglevoll/tracknet-tennis | 2.59 GB, version 2, last updated 2024-03-01. Licence "Unknown" |
| Google Drive (third-party mirror) | https://drive.google.com/drive/folders/11r0RUaQHX7I3ANkaYG4jOxXK1OYo01Ut | `Dataset.zip` 2.39 GB plus `model_best.pt` and a bounce model. Looks like a reimplementation's asset folder, not the original source |
| NYCU GitLab | https://gitlab.nol.cs.nycu.edu.tw/open-source/TrackNetv2 | Code only, no data, no LICENSE, last commit 2021. README links point to a dead host |

**Format** (confirmed on the SharePoint `Dataset.zip`, downloaded 2026-11-15): the zip's top-level folder `Dataset/` holds `game1` … `game10`, each with `ClipN/` folders of `NNNN.jpg` frames and one `Label.csv`, plus a `Readme.docx`. 95 clips, 19,835 frames, every image 1280x720. Every `Label.csv` has the header `file name,visibility,x-coordinate,y-coordinate,status` (example row `0000.jpg,1,599,423,0`), not our `Frame,Visibility,X,Y` layout; `data/labels.py` reads both. Visibility takes the values 0, 1, 2 and 3. Visibility 0 rows have empty coordinates; all 82 visibility 3 rows carry coordinates, and we treat them as ball-absent, as WASB does. Every clip is numbered 0000…n-1 with no gaps and one label row per image. Per-split counts are in the [pre-training review](2026-11-15-pre-training-review.md) §4.4.

**Note:** the TrackNetV6 paper's appendix reports that the publicly redistributed copy of this dataset is degraded and every method scores lower on it than in its original publication.

### Download

```bash
# Official SharePoint. The cookie jar is required; without it curl lands on a login page.
curl -L -c cj -b cj -o Dataset.zip \
  "https://nycu1-my.sharepoint.com/:u:/g/personal/tik_m365_nycu_edu_tw/ETCr6-M0e1VDhGCdMbvljcsBu31AJTO5xa_1cW8pHa7niA?download=1"

# Or Kaggle (API token in ~/.kaggle/kaggle.json)
kaggle datasets download sofuskonglevoll/tracknet-tennis
```

## Badminton — CoachAI shuttlecock dataset (78,200 frames)

| Source | Link |
|--------|------|
| SharePoint (official) | https://nycu1-my.sharepoint.com/:u:/g/personal/tik_m365_nycu_edu_tw/EWisYhAiai9Ju7L-tQp0ykEBZJd9VQkKqsFrjcqqYIDP-g |
| Docs | https://hackmd.io/@TUIK/rJkRW54cU |
| CoachAI GitHub | https://github.com/wywyWang/CoachAI-Projects (does not host the tracking data) |

**Format:** 1280x720, 30 fps, 26 videos. `TrackNetV2.zip` is 1.73 GB and contains `Professional/match1-23`, `Amateur/match1-3`, `Test/match1-3`: 201 rally mp4 files with 201 CSVs (`Frame, Visibility, X, Y`).

**Licence:** none stated.

**Pretrained weights for this data:** [qaz812345/TrackNetV3](https://github.com/qaz812345/TrackNetV3). Its LICENSE was updated 2026-08-25 to say MIT covers both the code and the pretrained checkpoints.

### Download

```bash
curl -L -c cj -b cj -o TrackNetV2.zip \
  "https://nycu1-my.sharepoint.com/:u:/g/personal/tik_m365_nycu_edu_tw/EWisYhAiai9Ju7L-tQp0ykEBZJd9VQkKqsFrjcqqYIDP-g?download=1"
```

## Multi-sport — RacketVision (released 2026-04-11)

| Source | Link |
|--------|------|
| Dataset (Hugging Face, not gated) | https://huggingface.co/datasets/linfeng302/RacketVision |
| Weights (Hugging Face) | https://huggingface.co/linfeng302/RacketVision-Models |
| Code | https://github.com/OrcustD/RacketVision |
| Paper | https://arxiv.org/abs/2511.17045 (AAAI 2026 Oral) |

```bash
hf download linfeng302/RacketVision --repo-type dataset --local-dir data
```

**Size:** 7.54 GB. 1,672 clips of 5–10 s from 942 YouTube professional broadcasts. Extracted frames and median backgrounds are not included; generate them with the repo's `DataPreprocess/extract_frames.py` and `create_median.py`.

| Sport | Clips | Frames | Ball labels | Train / val / test clips |
|---|---|---|---|---|
| Tennis | 431 | 150,399 | 21,544 | 350 / 38 / 43 |
| Badminton | 461 | 114,753 | 23,003 | 374 / 41 / 46 |
| Table tennis | 780 | 170,027 | 19,495 | 625 / 69 / 86 |

**Format:** all clips 1920x1080; frame rate varies (25, 30, 50, 60 fps). One CSV per rally at `<sport>/all/<match>/csv/<rally>_ball.csv` with columns `Frame,Visibility,X,Y` in 1080p pixels.

**Caveats:**

- **Labels are sparse.** Only 20% of frames are labelled, evenly spaced (every 5th frame at 25 fps, every 12th at 60 fps). With our 3-in/3-out model at most one of the three output frames has ground truth, so training on it needs a per-frame loss mask.
- **The dense labels are not human labels.** `<sport>/interp_ball/.../results.csv` holds interpolated predictions from their TrackNetV3 baseline.
- **Footage rights are unresolved.** The repo and dataset card say MIT, but the videos are YouTube broadcasts and neither the paper nor the card addresses rights to them. Issue #3 (2026-10-26) asks whether commercial training is allowed and has no reply. Treat as research-safe only.

**Weights shipped:** `balltrack_best.pth` (TrackNetV3, 46.6 MB) plus racket detection, pose and trajectory checkpoints. The model repo has no licence file.

## Pickleball — TrackNet-Pickleball (12,810 labelled frames, one match)

| Source | Link |
|--------|------|
| GitHub | https://github.com/AndrewDettor/TrackNet-Pickleball |

**The repo contains no dataset.** It holds code, notebooks and a PDF. What exists is on Google Drive, linked from the repo's Step2 and Step4 READMEs:

- `InPlayBalls_54min_vid_labels.csv`: 12,810 rows, 10,464 visible. Columns `Frame,Ball,x,y` with **normalised** x, y.
- A folder of preprocessed 512x288 three-frame batches (`x_data_1..8`, `y_data_1..8`, about 1.65–2.47 GB each).

**The raw video is not published** (issue #1 asking for it, opened 2024-12, has no reply). No licence. No commits since 2023-12-22.

**Quality:** low. Single match, and we cannot re-extract frames or re-check labels. We need our own recordings (target 50K+ frames).

## New since April 2026

Ranked by usefulness to us.

| Dataset | Sport | Size | Camera | Labels | Licence | Link | Status |
|---|---|---|---|---|---|---|---|
| BlurBall (CVsports @ CVPR 2026) | Table tennis | ~64K frames, 26 recordings | Fixed static cameras | Per-video CSV: frame, visibility, x, y, blur angle, blur length. Ball labelled at the centre of the blur streak | Code MIT; data licence unstated | https://github.com/cogsys-tuebingen/BlurBall | Repo loaded; download host timed out on 2026-11-15 |
| ETH one-shot shuttle detection (arXiv 2603.06691) | Badminton | 20,510 frames, 11 indoor/outdoor locations | Stationary industrial camera, 1920x1200 @ 60 fps | YOLO bbox. Frames without a visible shuttle were removed, so no negatives and no visibility flag | Unstated; code AGPL-3.0 | https://drive.google.com/drive/folders/1neBfC7-Fp-l53muxZiIbIET6lYH5xv9s | Folder listing loaded |
| tenis_backview (Kaggle) | Tennis | 10 videos, ~90 MB | Back of court, 1080x1920 @ 60 fps | Per-frame CSV: ball centre, court-line points, player keypoints | CC0 per a search snippet | https://www.kaggle.com/datasets/gastonarielfrancois/tenis-backview | **Unverified** (search snippets only) |
| PadelTracker100 | Padel | ~100K frames, ball in 57K | Elevated broadcast-style | COCO bbox; also pose and shot events | Labels CC BY 4.0; videos not included | https://zenodo.org/records/14653706 | Labels confirmed. Videos must be fetched from YouTube and re-aligned |
| ACE-Trajectories | Tennis | 517 trajectories, labels only | Re-annotates TrackNet tennis | Bounce timestamps, 16 court keypoints, camera matrices | GPL-3.0 | https://huggingface.co/datasets/XSpaceCoderX/ACE-Trajectories_withTosses | Confirmed. Adds bounce and court labels to data we already use |
| TTA (TOTNet) | Table tennis | 9,159 samples | Professional matches | TrackNet-style CSV | CC BY-NC 4.0, **no commercial use** | https://github.com/AugustRushG/TOTNet | Gated; requires a signed agreement |
| Tennis bounce/hit benchmark (arXiv 2610.38042) | Tennis | 8 clips, 345 events | Mixed, including static phone | Event frames only | CC BY 4.0 | https://github.com/David-dsv/tennis-bounce-hit-benchmark | Confirmed. Evaluation only, not training |

Notes:

- **ETH shuttle set:** confirm frames within a folder are consecutive before building 3-frame windows from it; visibility can only be reconstructed from gaps.
- **Watch list:** MIT egocentric tennis (arXiv 2610.37619) includes static court-corner cameras, the closest match to a phone on a fence. Data is promised on acceptance; nothing is released yet.
- **Not usable or not found:** TrackNetV4's multi-ball set (request-gated, research only); NCUTT and the Multi-View Tennis Ball Dataset (IEEE DataPort, paywalled); Roboflow pickleball sets (single-frame bbox, small); no rendered synthetic ball datasets; no data released with TrackNetV5 or TrackNetV6.

## Other

| Dataset | Sport | Size | Link | Notes |
|---------|-------|------|------|-------|
| WASB-SBDT | 5 sports | Hosts no data | https://github.com/nttcom/WASB-SBDT | Links to the originals. Tennis and badminton are the two SharePoint zips above. Soccer is non-commercial research only; volleyball is on Drive (57.8 GB); the basketball page is 404. Code MIT |
| OpenTTGames | Table tennis | ~35.4 GB, 1080p @ 120 fps | https://lab.osai.ai/ | CC BY-NC-SA 4.0 (non-commercial). Ball labels exist only in windows of −4..+12 frames around each of 4,271 events. Direct files at `https://lab.osai.ai/datasets/openttgames/data/{game_1..5,test_1..7}.{mp4,zip}` |
| Roboflow Hard Court | Tennis | 9,836 (April figure) | https://universe.roboflow.com/tennistracking/hard-court-tennis-ball/dataset/8 | Could not re-check; see log |

## Licensing for commercial use

Several sets are fine for research but unclear for a commercial product. Confirm with the authors before any of these enter a model we ship:

- **RacketVision:** MIT label, YouTube footage rights unanswered.
- **BlurBall, ETH shuttle set, TrackNet tennis, CoachAI badminton:** no data licence stated.
- **PadelTracker100:** footage belongs to World Padel Tour.
- **OpenTTGames, TTA:** explicitly non-commercial.

---

## Verification log

Last verified: **2026-11-15**

| Source | Status | Notes |
|--------|--------|-------|
| NYCU SharePoint tennis `Dataset.zip` | Live, downloaded | HEAD 200, 2,564,111,322 bytes. One transient 503. Downloaded and extracted in full; contents under Tennis above |
| Kaggle tracknet-tennis | Live | 2,593,200,419 bytes, version 2, last updated 2024-03-01 |
| Google Drive TrackNet folder | Live | Third-party mirror; files modified Jan–Jul 2023 |
| NYCU GitLab TrackNetv2 | Live | Code only. Last commit 2021-04-12 |
| CoachAI SharePoint (badminton) | Live | HEAD 200, 1,731,293,072 bytes. Zip index read: 201 mp4 + 201 csv |
| CoachAI hackmd page | Live | Content unchanged |
| wywyWang/CoachAI-Projects | Live | 292 stars, last push 2026-07-17 (unrelated to tracking) |
| OrcustD/RacketVision | **Released** | 100 stars, last push 2026-04-11. Hugging Face dataset and model repos loaded, sample files read |
| AndrewDettor/TrackNet-Pickleball | Live, no data in repo | 39 stars, 14 forks, last push 2023-12-22. Labels CSV downloaded from Drive |
| nttcom/WASB-SBDT | Live | 212 stars, last push 2023-11-23. Training code still "TBA" |
| OpenTTGames (lab.osai.ai) | Live | Page loaded, HEAD on 3 files |
| cogsys-tuebingen/BlurBall | Repo live | Download host (NextCloud) timed out |
| ETH shuttle detection Drive folder | Live | Folder listing loaded |
| PadelTracker100 (Zenodo) | Live | Labels only |
| Roboflow Hard Court | **Could not verify** | Cloudflare 403; API needs a key. Manual check needed |
| Kaggle tenis_backview | **Could not verify** | Page renders empty without JavaScript |
