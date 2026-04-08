# Training Data Sources

## Tennis — TrackNet Dataset (~20K frames)

| Source | Link | Notes |
|--------|------|-------|
| Kaggle | https://www.kaggle.com/datasets/sofuskonglevoll/tracknet-tennis | Easiest download via CLI |
| Google Drive mirror | https://drive.google.com/drive/folders/11r0RUaQHX7I3ANkaYG4jOxXK1OYo01Ut | Original V1 dataset |
| NYCU GitLab | https://gitlab.nol.cs.nycu.edu.tw/open-source/TrackNetv2 | Official V2, link may be degraded |

**Format:** 1280x720, 30fps. CSV labels: Frame, Visibility, X, Y

### Kaggle download

```bash
pip install kaggle
# Get API token from https://www.kaggle.com/settings -> "Create New Token"
# Save to ~/.kaggle/kaggle.json
kaggle datasets download sofuskonglevoll/tracknet-tennis
```

## Badminton — CoachAI Shuttlecock Dataset (78K frames)

| Source | Link |
|--------|------|
| SharePoint (official) | https://nycu1-my.sharepoint.com/:u:/g/personal/tik_m365_nycu_edu_tw/EWisYhAiai9Ju7L-tQp0ykEBZJd9VQkKqsFrjcqqYIDP-g |
| CoachAI GitHub | https://github.com/wywyWang/CoachAI-Projects |
| Docs | https://hackmd.io/@TUIK/rJkRW54cU |

**Format:** 1280x720, 30fps. 26 broadcast videos. CSV: Frame, Visibility (0/1), X, Y

**Download:** Manual only (SharePoint link). No CLI option found.

## Pickleball (~12K frames, single match)

| Source | Link |
|--------|------|
| GitHub | https://github.com/AndrewDettor/TrackNet-Pickleball |

**Quality:** Low — single match, limited variety. Need to collect more (target 50K+ frames).

## Multi-sport — RacketVision (435,179 frames)

| Source | Link | Status |
|--------|------|--------|
| GitHub | https://github.com/OrcustD/RacketVision | Repo public (created 2025-11-21, 55+ stars). Code and data marked **"Coming Soon"** as of 2026-04-07 — README asks users to star to be notified |
| arXiv | https://arxiv.org/abs/2511.17045 | Confirmed AAAI 2026 Oral. Authors: Dong, Yang, Wu, Wang, Hou, Zhong, Sun |

1,672 video clips / 435,179 frames / 12,755 seconds covering tennis, badminton, and table tennis. First benchmark with joint ball + racket pose annotations from 942 professional matches. Targets fine-grained ball tracking, articulated racket pose estimation, and trajectory forecasting.

## Other

| Dataset | Sport | Frames | Link |
|---------|-------|--------|------|
| WASB-SBDT | Multi (5 sports) | varies | https://github.com/nttcom/WASB-SBDT |
| OpenTTGames | Table tennis | ~50K+ | https://lab.osai.ai/ (CC BY-NC-SA 4.0) |
| Roboflow Hard Court | Tennis | 9,836 | https://universe.roboflow.com/tennistracking/hard-court-tennis-ball/dataset/8 |

---

## Verification log

Last verified: **2026-04-07**

| Source | Status | Notes |
|--------|--------|-------|
| Kaggle tracknet-tennis | Live | 2.56 GB, last updated 2024-03-01, version 2 |
| NYCU GitLab TrackNetv2 | Live | 29 commits, README present |
| CoachAI SharePoint (badminton) | Live | 78,200 frames documented on Hackmd |
| CoachAI Hackmd page | Live | Last updated July 2023 |
| wywyWang/CoachAI-Projects | Live | 223 stars, last push 2024-08-10 |
| AndrewDettor/TrackNet-Pickleball | Live | 24 stars, 11 forks, last push 2023-12-22 |
| OrcustD/RacketVision | Repo live, data **Coming Soon** | 55 stars, created 2025-11-21 |
| nttcom/WASB-SBDT | Live | 166 stars, BMVC 2023, last push 2023-11-23 |
| OpenTTGames (lab.osai.ai) | Live | ~35 GB, CC BY-NC-SA 4.0 |
| Roboflow Hard Court | Could not auto-verify (HTTP 403 anti-bot) | Manual check recommended |
