# Vision Workflow README

This document is the practical operator guide for the `vision/` workflow:

- run detection and screenshot capture
- label screenshots in the desktop UI
- export labels for training/evaluation

For roadmap and slice sequencing, see `vision/PLAN.md`.

## 1) Run inference and save screenshots

From repo root:

```bash
python -m vision.infer --input <video-or-dir> --out-dir <out> --screenshots
```

Use env vars in `.env` / `.env.example` for defaults:

- `VISION_SCREENSHOT_DIR`
- `VISION_VIDEO_SAMPLE_INTERVAL_SEC`
- `VISION_SCREENSHOT_MIN_VIDEO_SEC`
- `VISION_SCREENSHOT_MIN_INTERVAL_SEC`

Expected outputs:

- PNG screenshot files
- sibling JSON sidecars with model output (`present`, `count`, `boxes`, etc)

## 2) Run labeling app

```bash
python -m vision.label_app
```

Open `http://127.0.0.1:8765`.

### Assist maps for region grow (optional)

Offline preprocessing adds two PNGs next to each screenshot (same directory):

- `{stem}_assist_fg.png` — foreground confidence vs a median or supplied background
- `{stem}_assist_edge.png` — edges plus optional superpixel boundaries (needs `opencv-contrib-python-headless` for SLIC)

Generate them from repo root (defaults to `VISION_SCREENSHOT_DIR`):

```bash
python -m vision.label_assist_preprocess --help
# Example:
python -m vision.label_assist_preprocess --dir /path/to/screenshots
```

In **Region grow + brush** mode, the UI can **gate** flood fill using FG (skip low-FG pixels; disabled for **Erase** / train id 0) and **block** expansion across strong edge pixels. Tweak **FG min** and **Edge cross** if grow stops too early or still leaks.

The UI is desktop-first:

- screenshot and overlays on the left
- labeling form on the right

## 3) Labeling instructions

### Core fields

- **Train visible**
- **How many trains to label separately**
- per-train panels: provider, engine/consist, car count
- notes

### YOLO overlay vs human labels

- Yellow rectangles are **model hints** from sidecar JSON.
- Human labels are source-of-truth for training/eval.

If YOLO says 1 but you see 2 trains: set train count to 2 and label both trains.

### Spatial overlay (instance separation)

Choose **Coarse grid** or **Region grow + brush**:

**Grid mode**

- Choose a brush: `Train 1`, `Train 2`, ... or `Erase`
- Click or drag across cells
- Stored in `train_grid_json` (per-cell train ids)

**Region grow + brush**

- **Grow (click)**: flood-fill similarly-colored pixels (LAB distance) from the click; tune **Grow sensitivity**
- Optional **Gate grow with FG** / **Block at edges** when `*_assist_*.png` sidecars exist (see above)
- **Brush (drag)**: circular stamp to paint or erase; tune **Brush radius**
- Mask is stored downsampled (max side ~400px, ≤200k pixels) as **`train_mask_json`**: `{w,h,rle}` run-length encoding (`0` = empty, `1..6` = train id)

Use **Show paint on image** when overlays are distracting (grid + mask both respect this).

You may keep **both** grid and mask on one row if you switch modes; exports can include both fields.

### Ambiguous cell rule

A single grid cell can store only one train ID.

If two trains share one cell:

1. assign the dominant train in that cell
2. keep per-train metadata correct in train panels
3. add a notes comment (e.g. "two trains overlap in one cell")

## 4) Export labels

Download from UI or call:

`GET /label/export.jsonl`

Export includes non-skipped rows from `db/vision_labels.db`.

Important fields:

- `present`
- `num_trains`
- `trains_json` (per-train metadata array)
- `train_grid_json` (optional coarse grid)
- `train_mask_json` (optional region-grow / brush mask, RLE)

Compatibility note:

- top-level `provider`, `engine_model`, `num_cars` mirror train 1 for simpler downstream scripts.

## 5) Suggested QA process for multiple volunteer labelers

When 5+ labelers are active, use lightweight consensus:

- double-label a random 10-20% sample
- prioritize hard frames for overlap (night, occlusion, two trains close together, merge cases)
- resolve disagreements by notes review + quick second pass

This catches drift while keeping throughput high.

## 6) What this dataset is best for first

Most reliable near-term targets:

- train presence and count quality
- multi-train vs single-train identification
- detector error analysis (misses, merges, false alarms)

Per-train attributes (provider/consist/cars) are still valuable, but quality depends on how separable trains are in-frame.
