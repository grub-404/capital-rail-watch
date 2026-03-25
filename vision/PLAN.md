# Vision feature slice plan

Incremental delivery for pretrained YOLO (train / locomotive / railcar), yes-no presence, boxes + count, screenshot capture, yt-dlp test corpus, and a minimal labeling webapp. Gamification comes in a later slice.

## Principles

- **Pretrained first**: YOLO with classes such as `train`, `locomotive`, `railcar` (exact names depend on the weight file; map them to one logical “rail vehicle” for yes/no and count).
- **Offline before live**: yt-dlp clips are the regression set; live stream wiring is optional until mid-plan.
- **Screenshots as a product**: deterministic naming and JSON sidecars so labeling and debugging stay cheap.

---

## Slice 0 — Repo + contracts

**Goal:** One place for vision code, env vars, and a frozen JSON schema for detections.

**Deliverables**

- This `vision/` package: config loader, shared types (Pydantic or dataclasses), constants for class-name mapping.
- Root `.env.example` entries (when implemented): e.g. `VISION_MODEL_PATH`, `VISION_CONF_THRESHOLD`, `VISION_SCREENSHOT_DIR`, optional `YTDLP_OUTPUT_DIR`.
- **Detection JSON schema** (in code + one example fixture): `present`, `count`, `boxes[]` with `{x, y, w, h, conf, label}`, `frame_id`, `source`, `timestamp_utc`.

**Testing**

- Unit tests: schema validation (valid / invalid payloads).
- Unit tests: class mapping (“locomotive” + “railcar” → counted as train for yes/no).

**Exit criteria**

- Tests runnable locally (`pytest`); no model download required for unit tests (mock model or skip integration behind an env flag in later slices).

**Status:** Implemented — `vision/schema.py`, `vision/labels.py`, `vision/config.py`, `vision/fixtures/detection_example.json`, `tests/vision/`, root `requirements-dev.txt`. Run `python -m pytest tests/ -q` locally.

---

## Slice 1 — Pretrained inference CLI + fixtures

**Goal:** Run YOLO on single images and video files; output JSON and optional annotated image.

**Deliverables**

- CLI: e.g. `python -m vision.infer --input path --out-dir ...` or `scripts/vision_infer.py`.
- Load pretrained weights; map COCO or custom class names to the unified train-like set.
- Write `result.json` alongside outputs.

**Testing**

- **Golden file test**: one or two small test images with expected `count` and rough box tolerances.
- **Smoke test**: run on a 2–3 second clip; assert no crash, JSON parses, `count >= 0`.

**Exit criteria**

- Reproducible command developers can run locally; usage documented in module docstring or README.

**Status:** Implemented — `vision/yolo_infer.py`, `vision/infer.py` (`python -m vision.infer`), `ultralytics` in `requirements-dev.txt`, `tests/vision/test_infer.py`, `vision/test_stills/`.

---

## Slice 2 — yt-dlp harness + test corpus layout

**Goal:** Download segments of older streams into a standard folder tree for batch evaluation.

**Deliverables**

- Script wrapping yt-dlp (`--download-sections` or trim with ffmpeg after download).
- Convention: `data/clips/{source_id}/{clip_id}.mp4` plus `manifest.csv` (url, time range, notes).

**Testing**

- Dry-run mode: validates URLs/args, writes manifest only.
- Optional integration test (`@pytest.mark.integration`): one short known-good sample; skip unless `YTDLP_E2E=1` (or similar).

**Exit criteria**

- Anyone can refresh the regression clip set without changing application code.

**Status:** Implemented — `vision/clip_fetch.py`, `vision/fetch.py` (`python -m vision.fetch`), `data/clips/manifest.csv`, `scripts/fetch_clips.sh`, `yt-dlp` in `requirements-dev.txt`, `tests/vision/test_fetch.py`, `pytest.ini` (`integration` mark).

---

## Slice 3 — Screenshot saver + session metadata

**Goal:** When a train is present (or on a configurable interval), save PNG plus JSON sidecar with the full detection payload.

**Deliverables**

- Writer: input frame (array or path); output screenshot path + metadata.
- Naming: `{timestamp}_{source}_{frame_index}.png` and matching `.json`.
- Config: minimum interval between shots; optional crude retention or quota.

**Testing**

- Unit: fake frame → writer creates files; JSON round-trips.
- With mock detector: N frames → correct file count and monotonic timestamps.

**Exit criteria**

- A directory the label app (slice 5) can consume.

**Status:** Implemented — `vision/screenshot.py` (`ScreenshotWriter`), wired to `python -m vision.infer --screenshots`; inference sampling **`VISION_VIDEO_SAMPLE_INTERVAL_SEC`**; screenshot spacing **`VISION_SCREENSHOT_MIN_VIDEO_SEC`** (video timeline, per clip) + optional **`VISION_SCREENSHOT_MIN_INTERVAL_SEC`** (wall clock); filenames include **`vidt…ms`** + frame; tests in `tests/vision/test_screenshot.py`.

---

## Slice 4 — HTTP API + optional bridge from Flask

**Goal:** Same logic as CLI, callable over HTTP for OBS, browser, or a poller.

**Deliverables**

- Small **FastAPI** service *or* a few routes on the existing Flask app (prefer separate process if it keeps torch isolated from `backend/server.py`).
- `POST /vision/infer` (multipart image or base64 JSON) → detection JSON.
- Optional: infer + save screenshot in one call.

**Testing**

- `pytest` with `httpx` / `requests`: upload test image → 200 + schema match.
- Optional: max payload / timeout behavior.

**Exit criteria**

- Overlay or a standalone watcher can integrate without tangling ingest code.

---

## Slice 5 — Minimal label webapp (MVP)

**Goal:** Railfans mark train yes/no, number of trains, and optional notes; accounts optional at first.

**Deliverables**

- Simple UI + backend: list unlabeled screenshots (from directory or DB); show image; submit `present`, `count`, notes.
- Store labels in SQLite or CSV keyed by `screenshot_id` or file hash.
- Export: `labels.jsonl` for future training.

**Testing**

- Backend: temp dir with fake images → list → label → export; assert contents.
- Optional E2E (Playwright): one label flow → row in DB.

**Exit criteria**

- Real labeling session possible on screenshots from slice 3.

---

## Slice 6 — Batch evaluation on clip corpus

**Goal:** Measure behavior on yt-dlp clips, not only “it runs.”

**Deliverables**

- Script: run detector over every clip in manifest; aggregate metrics (precision/recall if human labels exist, or simpler consistency stats: % frames with detection, average count).
- Report: CSV or HTML per clip.

**Testing**

- Golden: small manifest with expected aggregate stats on a toy clip.

**Exit criteria**

- Threshold or weight changes require rerunning the report and comparing numbers.

---

## Slice 7 — (Later) Gamification + prestige

**Goal:** Accounts, leaderboard, streaks, verified labeler badges — **after** MVP labeling is stable.

**Testing**

- Rate limits; duplicate labels; consensus when two users disagree.

---

## Cross-cutting test matrix

| Concern | What to verify |
|--------|----------------|
| Class mapping | All pretrained labels map; unknown labels do not crash. |
| Thresholds | Boundary confidence values flip `present` as expected. |
| Multi-train | Overlapping boxes: document dedupe or no-dedupe rule; test it. |
| Disk I/O | Permissions; graceful behavior when disk is full. |
| Public API | Max upload size; content-type checks if exposed. |

---

## Milestones

- **Milestone A (internal):** Slices 0–3 + tests — generate screenshots from streams or clips.
- **Milestone B (community):** Slices 4–5 + tests — external labelers; exportable dataset.

---

## Dependency order

```
0 → 1 → 2   (contracts, inference, offline corpus)
    ↓
    3       (screenshots)
    ↓
    4       (HTTP)
    ↓
    5       (labeling)
    ↓
    6       (batch eval)
    ↓
    7       (gamification, optional)
```
