# Training plan: from SAM-style labels to stream assist (detect, outline, badge)

This document assumes you already have a **fully labeled dataset** (~10k images to start, growing over time). Each image has:

- **Per-pixel instance assignment**: regions labeled as train 1, train 2, … (or background), e.g. from SAM + manual cleanup in your UI.
- **Per-train metadata**: at least **provider** (e.g. Amtrak), **engine/consist model** (free text or normalized), **car count** (integer).

Goal on a **live stream**: (1) coarse **what is on screen**, (2) **outline** moving objects, (3) **badges** (e.g. Amtrak logo + train index) anchored to each object.

---

## 1. Freeze what “ground truth” means

Before training anything, pin down definitions so labels and metrics stay consistent:

| Signal | What the model should learn | Label source |
|--------|-----------------------------|--------------|
| **Object presence** | Is there at least one train? | `present` + non-empty mask |
| **Instance masks** | Which pixels belong to which physical object | RLE / raster mask per train id |
| **Instance boxes** | Tight boxes for tracking and UI anchoring | Derived from masks (min-area rect or axis-aligned bbox) |
| **Provider / operator** | Amtrak, MARC, … | Per-train field |
| **Engine model** | Fine-grained class or embedding target | Per-train text → normalized label or multi-label |
| **Car count** | Regression or ordinal bucket | Per-train integer |

**Engine model** is the messiest: free text should be **canonicalized** (mapping table + “other”) or treated as **hierarchical** (operator → family → variant). Otherwise the head will never stabilize.

---

## 2. Export format for training (one-time + repeatable)

From your DB / JSONL:

1. **Images**: paths to PNGs (unchanged).
2. **Instance segmentation** (recommended primary supervision):
   - YOLO-seg style: one label file per image with **polygon or mask** per instance, **class id** = your coarse class (e.g. single class `train` to start, or `passenger_train` / `transit` if you split).
   - Or COCO panoptic/segmentation JSON if you prefer Detectron2 / MMDetection.
3. **Derived boxes**: for each instance, store `xyxy` from mask (for detection-only baselines and trackers).
4. **Metadata sidecar** per image (JSON): list of instances with `{train_index, provider_id, engine_id_or_text, num_cars}` aligned to instance order **or** stable instance ids stored at export time.

**Why repeat this step in the loop:** whenever you change taxonomy (new provider, engine normalization), re-export so training and eval use the same mapping.

---

## 3. Model strategy (what to train for the three stream jobs)

The three stream features are **not** one monolithic YOLO checkbox; they are a **stack** that shares a backbone when useful.

### 3.1 Highlight outline (instance shape)

**Train:** an **instance segmentation** model (e.g. **YOLOv8-seg / YOLO11-seg** in Ultralytics, or Mask R-CNN family).

- **Input:** frame (or short clip window later).
- **Output:** per-instance mask or contour → draw outline in the UI.
- **Labels:** your SAM-refined masks are ideal supervision.

Starting with **one class “train”** (all instances same semantic class) is valid: the network still learns **instance separation** from mask boundaries. Add coarse classes later if you have consistent labels.

### 3.2 Classify what’s on screen (easier path)

Two layers:

1. **Image-level / ROI-level classification**  
   - **Simple:** multiclass **provider** from a **cropped ROI** (box expanded from mask) → small CNN or ViT head.  
   - **Engine model:** separate head or hierarchical classification after provider.

2. **Optional open-vocabulary** (later)  
   - **YOLO-World** / grounding-style models if you need rare text queries; heavier and trickier to evaluate. Prefer fixed taxonomy first.

**Labels:** provider + engine + cars from your per-train metadata, keyed to the same instance id as the mask export.

### 3.3 Badge + logo + train number

This is **UI + tracking**, not only detection:

1. **Detect/segment** each frame (or every *n*th frame) → masks/boxes.
2. **Track** identities across time (**ByteTrack**, **BoT-SORT**, or a simple IoU matcher if frame rate is low and motion is smooth).
3. **Assign “train #”** for the stream: stable **track id** (1, 2, …) or match to a **database** of known consists if you ever have that.
4. **Badge content:**  
   - Logo = lookup from **provider** class.  
   - Text = `Track 2` or `Train B` + optional **engine short name** from classifier.

**Train # in labels vs stream:** your dataset’s “train 1 / train 2” is **per-image instance index**, not global identity. Do **not** force the model to predict the same index across videos; use **tracking** at runtime for stable badges.

---

## 4. Recommended phased roadmap

### Phase A — Baseline that works (prove the pipeline)

1. Export masks + derived boxes.
2. Train **segmentation** with one class `train`, high-res input if GPUs allow.
3. Metrics: mask IoU, boundary F-score on a **held-out time split** (different days/clips), not random frames from the same video.
4. Stream prototype: run seg every *k* frames, **track** boxes/masks, draw outlines + placeholder badges (“Train A”, “Train B”).

**Why a time split:** trains and lighting are correlated in a clip; random splits lie about generalization.

### Phase B — Metadata heads (provider, cars, engine)

1. Crop instances from boxes (padding 10–20%).
2. Train small **multi-task** head or separate classifiers:
   - Provider: categorical cross-entropy.
   - Cars: regression (Huber) or buckets (0–3, 4–6, 7+).
   - Engine: start with **top-K frequent** classes + `other`; or contrastive / metric learning if labels stay noisy.

3. At inference: run seg → crop → classify per track (temporal **majority vote** over last *N* frames to stabilize badges).

### Phase C — Scale and hard negatives

1. Mine **false positives** (background blobs), **merged trains**, **occlusion** from failed QA or low IoU preds.
2. Prioritize labeling those (your labelers + SAM speed).
3. Optionally add **auxiliary losses** (edge alignment) only if masks are still fuzzy after more data.

---

## 5. Intermittent “label → export → train” loop (explicit)

You should **explicitly** repeat parts of the pipeline whenever any of these happen:

| Trigger | What to redo | Why |
|---------|----------------|-----|
| +5–20% new labeled images | Re-export + **fine-tune** (few epochs) | New scenes dominate loss; stale model drifts. |
| New operator / camera / bitrate | Re-export if taxonomy changed + fine-tune | Distribution shift. |
| Engine taxonomy updated | Re-export labels + retrain heads | Old class ids no longer match. |
| Segmentation good enough but badges noisy | Add/refresh **crop classifier** only | Cheaper than full seg retrain. |
| Before a “release” | Full eval on **fixed** benchmark split | Prevents comparing apples to oranges across weeks. |

This is **not** wasted work: each loop shrinks the gap between **training distribution** and **live stream distribution**. Unlimited future data only helps if you **sample** and **validate** deliberately (active learning on errors).

---

## 6. Concrete tooling sketch (YOLO-centric)

- **Framework:** Ultralytics YOLO **segment** train on exported YOLO-dataset layout.
- **Tracking:** `ultralytics` trackers or standalone ByteTrack on det/seg boxes.
- **Second stage:** PyTorch **Lightning** or plain PyTorch for small crop classifiers; or a second YOLO **cls** model on cropped chips.

**Hardware / spend:** see **§7** for M4 Max vs cloud GPU tradeoffs and rough dollar estimates.

---

## 7. Hardware estimates & cloud costs (M4 Max local + optional cloud)

**Disclaimer:** Cloud list prices change by region, contract, and spot vs on-demand. Treat everything below as **order-of-magnitude** for budgeting; check each vendor’s calculator before you commit.

### 7.1 What your M4 Max is good for

Apple Silicon (MPS in PyTorch, or Core ML after export) is strongest for **inference**, **prototyping**, and **small fine-tunes**. Unified memory size matters a lot (e.g. 36 GiB vs 128 GiB).

| Workload | M4 Max (typical) | Notes |
|----------|------------------|--------|
| **Interactive labeling / SAM in your UI** | Comfortable | You are already doing this locally. |
| **Stream inference** (seg + track + light classifier) | Often viable | Prefer **smaller** seg weights (`yolo11n-seg`, `yolov8n-seg`); run at **720p** or stride frames (e.g. infer every 2nd–5th frame, track in between). |
| **Export for production** | Core ML / ONNX | Good path for low-latency Mac or iOS-adjacent deployment. |
| **From-scratch / large seg train on ~10k full-res images** | Slow / painful | May take **many hours to days** vs **hours** on a mid-range NVIDIA GPU; batch size and image size will be limiting. |
| **Crop classifier (provider / engine / cars)** | Very reasonable | Small ResNet / ViT-tiny on 224×224 crops trains quickly on MPS. |

**Rule of thumb:** use the Mac for **iteration and deployment**; rent a **single cloud GPU** for **heavy YOLO-seg training** when you want wall-clock speed.

### 7.2 When to pay for cloud GPUs

Rent cloud when any of these dominate:

- Training **segmentation** for many epochs at **high input size** (e.g. 960–1280 long side).
- Hyperparameter sweeps (many runs).
- Larger models (`m/l/x` seg) or future **video** / heavier heads.

Stay local when:

- Fine-tuning a **small** seg model for a few epochs after a big cloud run.
- Training **crop classifiers** and calibrating heads.
- Running **eval** and **stream demos** on a single machine.

### 7.3 Rough training time (illustrative)

Assuming **~10k images**, YOLO-style **instance seg**, single GPU, ballpark only:

| Setup | Order of magnitude |
|-------|---------------------|
| **nano / small** seg, img ~640, 50–100 epochs | **~2–12 GPU-hours** |
| **medium** seg, img ~896–1024, 100 epochs | **~12–48 GPU-hours** |
| **Crop classifier** (per-instance crops, small CNN) | **~0.5–4 GPU-hours** |

Doubling dataset size roughly scales linearly in compute until you hit I/O or augmentation bottlenecks.

### 7.4 Cloud cost bands (USD, on-demand–ish, 2025–2026 style)

Use **GPU-hours × $/hr** for a quick budget. Examples of **typical** on-demand hourly rates (varies by region and instance; **spot** can be **30–70%** lower but interruptible):

| Tier | Example SKUs | Rough $/hr (on-demand) | Good for |
|------|----------------|-------------------------|----------|
| **Budget** | RTX 4090 / 4080 class (consumer GPUs on rental marketplaces) | **~$0.35–$0.90** | Fast iteration, small/medium seg |
| **Sweet spot** | **L40S** (e.g. AWS **g6e.xlarge**–class), A10 | **~$1.5–$3.5** | Default recommendation for YOLO-seg at 10k scale |
| **Heavy** | **A100 40/80 GB**, **H100** | **~$2.5–$8+** | Large models, very high res, or impatient schedules |

**Example back-of-envelope:**

- **20 GPU-hours** at **$2/hr** → **~$40** per full training run.  
- **5 runs** (sweeps / mistakes) → **~$200** in compute, plus storage and egress.

Add **~$5–30/mo** for object storage (datasets + checkpoints) if you keep everything in S3/GCS equivalent; egress can dominate if you download huge archives repeatedly—prefer **train in the same region** as the data.

### 7.5 M4 Max + cloud hybrid workflow (recommended)

1. **Develop & export** locally on M4 Max (dataset scripts, small overfit tests, crop models).
2. **Big seg train** on one **L40S / A10** instance for a night or weekend.
3. **Download best checkpoint** (or push to artifact store); **quantize / Core ML convert** on Mac if needed.
4. **Stream prototype** on M4 Max; measure FPS and latency.
5. **Repeat** when the label → export → train loop adds enough new data (§5)—each repeat might be **only fine-tune** (few GPU-hours) instead of full trains.

### 7.6 Can the trained model run on “regular” hardware? (e.g. older gaming PC)

**Yes, if you design for it at train time.** The model is not tied to an M4—you export weights and run inference wherever you ship the app.

**Gaming PC with an NVIDIA GPU (even older)** is often a **good** deployment target for YOLO-style models: CUDA + **TensorRT** or **ONNX Runtime (CUDA)** is mature and fast. An older card (e.g. GTX 1060/1660, RTX 2060) can still run a **small** segmentation model at **720p** with **strided inference** (e.g. run the net every 2nd–4th frame, track in between). An M4 Max may beat a **very** old GPU on power efficiency and some pipelines, but a **mid-range desktop GPU** is usually **competitive or faster** for vanilla Ultralytics/CUDA inference.

**Practical knobs for weak / older machines:**

| Knob | Effect |
|------|--------|
| **Smallest seg variant** (`n` / `nano`) | Largest FPS win; train and deploy the same small variant when possible. |
| **Lower input size** (e.g. 640 vs 960) | Big speedup; slight accuracy loss—often acceptable for overlays. |
| **Infer every Nth frame + tracker** | Cuts GPU cost linearly; badges stay stable if tracking is good. |
| **Fewer simultaneous instances** | Worst case is multiple large masks per frame. |
| **FP16 / INT8** (TensorRT, ORT) | Often **1.5–3×** faster on NVIDIA; validate accuracy on your benchmark. |
| **CPU-only** | Possible for emergencies, but expect **much** lower FPS (often single digits at useful res)—treat as fallback, not primary stream path. |

**Summary for your friend’s stream box:** target **YOLO-seg nano/small + 720p (or smaller letterboxed) + stride + TensorRT**; profile on their exact GPU. If it’s still tight, drop to **detection-only** boxes for tracking and draw **simple rectangles** instead of full masks until they upgrade hardware.

### 7.7 Live highlighting & tracking (what actually runs each frame)

For **streams**, treat “smooth outlines and stable badges” as a **two-speed** pipeline. That is what makes live highlighting feasible on a modest gaming PC.

| Stage | How often | Hardware | Role |
|--------|-----------|----------|------|
| **Detection / segmentation** | Every **N** frames (e.g. N=2–5) or on a **timer** (~5–15 Hz) | GPU (heavy) | Finds objects, refreshes masks or boxes |
| **Multi-object tracking** | **Every frame** (full video FPS) | **CPU is fine** for many trackers | Keeps **stable IDs** (track 1, 2, …), predicts motion between DNN refreshes |
| **Overlay draw** | Every frame | GPU (compositing) or CPU (simple) | Draws last mask/box, badge anchor, trail |

**Highlighting between DNN frames**

- **Cheap:** draw the **last** mask or **axis-aligned box** from the tracker’s predicted position; small **linear interpolation** or Kalman-smoothed box corners look smooth.
- **Medium:** keep a **low-res mask** and **shift/scale** it with the tracked box (good enough for many rail scenes).
- **Expensive:** run full seg **every** frame—usually unnecessary if the tracker is decent.

**Trackers that fit this stack:** **ByteTrack**, **BoT-SORT**, or Ultralytics’ built-in trackers fed by your seg **boxes** (or mask-derived boxes). They associate detections to IDs with IoU + motion; no extra neural net required for the per-frame part.

**Why this matters for “regular” PCs:** you are **not** asking the GPU to hit 30–60 FPS on a huge seg model—you only need **5–15 good refreshes per second**, while the **tracker** carries identity and smooth motion at full FPS on CPU.

**Caveats (brief):** occlusions, crossing trains, and camera cuts cause **ID switches**; mitigations include higher DNN rate on “busy” scenes, short hysteresis on badge text, and optional **Re-ID** (extra model) only if you truly need identities across long occlusions.

This aligns with the badge story in **§3.3**: the **train number on stream is the track ID**, not the per-image instance index from labeling.

---

## 8. Evaluation checklist (before trusting the stream)

- **Segmentation:** mask IoU / boundary distance on **time-held-out** clips.
- **Detection proxy:** box mAP from derived boxes if you care about tracking input quality.
- **Classification:** per-provider accuracy **per track** (majority vote), not only per frame (reduces flicker).
- **Stream UX:** latency budget (ms per frame), **max simultaneous instances**, behavior on occlusion and tunnel/dark frames.
- **Tracking:** ID switch rate on held-out clips at your chosen **infer stride** (full FPS overlay, lower DNN rate).

---

## 9. Summary

1. **Treat your SAM-quality masks as gold** for **instance segmentation** training (outlines).
2. **Derive boxes** for tracking and for **crop-based** provider/engine/cars models (badges).
3. **Separate** “semantic segmentation / instance” from **identity over time**: use a **tracker** for stable train numbers and badge placement on stream.
4. **Repeat** export + fine-tune when data or taxonomy grows—each repeat is a **distribution alignment** step, not rework for its own sake.

This gets you from a fully labeled 10k-image set to a stream that can classify, outline, and badge moving objects, with a clear path to absorb unlimited future labels.
