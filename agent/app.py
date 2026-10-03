"""Agent 3 (Python) - Object Detection & Tracking API.  Run: python agent\\app.py  (port 5001)
Any description (typos allowed) -> detect candidates (tiled, so far/small objects are found) ->
CLIP-rank against the description -> track each candidate -> return the TOP 3 with thumbnails.
Uses the GPU automatically if CUDA is available, otherwise the CPU."""
import base64, difflib
from flask import Flask, request, jsonify
import cv2
import numpy as np
import torch
import clip
from PIL import Image
from ultralytics import YOLOWorld

# ---------- device (GPU if available, else CPU) ----------
GPU = torch.cuda.is_available()
DEVICE = "cuda" if GPU else "cpu"

CONF = 0.05             # detector sensitivity (low = finds small / far objects)
MIN_HITS = 2            # confirmations needed for a candidate to count
MIN_MATCH = 0.30        # minimum CLIP score for a detection to be kept
MAX_TRACKS = 20         # safety cap on candidate tracks
MISS_LIMIT = 3          # checks in a row without the object => lost
TOP_K = 8               # max ranked matches to return
TILE_FRAC = 0.6         # each of the 2x2 tiles covers 60% of the frame (20% overlap)

if GPU:                 # fast machine: check often, use full frame + tiles
    TRACK_FPS = 10
    SEARCH_EVERY_SEC = 1.5
    VERIFY_EVERY_SEC = 2
    MIN_GAP = 1.0           # never run the detector more often than this (seconds of video)
    USE_FULL_FRAME = True
    TILE_IMGSZ = 960
    MAX_SIDE = 1920         # frames larger than this are shrunk before analysis
    MAX_CANDIDATES = 30
else:                   # CPU: check less often, tiles only
    TRACK_FPS = 5
    SEARCH_EVERY_SEC = 3
    VERIFY_EVERY_SEC = 4
    MIN_GAP = 2.0
    USE_FULL_FRAME = False
    TILE_IMGSZ = 640
    MAX_SIDE = 1280
    MAX_CANDIDATES = 20

COLORS = ["red", "pink", "orange", "yellow", "green", "blue",
          "purple", "black", "white", "gray", "brown", "beige"]
COLOR_ALIAS = {"grey": "gray", "violet": "purple", "maroon": "red", "navy": "blue",
               "cream": "beige", "golden": "yellow", "gold": "yellow", "silver": "gray",
               "skyblue": "blue", "magenta": "pink", "rose": "pink"}
# object word -> prompts given to the detector (several wordings find more candidates)
OBJECTS = {
    "bag": ["bag", "backpack", "handbag"], "backpack": ["backpack", "bag"],
    "bottle": ["bottle", "water bottle"], "book": ["book", "notebook"],
    "phone": ["cell phone"], "mobile": ["cell phone"], "laptop": ["laptop"],
    "umbrella": ["umbrella"], "wallet": ["wallet"], "watch": ["watch"],
    "cup": ["cup", "mug"], "box": ["box", "lunch box"], "tiffin": ["lunch box"],
    "shoe": ["shoe"], "cap": ["cap", "hat"], "jacket": ["jacket", "coat"],
    "charger": ["charger"], "glasses": ["glasses"], "pen": ["pen"], "chair": ["chair"],
}
VOCAB = COLORS + list(COLOR_ALIAS) + list(OBJECTS)
STOP = {"a", "an", "the", "and", "or", "with", "of", "my", "is", "has", "it", "its",
        "one", "that", "which", "color", "colour", "coloured", "colored"}
STATE = {}              # per-request settings

app = Flask(__name__)
model = YOLOWorld("yolov8s-worldv2.pt")
model.to(DEVICE)
clip_model, clip_pre = clip.load("ViT-B/32", device=DEVICE)
print(f"[agent] running on {DEVICE.upper()}"
      + (f" ({torch.cuda.get_device_name(0)})" if GPU else " (no GPU found)"))


# ---------- understanding the description ----------
def fix_word(w):
    if w in VOCAB or len(w) < 4:
        return COLOR_ALIAS.get(w, w)
    m = difflib.get_close_matches(w, VOCAB, n=1, cutoff=0.75)
    w = m[0] if m else w
    return COLOR_ALIAS.get(w, w)


def parse(text):
    raw = "".join(ch if ch.isalnum() else " " for ch in text.lower()).split()
    words = [fix_word(w) for w in raw if w not in STOP]
    colors = [w for w in words if w in COLORS]
    nouns = [w for w in words if w in OBJECTS]
    targets = []
    for n in nouns:
        targets += [p for p in OBJECTS[n] if p not in targets]
    if not targets:                                  # unknown object: try last word / last two words
        if words:
            targets = [words[-1]]
            if len(words) > 1:
                targets.append(" ".join(words[-2:]))
    noun = nouns[0] if nouns else (words[-1] if words else "object")
    return words, colors, noun, targets or ["object"]


def build_prompts(words, colors, noun):
    """Returns (prompts, indices of the prompts that mean 'this matches the description')."""
    if colors:
        base = " ".join(w for w in words if w not in COLORS) or noun
        prompts = [f"a photo of a {c} {base}" for c in COLORS]      # one prompt per colour
        pos = [COLORS.index(c) for c in colors]                     # asked colours count as a match
        if len(colors) > 1:                                         # e.g. "pink and blue bag"
            prompts.append(f"a photo of a {' and '.join(colors)} {base}")
            pos.append(len(prompts) - 1)
        prompts += ["a photo of a floor", "a photo of a desk", "a photo of a child", "a photo of a wall"]
        return prompts, pos
    desc = " ".join(words)
    return [f"a photo of a {desc}", f"a photo of a {noun}", "a photo of the floor", "a photo of a desk",
            "a photo of a person", "a photo of a chair", "a photo of a wall"], [0]


# ---------- helpers ----------
def near(obj, person, pad=0.5):
    w, h = obj[2] - obj[0], obj[3] - obj[1]
    return not (person[2] < obj[0] - pad * w or person[0] > obj[2] + pad * w or
                person[3] < obj[1] - pad * h or person[1] > obj[3] + pad * h)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0


def make_tracker():
    # KCF first: much faster than CSRT
    for name in ("TrackerKCF_create", "TrackerCSRT_create", "TrackerMIL_create"):
        create = getattr(cv2, name, None) or getattr(getattr(cv2, "legacy", None), name, None)
        if create:
            try:
                return create()
            except Exception:
                continue
    return None


def start_tracker(frame, cx, cy, w, h):
    tracker = make_tracker()
    if tracker is None:
        return None
    pad = max(8, 0.5 * max(w, h))
    H, W = frame.shape[:2]
    x1, y1 = max(0, int(cx - w / 2 - pad)), max(0, int(cy - h / 2 - pad))
    x2, y2 = min(W, int(cx + w / 2 + pad)), min(H, int(cy + h / 2 + pad))
    try:
        tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
        return tracker
    except Exception:
        return None


def pick(objs, cx, cy, w, h):
    best, best_d = None, 3 * max(w, h, 20)
    for o in objs:
        x1, y1, x2, y2 = o[1]
        d = (((x1 + x2) / 2 - cx) ** 2 + ((y1 + y2) / 2 - cy) ** 2) ** 0.5
        if d < best_d:
            best, best_d = o, d
    return best


def crop(frame, box, pad=0.15):
    x1, y1, x2, y2 = box
    H, W = frame.shape[:2]
    px, py = pad * (x2 - x1), pad * (y2 - y1)
    c = frame[max(0, int(y1 - py)):min(H, int(y2 + py)), max(0, int(x1 - px)):min(W, int(x2 + px))]
    return c if c.size else np.zeros((8, 8, 3), np.uint8)


def thumb(frame, box):
    c = crop(frame, box, 0.4)
    s = max(1.0, 160 / max(c.shape[:2]))
    c = cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", c)
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else None


# ---------- CLIP scoring ----------
def encode_text(prompts):
    with torch.no_grad():
        t = clip_model.encode_text(clip.tokenize(prompts).to(DEVICE)).float()
        return t / t.norm(dim=-1, keepdim=True)


def match_scores(frame, boxes):
    """Probability that each crop matches the description (vs look-alike alternatives)."""
    imgs = [clip_pre(Image.fromarray(cv2.cvtColor(crop(frame, b), cv2.COLOR_BGR2RGB))) for b in boxes]
    with torch.no_grad():
        f = clip_model.encode_image(torch.stack(imgs).to(DEVICE)).float()
        f = f / f.norm(dim=-1, keepdim=True)
        probs = (100 * f @ STATE["txt"].T).softmax(dim=-1)
    return probs[:, STATE["pos"]].sum(dim=1).clamp(max=1.0).tolist()


def predict_region(img, x0, y0, imgsz, n_t, cands, people):
    r = model.predict(np.ascontiguousarray(img), conf=CONF, imgsz=imgsz,
                      device=DEVICE, half=GPU, verbose=False)[0]
    for b in r.boxes:
        bx1, by1, bx2, by2 = b.xyxy[0].tolist()
        item = (float(b.conf), [bx1 + x0, by1 + y0, bx2 + x0, by2 + y0])
        (cands if int(b.cls) < n_t else people).append(item)


def detector(frame):
    """Detect on 2x2 overlapping tiles (finds far/small objects), plus the full frame on GPU."""
    H, W = frame.shape[:2]
    th, tw = int(H * TILE_FRAC), int(W * TILE_FRAC)
    n_t = len(STATE["targets"])                      # classes 0..n_t-1 = target, last = person
    cands, people = [], []
    for y0 in (0, H - th):
        for x0 in (0, W - tw):
            predict_region(frame[y0:y0 + th, x0:x0 + tw], x0, y0, TILE_IMGSZ, n_t, cands, people)
    if USE_FULL_FRAME:
        predict_region(frame, 0, 0, 1280, n_t, cands, people)

    cands.sort(key=lambda c: -c[0])
    kept = []                                        # merge duplicates (tiles overlap, prompts overlap)
    for c in cands:
        if all(iou(c[1], k[1]) < 0.5 for k in kept):
            kept.append(c)
    cands = kept[:MAX_CANDIDATES]

    if cands and STATE["use_clip"]:
        probs = match_scores(frame, [box for _, box in cands])
        objs = [(p, box) for p, (_, box) in zip(probs, cands) if p >= MIN_MATCH]
    else:
        objs = cands
    return objs, people


# ---------- tracking ----------
def lock(tr, hit, frame, people):
    score, (x1, y1, x2, y2) = hit
    tr.update(conf=score, cx=(x1 + x2) / 2, cy=(y1 + y2) / 2, w=x2 - x1, h=y2 - y1,
              active=True, misses=0, first_miss=None)
    tr["tracker"] = start_tracker(frame, tr["cx"], tr["cy"], tr["w"], tr["h"])
    tr["hits"] += 1
    tr["score_sum"] += score
    tr["person_near"] = any(near((x1, y1, x2, y2), p) for _, p in people)
    if score > tr["best"]:
        tr["best"], tr["thumb"] = score, thumb(frame, (x1, y1, x2, y2))


def track_overlap(a, b):
    """Average box overlap of two tracks over the frames they share (1 = same object)."""
    bf = {x["f"]: x for x in b["hist"]}
    vals = []
    for x in a["hist"]:
        y = bf.get(x["f"])
        if y:
            vals.append(iou((x["x1"], x["y1"], x["x2"], x["y2"]), (y["x1"], y["y1"], y["x2"], y["y2"])))
    return sum(vals) / len(vals) if vals else 0


def is_duplicate(hit, tracks):
    x1, y1, x2, y2 = hit[1]
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    for tr in tracks:
        if tr["active"] and ((cx - tr["cx"]) ** 2 + (cy - tr["cy"]) ** 2) ** 0.5 < 0.5 * max(w, h, tr["w"], tr["h"]):
            return True
    return False


def analyse(video):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width, height = int(cap.get(3)), int(cap.get(4))
    stride = max(1, round(fps / TRACK_FPS))

    scale = min(1.0, MAX_SIDE / max(width, height))   # analyse a smaller frame = faster
    inv = 1.0 / scale
    tracks, next_check, last_det, idx = [], 0.0, -99.0, -1
    while cap.grab():
        idx += 1
        if idx % stride:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        if scale < 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        t = idx / fps

        for tr in tracks:                            # fast: follow every active candidate
            if tr["active"] and tr["tracker"] is not None:
                ok, tb = tr["tracker"].update(frame)
                if ok:
                    tr["cx"], tr["cy"] = tb[0] + tb[2] / 2, tb[1] + tb[3] / 2
                else:                               # tracker lost it: let the next check re-find it
                    tr["tracker"] = None
                    next_check = min(next_check, last_det + MIN_GAP)

        if t >= next_check:                          # slow: detector + CLIP
            last_det = t
            objs, people = detector(frame)
            free = list(objs)
            for tr in [x for x in tracks if x["active"]]:
                hit = pick(free, tr["cx"], tr["cy"], tr["w"], tr["h"])
                if hit is None:
                    tr["misses"] += 1
                    if tr["first_miss"] is None:
                        tr["first_miss"] = idx
                    if tr["misses"] >= MISS_LIMIT:
                        tr["hist"] = [x for x in tr["hist"] if x["f"] <= tr["first_miss"]]
                        tr["active"] = False
                else:
                    free.remove(hit)
                    lock(tr, hit, frame, people)
            for hit in sorted(free, key=lambda o: -o[0]):
                if len(tracks) >= MAX_TRACKS or is_duplicate(hit, tracks):
                    continue
                tr = {"hits": 0, "score_sum": 0.0, "best": -1, "thumb": None, "hist": [], "tracker": None}
                lock(tr, hit, frame, people)
                tracks.append(tr)
            next_check = t + (VERIFY_EVERY_SEC if any(x["active"] for x in tracks) else SEARCH_EVERY_SEC)

        for tr in tracks:
            if tr["active"]:
                tr["hist"].append({"f": idx, "x1": (tr["cx"] - tr["w"] / 2) * inv, "y1": (tr["cy"] - tr["h"] / 2) * inv,
                                   "x2": (tr["cx"] + tr["w"] / 2) * inv, "y2": (tr["cy"] + tr["h"] / 2) * inv,
                                   "conf": round(tr["conf"], 2)})
    cap.release()

    valid = [tr for tr in tracks if tr["hits"] >= MIN_HITS and tr["hist"]]
    for tr in valid:                                 # rank: average match, trusted more with more confirmations
        tr["avg"] = tr["score_sum"] / tr["hits"]
        tr["rank_score"] = tr["avg"] * (0.5 + 0.5 * min(1.0, tr["hits"] / 4))
    top = []                                         # best first, skipping tracks that are the same object
    for tr in sorted(valid, key=lambda tr: -tr["rank_score"]):
        if all(track_overlap(tr, k) < 0.5 for k in top):
            top.append(tr)
        if len(top) >= TOP_K:
            break

    result = {"fps": fps, "width": width, "height": height, "stride": stride,
              "objects": len(valid), "hits": sum(tr["hits"] for tr in top)}
    if not top:
        result.update(outcome="NOT_DETECTED", frames=[], candidates=[], last_seen_sec=None)
        return result

    candidates, by_frame = [], {}
    for rank, tr in enumerate(top, start=1):
        for b in tr["hist"]:                         # box id = rank, so the page can filter by rank
            by_frame.setdefault(b["f"], []).append({**{k: v for k, v in b.items() if k != "f"}, "id": rank})
        activity = (not tr["active"]) and tr["person_near"]
        candidates.append({
            "rank": rank, "score": round(tr["avg"], 2), "confirmations": tr["hits"],
            "first_seen_sec": round(tr["hist"][0]["f"] / fps, 1),
            "last_seen_sec": round(tr["hist"][-1]["f"] / fps, 1),
            "outcome": "FOUND_WITH_ACTIVITY" if activity else "FOUND_NO_ACTIVITY",
            "thumb": tr["thumb"]})
    result.update(outcome=candidates[0]["outcome"], candidates=candidates,
                  frames=[{"f": f, "boxes": by_frame[f]} for f in sorted(by_frame)],
                  last_seen_sec=candidates[0]["last_seen_sec"])
    return result


@app.post("/detect")
def detect_route():
    data = request.get_json()
    video, text = data["video_path"], data["object"].strip()

    words, colors, noun, targets = parse(text)
    STATE["targets"] = targets
    STATE["use_clip"] = len(words) > 1               # a bare noun needs no description check
    prompts, pos = build_prompts(words, colors, noun)
    STATE["txt"], STATE["pos"] = encode_text(prompts), pos

    model.predictor = None
    model.set_classes(targets + ["person"])
    result = analyse(video)
    result["object"] = text
    result["understood"] = " ".join(words)
    print(f"[agent] '{text}' -> understood '{' '.join(words)}', detector prompts {targets}: "
          f"{len(result['candidates'])} ranked match(es), {result['outcome']}")
    for c in result["candidates"]:
        print(f"   #{c['rank']} score {c['score']} seen {c['first_seen_sec']}s-{c['last_seen_sec']}s")
    return jsonify(result)


if __name__ == "__main__":
    app.run(port=5001, threaded=False)