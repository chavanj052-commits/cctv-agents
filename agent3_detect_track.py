"""
Agent 3 - Object Detection & Tracking
Usage:  python agent3_detect_track.py footage.mp4 "black backpack"
Install: pip install ultralytics opencv-python

Outcomes:
  FOUND_WITH_ACTIVITY  -> object seen, then disappeared while a person was next to it (clip saved)
  FOUND_NO_ACTIVITY    -> object seen but no clear taking event (last-seen frame saved)
  NOT_DETECTED         -> object not confidently detected (send full feed for manual review)
"""
import sys, json, cv2
from ultralytics import YOLOWorld

CONF = 0.30        # detection confidence threshold
MIN_HITS = 5       # frames the object must be seen to count as "detected"
GONE_SEC = 2       # object missing this long at the end => it "disappeared"


def near(obj, person, pad=0.5):
    """True if the person's box touches the object's box (expanded by pad)."""
    w, h = obj[2] - obj[0], obj[3] - obj[1]
    return not (person[2] < obj[0] - pad * w or person[0] > obj[2] + pad * w or
                person[3] < obj[1] - pad * h or person[1] > obj[3] + pad * h)


def save_clip(video, start, end, path, fps):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    w, h = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(end - start):
        ok, frame = cap.read()
        if not ok:
            break
        out.write(frame)
    cap.release(); out.release()


def run(video, name):
    fps = cv2.VideoCapture(video).get(cv2.CAP_PROP_FPS) or 25
    model = YOLOWorld("yolov8s-worldv2.pt")      # open-vocabulary: any object name works
    model.set_classes([name, "person"])          # class 0 = target, class 1 = person

    hits, total, last_idx, last_img, person_near = 0, 0, None, None, False

    for i, r in enumerate(model.track(video, stream=True, persist=True, conf=CONF, verbose=False)):
        total = i + 1
        objs = [b for b in r.boxes if int(b.cls) == 0]
        people = [b.xyxy[0].tolist() for b in r.boxes if int(b.cls) == 1]
        if objs:
            best = max(objs, key=lambda b: float(b.conf))
            box = best.xyxy[0].tolist()
            hits += 1
            last_idx, last_img = i, r.plot()     # annotated frame with box + track id
            person_near = any(near(box, p) for p in people)

    if hits < MIN_HITS:
        return {"outcome": "NOT_DETECTED", "object": name}

    disappeared = (total - 1 - last_idx) > GONE_SEC * fps
    if disappeared and person_near:
        start, end = max(0, last_idx - int(3 * fps)), min(total, last_idx + int(2 * fps))
        save_clip(video, start, end, "activity_clip.mp4", fps)
        return {"outcome": "FOUND_WITH_ACTIVITY", "object": name,
                "last_seen_sec": round(last_idx / fps, 1), "clip": "activity_clip.mp4"}

    cv2.imwrite("last_seen.jpg", last_img)
    return {"outcome": "FOUND_NO_ACTIVITY", "object": name,
            "last_seen_sec": round(last_idx / fps, 1), "frame": "last_seen.jpg"}


if __name__ == "__main__":
    print(json.dumps(run(sys.argv[1], sys.argv[2]), indent=2))