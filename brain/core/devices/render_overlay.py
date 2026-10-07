# runs on PC, on demand, for review — render_overlay.py
import cv2
import json
import bisect

def render_overlay(video_path: str, log_path: str, output_path: str):
    detections_by_time = []
    with open(log_path) as f:
        for line in f:
            entry = json.loads(line)
            detections_by_time.append(entry)
    timestamps = [d["t"] for d in detections_by_time]

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps,
                           (int(cap.get(3)), int(cap.get(4))))

    start_time = timestamps[0] if timestamps else 0
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_time = start_time + (frame_idx / fps)
        idx = bisect.bisect_left(timestamps, frame_time)
        idx = min(idx, len(detections_by_time) - 1)
        for det in detections_by_time[idx]["detections"]:
            x, y, w, h = det["x"], det["y"], det["w"], det["h"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{det['label']} {det['conf']:.2f}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()