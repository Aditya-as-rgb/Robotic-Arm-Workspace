import cv2
import numpy as np
import time
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBStreamType, OBFormat, AlignFilter
from ultralytics import YOLO

# --- Mouse click state ---
clicked_points = []  # list of (x, y, label, distance)

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append([x, y, "", 0.0])

# Load model
model = YOLO("yolo11s.pt")

# Setup camera
pipeline = Pipeline()
config = Config()

color_profile = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR).get_default_video_stream_profile()
config.enable_stream(color_profile)

depth_profile = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR).get_default_video_stream_profile()
config.enable_stream(depth_profile)

pipeline.start(config)
align = AlignFilter(OBStreamType.COLOR_STREAM)

# FPS
fps = 0
frame_count = 0
fps_start = time.time()

cv2.namedWindow("Distance Measurement")
cv2.setMouseCallback("Distance Measurement", mouse_callback)

print("Click on any object to measure distance.")
print("Press 'c' to clear markers | 'q' to quit")

while True:
    frames = pipeline.wait_for_frames(100)
    if frames is None:
        continue

    aligned = align.process(frames)
    working_frames = aligned if aligned is not None else frames

    # --- Color ---
    color_frame = working_frames.get_color_frame()
    if color_frame is None:
        continue

    w, h = color_frame.get_width(), color_frame.get_height()
    color_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)

    fmt = color_frame.get_format()
    if fmt == OBFormat.MJPG:
        color_bgr = cv2.imdecode(color_data, cv2.IMREAD_COLOR)
    elif fmt == OBFormat.RGB:
        color_bgr = cv2.cvtColor(color_data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
    elif fmt == OBFormat.BGR:
        color_bgr = color_data.reshape(h, w, 3)
    elif fmt == OBFormat.YUYV:
        color_bgr = cv2.cvtColor(color_data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUYV)
    else:
        continue

    if color_bgr is None:
        continue

    # --- Depth ---
    depth_map = None
    depth_frame = working_frames.get_depth_frame()
    if depth_frame:
        dw, dh = depth_frame.get_width(), depth_frame.get_height()
        scale = depth_frame.get_depth_scale()
        depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(dh, dw)
        depth_map = (depth_raw * scale).astype(np.float32)  # mm
        if depth_map.shape[:2] != (h, w):
            depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_NEAREST)

    # --- YOLO ---
    results = model(color_bgr, verbose=False)[0]

    # --- Draw YOLO detections with distance ---
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]

        dist_text = ""
        if depth_map is not None:
            cx1 = max(0, x1 + (x2 - x1) // 4)
            cy1 = max(0, y1 + (y2 - y1) // 4)
            cx2 = min(w, x1 + 3 * (x2 - x1) // 4)
            cy2 = min(h, y1 + 3 * (y2 - y1) // 4)
            roi = depth_map[cy1:cy2, cx1:cx2]
            valid = roi[roi > 0]
            if valid.size > 0:
                dist_mm = np.median(valid)
                dist_text = f" {dist_mm/1000:.2f}m" if dist_mm >= 1000 else f" {dist_mm:.0f}mm"

        # Box color by distance
        if "m" in dist_text:
            try:
                d = float(dist_text.strip().replace("m",""))
                if d < 1.0:
                    box_color = (0, 0, 255)    # red  = very close
                elif d < 2.5:
                    box_color = (0, 165, 255)  # orange = medium
                else:
                    box_color = (0, 255, 0)    # green = far
            except:
                box_color = (0, 255, 0)
        else:
            box_color = (0, 255, 0)

        cv2.rectangle(color_bgr, (x1, y1), (x2, y2), box_color, 2)
        text = f"{label} {conf:.2f}{dist_text}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(color_bgr, (x1, y1 - th - 8), (x1 + tw, y1), box_color, -1)
        cv2.putText(color_bgr, text, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # --- Handle mouse click measurements ---
    if depth_map is not None:
        for pt in clicked_points:
            px, py = pt[0], pt[1]
            if 0 <= px < w and 0 <= py < h:
                # Sample 5x5 patch around click for stability
                x1c = max(0, px - 5)
                y1c = max(0, py - 5)
                x2c = min(w, px + 5)
                y2c = min(h, py + 5)
                patch = depth_map[y1c:y2c, x1c:x2c]
                valid = patch[patch > 0]

                if valid.size > 0:
                    dist_mm = np.median(valid)
                    pt[3] = dist_mm

                    # Find which YOLO label is at this point
                    pt[2] = "point"
                    for box in results.boxes:
                        bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                        if bx1 <= px <= bx2 and by1 <= py <= by2:
                            pt[2] = model.names[int(box.cls[0])]
                            break

    # --- Draw click markers ---
    for pt in clicked_points:
        px, py, lbl, dist_mm = pt
        if dist_mm > 0:
            dist_str = f"{dist_mm/1000:.2f}m" if dist_mm >= 1000 else f"{dist_mm:.0f}mm"
            marker_text = f"{lbl}: {dist_str}"
        else:
            marker_text = "no depth"

        # Crosshair
        cv2.drawMarker(color_bgr, (px, py), (255, 0, 255),
                       cv2.MARKER_CROSS, 20, 2)
        # Label background
        (tw, th), _ = cv2.getTextSize(marker_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(color_bgr, (px + 12, py - th - 6),
                      (px + 12 + tw, py + 4), (255, 0, 255), -1)
        cv2.putText(color_bgr, marker_text, (px + 12, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # --- FPS ---
    frame_count += 1
    if frame_count >= 30:
        elapsed = time.time() - fps_start
        fps = frame_count / elapsed
        frame_count = 0
        fps_start = time.time()

    inf_ms = results.speed.get('inference', 0)
    cv2.putText(color_bgr, f"FPS: {fps:.1f}  Inference: {inf_ms:.1f}ms", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(color_bgr, "Click=measure | C=clear | Q=quit", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Distance Measurement", color_bgr)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        clicked_points.clear()
        print("Markers cleared")

pipeline.stop()
cv2.destroyAllWindows()
