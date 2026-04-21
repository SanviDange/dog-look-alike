import json
import os
import math

# Path to your dataset
LABELS_DIR = "/Users/test/.cache/kagglehub/datasets/georgemartvel/dogflw/versions/1/DogFLW/train/labels"
IMAGES_DIR = "/Users/test/.cache/kagglehub/datasets/georgemartvel/dogflw/versions/1/DogFLW/train/images"
OUTPUT_FILE = "dog_vectors.json"

def dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def midpoint(a, b):
    return [(a[0]+b[0])/2, (a[1]+b[1])/2]

def extract_vector(landmarks, bounding_boxes):
    try:
        x1 = float(bounding_boxes[0])
        y1 = float(bounding_boxes[1])
        x2 = float(bounding_boxes[2])
        y2 = float(bounding_boxes[3])
    except (ValueError, IndexError):
        return None

    face_width  = x2 - x1
    face_height = y2 - y1
    if face_width == 0 or face_height == 0:
        return None

    # ── ORIGINAL 10 FEATURES ──────────────────────────────────────

    # Left eye width (outer to inner corner)
    left_eye_width = dist(landmarks[6], landmarks[0]) / face_width
    # Right eye width
    right_eye_width = dist(landmarks[1], landmarks[7]) / face_width

    # Eye openness (top to bottom / eye width) — ratio not raw
    left_eye_h  = dist(landmarks[8], landmarks[4])
    right_eye_h = dist(landmarks[9], landmarks[5])
    left_eye_openness  = left_eye_h  / (dist(landmarks[6], landmarks[0]) + 1e-6)
    right_eye_openness = right_eye_h / (dist(landmarks[1], landmarks[7]) + 1e-6)

    # Eye centers
    left_eye_center  = midpoint(landmarks[0], landmarks[6])
    right_eye_center = midpoint(landmarks[1], landmarks[7])

    # Interocular distance (center to center)
    interocular = dist(left_eye_center, right_eye_center) / face_width

    # Nose width
    nose_width = dist(landmarks[32], landmarks[35]) / face_width

    # Mouth width
    mouth_width = dist(landmarks[38], landmarks[41]) / face_width

    # Face roundness
    face_roundness = face_height / face_width

    # Brow heights — normalized by face height, measured from top of bbox
    left_brow_height  = (landmarks[12][1] - y1) / face_height
    right_brow_height = (landmarks[13][1] - y1) / face_height

    # ── NEW FEATURES ─────────────────────────────────────────────

    # Brow width (distance between the two brow points) / face width
    brow_width = dist(landmarks[12], landmarks[13]) / face_width

    # Nose to mouth distance / face height
    # nose tip = midpoint of nostril points, mouth top = midpoint of mouth corners
    nose_tip    = midpoint(landmarks[32], landmarks[35])
    mouth_mid   = midpoint(landmarks[38], landmarks[41])
    nose_to_mouth = dist(nose_tip, mouth_mid) / face_height

    # Eye to nose distance / face height
    # midpoint between both eye centers down to midpoint of nostrils
    eyes_mid    = midpoint(left_eye_center, right_eye_center)
    eye_to_nose = dist(eyes_mid, nose_tip) / face_height

    # Mouth vertical position in face (how low is the mouth?)
    mouth_y_ratio = (mouth_mid[1] - y1) / face_height

    # Eye vertical position in face (how high are the eyes?)
    eye_y_ratio = (eyes_mid[1] - y1) / face_height

    # Nose width to mouth width ratio (pure shape ratio, no normalization needed)
    nose_to_mouth_width = nose_width / (mouth_width + 1e-6)

    # Eye width to interocular ratio — are eyes wide relative to spacing?
    eye_spacing_ratio = (left_eye_width + right_eye_width) / 2 / (interocular + 1e-6)

    # Face upper third: brow to eye distance / face height
    brow_to_eye_left  = abs(landmarks[12][1] - left_eye_center[1])  / face_height
    brow_to_eye_right = abs(landmarks[13][1] - right_eye_center[1]) / face_height
    brow_to_eye = (brow_to_eye_left + brow_to_eye_right) / 2

    return [
        # Original 10
        round(left_eye_width,       4),  # 0
        round(right_eye_width,      4),  # 1
        round(left_eye_openness,    4),  # 2
        round(right_eye_openness,   4),  # 3
        round(interocular,          4),  # 4
        round(nose_width,           4),  # 5
        round(mouth_width,          4),  # 6
        round(face_roundness,       4),  # 7
        round(left_brow_height,     4),  # 8
        round(right_brow_height,    4),  # 9
        # New 8
        round(brow_width,           4),  # 10
        round(nose_to_mouth,        4),  # 11
        round(eye_to_nose,          4),  # 12
        round(mouth_y_ratio,        4),  # 13
        round(eye_y_ratio,          4),  # 14
        round(nose_to_mouth_width,  4),  # 15
        round(eye_spacing_ratio,    4),  # 16
        round(brow_to_eye,          4),  # 17
    ]

# ── PROCESS ALL LABEL FILES ───────────────────────────────────

results = []
skipped = 0

for filename in os.listdir(LABELS_DIR):
    if not filename.endswith(".json"):
        continue

    label_path = os.path.join(LABELS_DIR, filename)
    image_name = filename.replace(".json", ".png")
    image_path = os.path.join(IMAGES_DIR, image_name)

    if not os.path.exists(image_path):
        skipped += 1
        continue

    with open(label_path) as f:
        data = json.load(f)

    landmarks     = data["landmarks"]
    bounding_boxes = data["bounding_boxes"]

    vector = extract_vector(landmarks, bounding_boxes)

    if vector is None:
        skipped += 1
        continue

    results.append({
        "image": image_name,
        "vector": vector
    })

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f)

print(f"✅ Done! {len(results)} dogs processed, {skipped} skipped.")
print(f"Saved to {OUTPUT_FILE}")
print(f"Vector size: 18 features per dog")