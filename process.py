import json
import os
import math

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

    # ── STRUCTURAL FEATURES (18) ─────────────────────────────────

    left_eye_width  = dist(landmarks[6], landmarks[0]) / face_width
    right_eye_width = dist(landmarks[1], landmarks[7]) / face_width

    left_eye_h  = dist(landmarks[8], landmarks[4])
    right_eye_h = dist(landmarks[9], landmarks[5])
    left_eye_openness  = left_eye_h  / (dist(landmarks[6], landmarks[0]) + 1e-6)
    right_eye_openness = right_eye_h / (dist(landmarks[1], landmarks[7]) + 1e-6)

    left_eye_center  = midpoint(landmarks[0], landmarks[6])
    right_eye_center = midpoint(landmarks[1], landmarks[7])
    interocular = dist(left_eye_center, right_eye_center) / face_width

    nose_width  = dist(landmarks[32], landmarks[35]) / face_width
    mouth_width = dist(landmarks[38], landmarks[41]) / face_width
    face_roundness = face_height / face_width

    left_brow_height  = (landmarks[12][1] - y1) / face_height
    right_brow_height = (landmarks[13][1] - y1) / face_height
    brow_width = dist(landmarks[12], landmarks[13]) / face_width

    nose_tip  = midpoint(landmarks[32], landmarks[35])
    mouth_mid = midpoint(landmarks[38], landmarks[41])
    nose_to_mouth = dist(nose_tip, mouth_mid) / face_height

    eyes_mid    = midpoint(left_eye_center, right_eye_center)
    eye_to_nose = dist(eyes_mid, nose_tip) / face_height

    mouth_y_ratio = (mouth_mid[1] - y1) / face_height
    eye_y_ratio   = (eyes_mid[1]  - y1) / face_height

    nose_to_mouth_width = nose_width / (mouth_width + 1e-6)
    avg_eye_width = (left_eye_width + right_eye_width) / 2
    eye_spacing_ratio = avg_eye_width / (interocular + 1e-6)

    brow_to_eye_left  = abs(landmarks[12][1] - left_eye_center[1])  / face_height
    brow_to_eye_right = abs(landmarks[13][1] - right_eye_center[1]) / face_height
    brow_to_eye = (brow_to_eye_left + brow_to_eye_right) / 2

    # ── EXPRESSION FEATURES (5) ──────────────────────────────────

    # Mouth openness — vertical gap, landmark 38=top lip, 45=bottom lip
    mouth_open = dist(landmarks[38], landmarks[45]) / face_height

    # Mouth corner raise — corners vs center (smile detection)
    mouth_center_y = mouth_mid[1]
    mouth_corner_raise = (mouth_center_y - (landmarks[39][1] + landmarks[40][1]) / 2) / face_height

    # Brow raise — how far brows are above eyes
    left_brow_raise  = (left_eye_center[1]  - landmarks[12][1]) / face_height
    right_brow_raise = (right_eye_center[1] - landmarks[13][1]) / face_height
    brow_raise = (left_brow_raise + right_brow_raise) / 2

    # Eye squeeze — raw eye height / face height (scrunched vs wide open)
    left_eye_squeeze  = left_eye_h  / face_height
    right_eye_squeeze = right_eye_h / face_height

    return [
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
        round(brow_width,           4),  # 10
        round(nose_to_mouth,        4),  # 11
        round(eye_to_nose,          4),  # 12
        round(mouth_y_ratio,        4),  # 13
        round(eye_y_ratio,          4),  # 14
        round(nose_to_mouth_width,  4),  # 15
        round(eye_spacing_ratio,    4),  # 16
        round(brow_to_eye,          4),  # 17
        round(mouth_open,           4),  # 18
        round(mouth_corner_raise,   4),  # 19
        round(brow_raise,           4),  # 20
        round(left_eye_squeeze,     4),  # 21
        round(right_eye_squeeze,    4),  # 22
    ]

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

    landmarks      = data["landmarks"]
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

print(f"Done! {len(results)} dogs processed, {skipped} skipped.")
print(f"Saved to {OUTPUT_FILE}")
print(f"Vector size: 23 features per dog")