import json
import os
import math

# Path to your dataset
LABELS_DIR = "/Users/test/.cache/kagglehub/datasets/georgemartvel/dogflw/versions/1/DogFLW/train/labels"
IMAGES_DIR = "/Users/test/.cache/kagglehub/datasets/georgemartvel/dogflw/versions/1/DogFLW/train/images"
OUTPUT_FILE = "dog_vectors.json"

def extract_vector(landmarks, bounding_boxes):
    # Bounding box
    try:
        x1 = float(bounding_boxes[0])
        y1 = float(bounding_boxes[1])
        x2 = float(bounding_boxes[2])
        y2 = float(bounding_boxes[3])
    except (ValueError, IndexError):
        return None
    
    face_width = x2 - x1
    face_height = y2 - y1

    if face_width == 0 or face_height == 0:
        return None

    # Key landmark indices based on DogFLW scheme
    # Left eye area: 0, 2, 4, 6, 8, 10
    # Right eye area: 1, 3, 5, 7, 9, 11
    # Nose: 24, 25, 32, 35
    # Mouth: 38, 41
    # Brows: 12, 13

    def dist(a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    # Left eye width (outer to inner corner)
    left_eye_width = dist(landmarks[6], landmarks[0]) / face_width

    # Right eye width
    right_eye_width = dist(landmarks[1], landmarks[7]) / face_width

    # Left eye openness (top to bottom)
    left_eye_openness = dist(landmarks[8], landmarks[4]) / face_width

    # Right eye openness
    right_eye_openness = dist(landmarks[9], landmarks[5]) / face_width

    # Eye center positions
    left_eye_center = [(landmarks[0][0] + landmarks[6][0]) / 2,
                       (landmarks[0][1] + landmarks[6][1]) / 2]
    right_eye_center = [(landmarks[1][0] + landmarks[7][0]) / 2,
                        (landmarks[1][1] + landmarks[7][1]) / 2]

    # Interocular distance (eye to eye)
    interocular = dist(left_eye_center, right_eye_center) / face_width

    # Nose width
    nose_width = dist(landmarks[32], landmarks[35]) / face_width

    # Mouth width
    mouth_width = dist(landmarks[38], landmarks[41]) / face_width

    # Face roundness
    face_roundness = face_height / face_width

    # Brow height (relative to face)
    left_brow_height = (landmarks[12][1] - y1) / face_height
    right_brow_height = (landmarks[13][1] - y1) / face_height

    return [
        round(left_eye_width, 4),
        round(right_eye_width, 4),
        round(left_eye_openness, 4),
        round(right_eye_openness, 4),
        round(interocular, 4),
        round(nose_width, 4),
        round(mouth_width, 4),
        round(face_roundness, 4),
        round(left_brow_height, 4),
        round(right_brow_height, 4),
    ]

# Process all label files
results = []
skipped = 0

for filename in os.listdir(LABELS_DIR):
    if not filename.endswith(".json"):
        continue

    label_path = os.path.join(LABELS_DIR, filename)
    image_name = filename.replace(".json", ".png")
    image_path = os.path.join(IMAGES_DIR, image_name)

    # Skip if image doesn't exist
    if not os.path.exists(image_path):
        skipped += 1
        continue

    with open(label_path) as f:
        data = json.load(f)

    landmarks = data["landmarks"]
    bounding_boxes = data["bounding_boxes"]

    vector = extract_vector(landmarks, bounding_boxes)

    if vector is None:
        skipped += 1
        continue

    results.append({
        "image": image_name,
        "vector": vector
    })

# Save output
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f)

print(f"✅ Done! {len(results)} dogs processed, {skipped} skipped.")
print(f"Saved to {OUTPUT_FILE}")