import fiftyone as fo


dataset = fo.Dataset.from_dir(
    data_path="/Volumes/T7/colonyCounter/data/fold_1/train",
    dataset_type=fo.types.COCODetectionDataset,  # Change with your type
    labels_path="/Volumes/T7/colonyCounter/data/fold_1/train/_annotations.coco.json",
    # max_samples=10,
)
view = dataset.view()

import cv2
# %%
sample = dataset.match({"filepath": {"$regex": "6242_A"}}).first()
image = cv2.imread("/Volumes/T7/colonyCounter/data/fold_1/train/6242_A.png")
h, w = image.shape[:2]

# Extract bounding boxes and labels from the sample
bboxes = []
labels = []
if sample.detections and sample.detections.detections:
    for det in sample.detections.detections:
        x, y, bw, bh = det.bounding_box
        bboxes.append([x * w, y * h, bw * w, bh * h])
        labels.append(det.label)

import albumentations as A
import numpy as np

def reflect_colonies(
    image,
    bboxes,
    num_reflections=(3, 8),
    opacity=(0.1, 0.35),
    max_offset=80,
    blur_ksize=(3, 7),
    flip_prob=0.5,
    p=0.5,
):
    """Add faint colony reflections as unlabeled distractors."""
    rng = np.random.default_rng()
    if rng.random() > p or len(bboxes) == 0:
        return image

    image = image.copy()
    h, w = image.shape[:2]
    if num_reflections is None:
        n = len(bboxes)
    else:
        n = rng.integers(num_reflections[0], num_reflections[1] + 1)
        n = min(n, len(bboxes))
    indices = rng.choice(len(bboxes), size=n, replace=False)
    angle = rng.uniform(0, 2 * np.pi)

    for idx in indices:
        x_min, y_min, bw, bh = [int(v) for v in bboxes[idx]]
        if bw < 2 or bh < 2:
            continue

        # Crop source colony
        patch = image[y_min:y_min + bh, x_min:x_min + bw].copy()

        # Apply reflection effects
        alpha = rng.uniform(opacity[0], opacity[1])
        k = rng.integers(blur_ksize[0] // 2, blur_ksize[1] // 2 + 1) * 2 + 1
        patch = cv2.GaussianBlur(patch, (k, k), 0)
        if rng.random() < flip_prob:
            patch = cv2.flip(patch, 1)
        if rng.random() < flip_prob * 0.3:
            patch = cv2.flip(patch, 0)

        # Offset placement (shared direction)
        min_dist = max(bw, bh)
        if min_dist >= max_offset:
            continue
        dist = rng.uniform(min_dist, max_offset)
        new_x = int(x_min + dist * np.cos(angle))
        new_y = int(y_min + dist * np.sin(angle))
        new_x = np.clip(new_x, 0, w - bw)
        new_y = np.clip(new_y, 0, h - bh)

        # Circular mask for the patch
        circle_mask = np.zeros((bh, bw), dtype=np.float32)
        cv2.ellipse(circle_mask, (bw // 2, bh // 2), (bw // 2, bh // 2), 0, 0, 360, alpha, -1)

        # Alpha-blend onto image using circular mask
        roi = image[new_y:new_y + bh, new_x:new_x + bw]
        if roi.shape[:2] != patch.shape[:2]:
            continue
        mask_3ch = circle_mask[:, :, np.newaxis]
        blended = (patch * mask_3ch + roi * (1.0 - mask_3ch)).astype(np.uint8)
        image[new_y:new_y + bh, new_x:new_x + bw] = blended

    return image


def edge_flare(
    src_radius=250,
    num_flare_circles_range=(3, 6),
    p=0.8,
):
    return A.OneOf(
        [
            A.RandomSunFlare(
                flare_roi=(0, 0, 0.05, 1),      # left edge
                angle_range=(0, 1),
                num_flare_circles_range=num_flare_circles_range,
                src_radius=src_radius,
                method="physics_based",
                p=1.0,
            ),
            A.RandomSunFlare(
                flare_roi=(0.95, 0, 1, 1),      # right edge
                angle_range=(0, 1),
                num_flare_circles_range=num_flare_circles_range,
                src_radius=src_radius,
                method="physics_based",
                p=1.0,
            ),
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.05),      # top edge
                angle_range=(0, 1),
                num_flare_circles_range=num_flare_circles_range,
                src_radius=src_radius,
                method="physics_based",
                p=1.0,
            ),
            A.RandomSunFlare(
                flare_roi=(0, 0.95, 1, 1),      # bottom edge
                angle_range=(0, 1),
                num_flare_circles_range=num_flare_circles_range,
                src_radius=src_radius,
                method="physics_based",
                p=1.0,
            ),
        ],
        p=p,
    )

# %%
pipeline = A.Compose(
    [
        A.RandomRotate90(p=0.75),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ChannelDropout(p=0.2),
        edge_flare(src_radius=400, num_flare_circles_range=(3, 6), p=0.1),
     ],
    bbox_params=A.BboxParams(format="coco", min_visibility=0.9, label_fields=["labels"]),
)
# Generate 20 transformed images
images = []
for i in range(20):
    transformed = pipeline(image=image, bboxes=bboxes, labels=labels)
    img = reflect_colonies(
        transformed["image"], transformed["bboxes"],
        # num_reflections=None, opacity=(0.1, 0.35), max_offset=80, p=0.1,
        num_reflections=None, opacity=(0.1, 0.35), max_offset=80, p=0.05,
    )
    for bbox, label in zip(transformed["bboxes"], transformed["labels"]):
        x_min, y_min, bw, bh = [int(v) for v in bbox]
        cv2.rectangle(img, (x_min, y_min), (x_min + bw, y_min + bh), (0, 255, 0), 2)
        cv2.putText(img, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    images.append(img)

# Arrange in a 4x5 grid
rows = []
for r in range(4):
    row_imgs = [cv2.resize(images[r * 5 + c], (w, h)) for c in range(5)]
    rows.append(np.hstack(row_imgs))
grid = np.vstack(rows)

output_path = "/tmp/transformed_preview.png"
cv2.imwrite(output_path, grid)
import subprocess
subprocess.Popen(["open", output_path])
