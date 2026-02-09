import fiftyone as fo


dataset = fo.Dataset.from_dir(
    data_path="/Volumes/T7/colonyCounter/data/fold_1/train",
    dataset_type=fo.types.COCODetectionDataset,  # Change with your type
    labels_path="/Volumes/T7/colonyCounter/data/fold_1/train/_annotations.coco.json",
    max_samples=10,
)
view = dataset.view()

import cv2

sample = view.take(1).first()
image = cv2.imread(sample.filepath)
h, w = image.shape[:2]

# Extract bounding boxes and labels from the sample
bboxes = []
labels = []
if sample.detections and sample.detections.detections:
    for det in sample.detections.detections:
        x, y, bw, bh = det.bounding_box
        bboxes.append([x * w, y * h, bw * w, bh * h])
        labels.append(det.label)

# %%
import albumentations as A
import numpy as np

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
        A.SquareSymmetry(p=0.75),
        A.ChannelDropout(p=0.2),
        edge_flare(src_radius=400, num_flare_circles_range=(3, 6), p=0.8),
     ],
    bbox_params=A.BboxParams(format="coco", min_visibility=0.9, label_fields=["labels"]),
)
# Generate 20 transformed images
images = []
for i in range(20):
    transformed = pipeline(image=image, bboxes=bboxes, labels=labels)
    img = transformed["image"].copy()
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
