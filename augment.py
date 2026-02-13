"""Apply augmentation pipeline to a COCO-format dataset, producing original + augmented images."""

import json
import random
import shutil
from pathlib import Path
from typing import Optional

import albumentations as A
import cv2
import numpy as np
import typer
from tqdm import tqdm

app = typer.Typer()


# ---------------------------------------------------------------------------
# Augmentation helpers (copied from main.py to keep this self-contained)
# ---------------------------------------------------------------------------

def imread_flatten_alpha(path: str) -> np.ndarray | None:
    """Read an image, compositing any alpha channel onto a black background."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        bgr = img[:, :, :3].astype(np.float32)
        img = (bgr * alpha).astype(np.uint8)
    return img


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

        patch = image[y_min : y_min + bh, x_min : x_min + bw].copy()

        alpha = rng.uniform(opacity[0], opacity[1])
        k = rng.integers(blur_ksize[0] // 2, blur_ksize[1] // 2 + 1) * 2 + 1
        patch = cv2.GaussianBlur(patch, (k, k), 0)
        if rng.random() < flip_prob:
            patch = cv2.flip(patch, 1)
        if rng.random() < flip_prob * 0.3:
            patch = cv2.flip(patch, 0)

        min_dist = max(bw, bh)
        if min_dist >= max_offset:
            continue
        dist = rng.uniform(min_dist, max_offset)
        new_x = int(x_min + dist * np.cos(angle))
        new_y = int(y_min + dist * np.sin(angle))
        new_x = np.clip(new_x, 0, w - bw)
        new_y = np.clip(new_y, 0, h - bh)

        circle_mask = np.zeros((bh, bw), dtype=np.float32)
        cv2.ellipse(circle_mask, (bw // 2, bh // 2), (bw // 2, bh // 2), 0, 0, 360, alpha, -1)

        roi = image[new_y : new_y + bh, new_x : new_x + bw]
        if roi.shape[:2] != patch.shape[:2]:
            continue
        mask_3ch = circle_mask[:, :, np.newaxis]
        blended = (patch * mask_3ch + roi * (1.0 - mask_3ch)).astype(np.uint8)
        image[new_y : new_y + bh, new_x : new_x + bw] = blended

    return image


def edge_flare(src_radius=250, num_flare_circles_range=(3, 6), p=0.8):
    return A.OneOf(
        [
            A.RandomSunFlare(
                flare_roi=(0, 0, 0.05, 1),
                angle_range=(0, 1),
                num_flare_circles_range=num_flare_circles_range,
                src_radius=src_radius,
                method="physics_based",
                p=1.0,
            ),
            A.RandomSunFlare(
                flare_roi=(0.95, 0, 1, 1),
                angle_range=(0, 1),
                num_flare_circles_range=num_flare_circles_range,
                src_radius=src_radius,
                method="physics_based",
                p=1.0,
            ),
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.05),
                angle_range=(0, 1),
                num_flare_circles_range=num_flare_circles_range,
                src_radius=src_radius,
                method="physics_based",
                p=1.0,
            ),
            A.RandomSunFlare(
                flare_roi=(0, 0.95, 1, 1),
                angle_range=(0, 1),
                num_flare_circles_range=num_flare_circles_range,
                src_radius=src_radius,
                method="physics_based",
                p=1.0,
            ),
        ],
        p=p,
    )


def build_pipeline(extra: bool = False):
    transforms = [
        # Geometric
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=180, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.75),
        # Photometric
        A.RandomBrightnessContrast(p=0.3),
        A.HueSaturationValue(p=0.2),
        # Noise & artifacts
        A.ISONoise(p=0.2),
        A.ImageCompression(p=0.1),
        A.ChannelDropout(p=0.1),
    ]
    if extra:
        transforms.insert(3, A.Perspective(scale=(0.02, 0.05), pad_mode=cv2.BORDER_CONSTANT, pad_val=0, p=0.05))
        transforms.append(A.Defocus(p=0.1, alias_blur=[0.02, 0.1], radius=[1, 2]))
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(format="coco", min_visibility=0.05, label_fields=["labels"]),
    )


# ---------------------------------------------------------------------------
# COCO dataset augmentation
# ---------------------------------------------------------------------------


def _load_coco_dataset(dir_path: Path):
    """Load a COCO dataset directory. Returns (coco_dict, anns_by_image, cat_id_to_name).

    Supports two formats:
    - Standard COCO: _annotations.coco.json
    - Per-image JSON (AGAR-style): {id}.json with {classes, labels: [{class, x, y, width, height}]}
      All classes are mapped to a single "colony" category.
    """
    ann_path = dir_path / "_annotations.coco.json"
    if ann_path.exists():
        with open(ann_path) as f:
            coco = json.load(f)
        cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
        anns_by_image: dict[int, list[dict]] = {}
        for ann in coco["annotations"]:
            anns_by_image.setdefault(ann["image_id"], []).append(ann)
        return coco, anns_by_image, cat_id_to_name

    # Fall back to per-image JSON format (AGAR-style)
    # Only scan JSON metadata — no imread, dimensions resolved lazily in _process_images
    json_files = sorted(dir_path.glob("*.json"))
    if not json_files:
        typer.echo(f"No annotations found in {dir_path}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Loading per-image JSON annotations from {dir_path}")
    categories = [{"id": 0, "name": "colony", "supercategory": "colony"}]
    images = []
    annotations = []
    anns_by_image = {}
    ann_id = 0

    for jf in json_files:
        # Find matching image file
        img_name = None
        for ext in (".jpg", ".png"):
            candidate = jf.with_suffix(ext)
            if candidate.exists():
                img_name = candidate.name
                break
        if img_name is None:
            continue

        with open(jf) as f:
            data = json.load(f)

        image_id = len(images)
        images.append({
            "id": image_id,
            "file_name": img_name,
            "width": 0,
            "height": 0,
        })

        img_anns = []
        for label in data.get("labels", []):
            ann = {
                "id": ann_id,
                "image_id": image_id,
                "category_id": 0,
                "bbox": [label["x"], label["y"], label["width"], label["height"]],
                "area": label["width"] * label["height"],
                "iscrowd": 0,
            }
            img_anns.append(ann)
            annotations.append(ann)
            ann_id += 1
        anns_by_image[image_id] = img_anns

    coco = {"categories": categories, "images": images, "annotations": annotations}
    cat_id_to_name = {0: "colony"}
    typer.echo(f"Loaded {len(images)} images with {len(annotations)} annotations")
    return coco, anns_by_image, cat_id_to_name


def _process_images(
    img_entries: list[dict],
    src_dir: Path,
    output: Path,
    anns_by_image: dict[int, list[dict]],
    cat_id_to_name: dict[int, str],
    name_to_cat_id: dict[str, int],
    pipeline,
    copies: int,
    next_image_id: int,
    next_ann_id: int,
    new_images: list[dict],
    new_annotations: list[dict],
    copy_original: bool,
    extra: bool = False,
    file_prefix: str = "",
    desc: str = "Augmenting",
):
    """Process a list of COCO image entries: optionally copy originals, then augment."""
    for img_entry in tqdm(img_entries, desc=desc):
        file_name = img_entry["file_name"]
        src_path = src_dir / file_name
        if not src_path.exists():
            continue

        out_file_name = f"{file_prefix}{file_name}" if file_prefix else file_name

        # Read image early so both original copy and augmentation use flattened alpha
        image = imread_flatten_alpha(str(src_path))
        if image is None:
            continue

        # --- Copy original ---
        if copy_original:
            cv2.imwrite(str(output / out_file_name), image)
            orig_entry = {**img_entry, "id": next_image_id, "file_name": out_file_name}
            new_images.append(orig_entry)
            for ann in anns_by_image.get(img_entry["id"], []):
                new_annotations.append(
                    {
                        **ann,
                        "id": next_ann_id,
                        "image_id": next_image_id,
                        "category_id": name_to_cat_id[cat_id_to_name[ann["category_id"]]],
                    }
                )
                next_ann_id += 1
            next_image_id += 1

        # Resolve dimensions lazily (needed for per-image JSON datasets)
        img_h, img_w = image.shape[:2]
        if img_entry["width"] == 0:
            img_entry["width"] = img_w
            img_entry["height"] = img_h

        img_anns = anns_by_image.get(img_entry["id"], [])
        bboxes = []
        labels = []
        for ann in img_anns:
            x = max(0, min(ann["bbox"][0], img_w))
            y = max(0, min(ann["bbox"][1], img_h))
            w = min(ann["bbox"][2], img_w - x)
            h = min(ann["bbox"][3], img_h - y)
            if w > 0 and h > 0:
                bboxes.append([x, y, w, h])
                labels.append(cat_id_to_name[ann["category_id"]])

        stem = Path(out_file_name).stem
        suffix = Path(out_file_name).suffix or ".png"

        if extra:
            flare = edge_flare(
                src_radius=max(30, int(min(img_h, img_w) * 0.4)),
                num_flare_circles_range=(3, 6),
                p=0.1,
            )

        for i in range(copies):
            transformed = pipeline(image=image, bboxes=bboxes, labels=labels)
            if extra:
                transformed["image"] = flare(image=transformed["image"])["image"]
                aug_img = reflect_colonies(
                    transformed["image"],
                    transformed["bboxes"],
                    num_reflections=None,
                    opacity=(0.1, 0.35),
                    max_offset=80,
                    p=1,
                )
            else:
                aug_img = transformed["image"]

            aug_name = f"{stem}_aug_{i}{suffix}"
            cv2.imwrite(str(output / aug_name), aug_img)

            new_images.append(
                {
                    "id": next_image_id,
                    "file_name": aug_name,
                    "width": img_w,
                    "height": img_h,
                }
            )

            for bbox, label in zip(transformed["bboxes"], transformed["labels"]):
                x, y, w, h = bbox
                new_annotations.append(
                    {
                        "id": next_ann_id,
                        "image_id": next_image_id,
                        "category_id": name_to_cat_id[label],
                        "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                        "area": round(w * h, 2),
                        "iscrowd": 0,
                    }
                )
                next_ann_id += 1

            next_image_id += 1

    return next_image_id, next_ann_id


def _parse_sample_arg(value: str) -> tuple[Path, int]:
    """Parse a --sample value like '/path/to/dataset:10'."""
    sep = value.rfind(":")
    if sep == -1:
        typer.echo("--sample must be in the format PATH:NUM", err=True)
        raise typer.Exit(1)
    path = Path(value[:sep])
    try:
        n = int(value[sep + 1 :])
    except ValueError:
        typer.echo(f"Invalid sample count: {value[sep + 1:]}", err=True)
        raise typer.Exit(1)
    return path, n


@app.command()
def augment(
    input: Path = typer.Option(..., help="Directory with images + _annotations.coco.json"),
    output: Path = typer.Option(..., help="Output directory for augmented dataset"),
    copies: int = typer.Option(5, help="Number of augmented copies per image"),
    sample: Optional[list[str]] = typer.Option(
        None, help="Sample from another dataset: PATH:NUM (repeatable)"
    ),
    test: bool = typer.Option(False, help="Test mode: only augment 4 random images"),
    extra: bool = typer.Option(False, help="Enable extra augmentations: reflect_colonies, perspective, edge_flare, defocus"),
):
    """Augment a COCO dataset: copy originals and generate augmented copies."""
    coco, anns_by_image, cat_id_to_name = _load_coco_dataset(input)

    output.mkdir(parents=True, exist_ok=True)

    name_to_cat_id = {c["name"]: c["id"] for c in coco["categories"]}
    pipeline = build_pipeline(extra=extra)

    new_images: list[dict] = []
    new_annotations: list[dict] = []
    next_image_id = 0
    next_ann_id = 0

    # --- Primary dataset ---
    primary_entries = coco["images"]
    if test:
        primary_entries = random.sample(primary_entries, min(4, len(primary_entries)))
        typer.echo("Test mode: processing 4 random images")
    next_image_id, next_ann_id = _process_images(
        img_entries=primary_entries,
        src_dir=input,
        output=output,
        anns_by_image=anns_by_image,
        cat_id_to_name=cat_id_to_name,
        name_to_cat_id=name_to_cat_id,
        pipeline=pipeline,
        copies=copies,
        next_image_id=next_image_id,
        next_ann_id=next_ann_id,
        new_images=new_images,
        new_annotations=new_annotations,
        copy_original=True,
        extra=extra,
        desc="Augmenting (primary)",
    )

    n_primary = len(new_images)

    # --- Sampled datasets ---
    if sample:
        for sample_spec in sample:
            sample_dir, sample_n = _parse_sample_arg(sample_spec)
            s_coco, s_anns, s_cat_id_to_name = _load_coco_dataset(sample_dir)

            # Validate that sampled category names exist in primary dataset
            for cat in s_coco["categories"]:
                if cat["name"] not in name_to_cat_id:
                    typer.echo(
                        f"Category '{cat['name']}' from {sample_dir} not found in primary dataset",
                        err=True,
                    )
                    raise typer.Exit(1)

            # Randomly sample images
            all_images = s_coco["images"]
            k = min(sample_n, len(all_images))
            sampled = random.sample(all_images, k)
            typer.echo(f"Sampled {k} images from {sample_dir}")

            # Use a prefix to avoid filename collisions
            prefix = f"sample_{sample_dir.stem}_"

            next_image_id, next_ann_id = _process_images(
                img_entries=sampled,
                src_dir=sample_dir,
                output=output,
                anns_by_image=s_anns,
                cat_id_to_name=s_cat_id_to_name,
                name_to_cat_id=name_to_cat_id,
                pipeline=pipeline,
                copies=1,
                next_image_id=next_image_id,
                next_ann_id=next_ann_id,
                new_images=new_images,
                new_annotations=new_annotations,
                copy_original=False,
                extra=extra,
                file_prefix=prefix,
                desc=f"Augmenting (sampled from {sample_dir.name})",
            )

    out_coco = {
        "categories": coco["categories"],
        "images": new_images,
        "annotations": new_annotations,
    }
    for key in coco:
        if key not in out_coco:
            out_coco[key] = coco[key]

    with open(output / "_annotations.coco.json", "w") as f:
        json.dump(out_coco, f)

    n_total = len(new_images)
    n_sampled_orig = n_total - n_primary - (n_total - n_primary) * copies // (copies + 1)
    typer.echo(
        f"Done: {n_primary} primary (orig+aug) + {n_total - n_primary} sampled (orig+aug) = {n_total} total images"
    )


if __name__ == "__main__":
    app()
