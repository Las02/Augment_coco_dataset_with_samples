# augment-data

Data augmentation toolkit for colony counter detection models. Takes COCO-format detection datasets, applies geometric and photometric augmentations plus domain-specific colony reflection distractors, and outputs new COCO datasets with original + augmented images.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

### Make targets

```bash
make augment              # Full augmentation run (default: 5 copies per image)
make test                 # Quick test: 4 random images to /tmp/augmented_test
make clean                # Remove test output
make augment COPIES=3     # Override copy count
```

### CLI

```bash
uv run python augment.py --copies 5 --input /path/to/dataset --output /path/to/output
```

Options:

| Flag | Description |
|---|---|
| `--input PATH` | Directory with images + `_annotations.coco.json` |
| `--output PATH` | Output directory for augmented dataset |
| `--copies N` | Number of augmented copies per image (default: 5) |
| `--sample PATH:NUM` | Sample NUM images from another dataset (repeatable) |
| `--test` | Test mode: only process 4 random images |

Sampled images receive 1 augmented copy (no original kept) and are prefixed with `sample_{stem}_` to avoid filename collisions.

## Augmentation pipeline

Built with [Albumentations](https://albumentations.ai/) (`build_pipeline()`):

- Random 90-degree rotation (p=0.75)
- Horizontal flip (p=0.5)
- Vertical flip (p=0.5)
- Channel dropout (p=0.2)
- Edge flare / sun flare (p=0.1)

Bounding boxes are tracked through all transforms with COCO format `[x, y, w, h]` and a minimum visibility threshold of 0.9.

### Colony reflections

A post-pipeline step (`reflect_colonies()`) alpha-blends faint, blurred copies of existing colony patches at random offsets to act as unlabeled distractors, helping the model learn to distinguish real colonies from artifacts.

## Data format

Input and output directories contain images alongside a `_annotations.coco.json` file. Two input formats are supported:

- **Standard COCO**: single `_annotations.coco.json` with all image and annotation entries
- **Per-image JSON (AGAR-style)**: one `{id}.json` per image with `{classes, labels: [{class, x, y, width, height}]}` -- all classes are mapped to a single "colony" category

Output always uses standard COCO format with fresh sequential IDs starting from 0.

## Project structure

| File | Description |
|---|---|
| `augment.py` | Production CLI (self-contained) |
| `main.py` | Scratch/notebook script for prototyping with FiftyOne |
| `Makefile` | Convenience targets for common workflows |

## Dependencies

- `albumentations==1.4.24` (pinned -- API may differ across versions)
- `opencv-python` -- image I/O and reflection blending
- `typer` -- CLI framework
- `tqdm` -- progress bars
- `fiftyone` -- dataset visualization (used by `main.py`)
