# augment-data

Data augmentation toolkit for colony counter detection models. Takes COCO-format detection datasets, applies geometric and photometric augmentations plus domain-specific colony reflection distractors, and outputs new COCO datasets with original + augmented images.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

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
| `--extra` | Enable extra augmentations (perspective, defocus, edge flare, colony reflections) |

Sampled images receive 1 augmented copy (no original kept) and are prefixed with `sample_{stem}_` to avoid filename collisions.

Example:
```
uv run augment.py --test --copies 8 --sample ../data/AGAR_DATASET_COCO_V2/fold_1/train:10 --input ../data/fold_1/train/ --output delme
```

## Augmentation pipeline

Built with [Albumentations](https://albumentations.ai/) (`build_pipeline()`):

**Base pipeline** (always active):
- Horizontal flip (p=0.5), Vertical flip (p=0.5)
- Arbitrary rotation up to 180° with black border fill (p=0.75)
- RandomBrightnessContrast (p=0.3), HueSaturationValue (p=0.2)
- ISONoise (p=0.2), ImageCompression (p=0.1), ChannelDropout (p=0.1)

**Extra augmentations** (`--extra` flag):
- Perspective warp (p=0.05)
- Defocus blur (p=0.1)
- Edge flare / sun flare (p=0.1)
- Colony reflections (`reflect_colonies()`)

Bounding boxes are tracked through all transforms with COCO format `[x, y, w, h]` and a minimum visibility threshold of 0.05.

### Colony reflections

A post-pipeline step (`reflect_colonies()`, enabled with `--extra`) alpha-blends faint, blurred copies of existing colony patches at random offsets to act as unlabeled distractors, helping the model learn to distinguish real colonies from artifacts.

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
