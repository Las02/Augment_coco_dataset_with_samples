# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Data augmentation toolkit for colony counter detection models. Takes COCO-format detection datasets, applies geometric/photometric augmentations plus domain-specific colony reflection distractors, and outputs new COCO datasets with original + augmented images.

## Commands

All commands use `uv` as the package runner.

```bash
make augment              # Full augmentation run (default: 5 copies per image)
make test                 # Quick test: 4 random images to /tmp/augmented_test
make clean                # Remove test output
make augment COPIES=3     # Override copy count
```

Direct CLI usage:
```bash
uv run python augment.py --copies 5 --input /path/to/dataset --output /path/to/output
uv run python augment.py --input ... --output ... --sample /path/to/other:20  # sample 20 images from another dataset
uv run python augment.py --input ... --output ... --test                      # test mode: 4 random images only
```

## Architecture

- **`augment.py`** — Production CLI (Typer). Self-contained: copies augmentation functions from main.py rather than importing. Key components:
  - `build_pipeline(extra)`: Albumentations Compose chain (flips, rotation, brightness/contrast, HSV, noise, compression, channel dropout; with `--extra`: perspective, defocus, edge flare) with COCO bbox params (`min_visibility=0.05`)
  - `reflect_colonies()`: Post-pipeline step that alpha-blends faint copies of existing colonies at random offsets as unlabeled distractors
  - `_process_images()`: Core loop handling original copying + N augmented copies, with ID management for COCO JSON
  - `--sample PATH:NUM`: Sampled images get only 1 augmented copy (no original kept), prefixed with `sample_{stem}_` to avoid collisions

- **`main.py`** — Scratch/notebook script for prototyping augmentations with FiftyOne visualization. Not imported by augment.py.

## Data Format

Input/output directories contain images + `_annotations.coco.json`. Bboxes use COCO format `[x, y, w, h]` in pixels. The output JSON gets fresh sequential IDs for both images and annotations starting from 0.

## Key Dependencies

- `albumentations==1.4.24` (pinned — API may differ across versions)
- `opencv-python` for image I/O and reflect_colonies blending
- `typer` for CLI, `tqdm` for progress
