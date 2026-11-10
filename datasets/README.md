# Raw Dataset Documentation

## 1. Overview

This directory contains the raw image dataset used to train, validate, and test the JordanCNN waste classifier for the BAI13143 Neural Network Basics Individual Project. The dataset is a **curated merge of two public Kaggle datasets**, filtered and remapped into **six uniform target classes** to ensure a consistent and manageable classification problem.

The raw images are **not** baked into the Docker images. They are mounted into the training and testing containers at runtime via `-v ./datasets:/workspace/datasets`, in line with the reproducibility requirements of Task 6.

## 2. Data Sources

| Source | Kaggle Dataset | Original Classes | Images (approx.) | Role in this project |
|---|---|---|---|---|
| Source A | [Garbage Classification](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification) | 6 (cardboard, glass, metal, paper, plastic, trash) | ~2,500 | Primary source; class definitions follow this dataset |
| Source B | [Garbage Classification (12 classes)](https://www.kaggle.com/datasets/mostafaabla/garbage-classification) | 12 (paper, cardboard, biological, metal, plastic, green-glass, brown-glass, white-glass, clothes, shoes, batteries, trash) | ~15,150 | Supplementary source; visually similar classes were merged into the six target classes |

**Merged and filtered dataset size**: approximately 5,000–6,000 images across six classes. The exact count is recorded in `split_stat.json` after running `python -m scripts.initialize`.

## 3. License and Usage

- **Academic use only.** The dataset is used strictly for academic and educational purposes as part of the BAI13143 Neural Network Basics Individual Project. It is not used for any commercial purpose.
- **Original licenses.** Each source dataset is distributed under its own license on Kaggle. Users must refer to the individual Kaggle dataset pages for the specific terms (e.g. CC0, CC BY, or "Data files © Original Authors"). Source B explicitly states that web-scraped images are owned by their original photographers and are used only to promote research.
- **No personal data.** No faces, names, identity numbers, or other personal data are included, in accordance with the Personal Data Protection Act 2010 (PDPA).
- **Attribution.** If you reuse this curated subset, please cite both original Kaggle datasets.

## 4. Class Standardization

To ensure a consistent six-class problem, the raw images from both sources are filtered and remapped as follows:

| Target class | Source A | Source B (merged sub-classes) |
|---|---|---|
| `cardboard` | cardboard | cardboard |
| `glass` | glass | green-glass, brown-glass, white-glass |
| `metal` | metal | metal |
| `paper` | paper | paper |
| `plastic` | plastic | plastic |
| `trash` | trash | trash, biological |

**Excluded classes from Source B**: `clothes`, `shoes`, `batteries`. These are visually and semantically distinct from the six target categories and would introduce label noise if forced into an existing class.

**Why merge the three glass sub-classes?** Source B splits glass into `green-glass`, `brown-glass`, and `white-glass` by colour. For an automated sorting pipeline, all glass items are routed to the same bin, so colour is not a useful target. Merging them yields a single `glass` class consistent with Source A.

**Why merge `biological` into `trash`?** Biological waste (rotten food, food remains) is routed to the same "other/general waste" bin as the `trash` category in a typical sorting system. Merging avoids a seventh class that would be visually ambiguous with `trash`.

## 5. Folder Structure

The `raw/` directory must follow this structure for the `Preprocessor` to work correctly:

```text
raw/
├── source_1/
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   └── trash/
└── source_2/
    ├── cardboard/
    ├── glass/
    ├── metal/
    ├── paper/
    ├── plastic/
    └── trash/
```

- `source_1/` and `source_2/` preserve provenance so that per-source statistics can be reported.
- Each sub-folder contains the raw images for that class after the mapping in Section 4.
- Images are resized to 128 × 128 and normalised to [0, 1] by `scripts/initialize.py` before being written to `processed/`.

## 6. Processed Output

Running `python -m scripts.initialize` produces:

```text
images/
├── processed/
│   ├── all/           # All images, resized and normalised
│   ├── training/      # 70% split
│   ├── validating/    # 15% split
│   └── testing/       # 15% split
├── split.json         # Manifest: path, class_index, split
└── split_stat.json    # Class distribution and totals
```

These outputs are generated deterministically from the raw data using the seed and ratios defined in `configs/settings.py`, and are excluded from Git and from Docker images.

## 7. Reproduction

1. Download both source datasets from Kaggle.
2. Apply the mapping in Section 4 to produce the raw/ structure in Section 5.
3. From the project root:

```bash
python -m scripts.initialize
python -m scripts.train
```

Or inside Docker:

```bash
docker run --rm \
  -v "$(pwd)/datasets:/workspace/datasets" \
  -v "$(pwd)/storage:/workspace/storage" \
  nn-train
```
