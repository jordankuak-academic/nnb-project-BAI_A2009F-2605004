# nnb-project-BAI_A2009F-2605004

**Recyclable Waste Classification Using a Custom Convolutional Neural Network**

BAI13143 Neural Network Basics · Individual Project · Raffles University

---

## 1. Project Overview

Improper sorting of recyclable waste causes contamination in recycling streams and reduces the efficiency of manual sorting. This project builds a custom convolutional neural network (**JordanCNN**) that classifies camera-captured images of household waste into **six categories** — `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash` — so that items can be routed to the correct bin in an automated sorting pipeline.

- **Task type**: Multi-class image classification (6 classes)
- **Input**: RGB image, resized to 128 × 128, normalised to [0, 1]
- **Output**: One of six class labels
- **Framework**: PyTorch (CPU)
- **Success criterion (Task 1)**: validation macro-F1 ≥ 0.80 and accuracy ≥ 80%, at least 0.15 above the majority-class baseline

The project is **reproducible end-to-end via Docker**: training runs in one container, testing in a separate, lean container, and the test metrics match those reported during training.

---

## 2. Repository Structure

```
nn-project-BAI_A2009F-2605004/
├── app/
│   ├──http/
│   │   └── routes.py
│   ├── models/
│   │   └── jordan_cnn.py
│   ├── services/
│   │   ├── dataset.py
│   │   ├── evaluator.py
│   │   ├── predictor.py
│   │   ├── preprocessor.py
│   │   └── trainer.py
│   └── toolkits/
│       └── logger.py
├── configs/
│   └── settings.py
├── scripts/
│   ├── initialize.py
│   ├── train.py
│   ├── test.py
│   └── host.py
├── resources/
├── docker/
│   ├── model_train/
│   │   ├── Dockerfile.train
│   │   └── requirements-train.txt
│   └── model_test/
│       ├── Dockerfile.test
│       └── requirements-test.txt
├── datasets/
│   ├── README.md
│   └── images/
├── storage/
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── LICENSE
└── README.md
```

---

## 3. Dataset

The dataset is a **curated merge of two public Kaggle datasets**, filtered and remapped into six target classes:

| Source | Kaggle Dataset | Role |
|---|---|---|
| A | Garbage Classification (asdasdasasdas) | Primary source (6 classes) |
| B | Garbage Classification — 12 classes (mostafaabla) | Supplementary; glass sub-classes merged, `biological` merged into `trash`, `clothes`/`shoes`/`batteries` excluded |

- **Classes**: `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`
- **Split**: 70% training / 15% validating / 15% testing (seed = 42)
- **Preprocessing**: resize to 128 × 128, normalise pixel values to [0, 1]

Full details, license, class-mapping rules and reproduction steps are in [`datasets/README.md`](datasets/README.md).

---

## 4. Network Architecture

**JordanCNN** — a custom CNN designed for this project:

| Stage | Layer | Output | Activation | Parameters |
|---|---|---|---|---|
| Conv block 1 | Conv3×3(3→32) + BN + ReLU + MaxPool2×2 | 32×64×64 | ReLU | 928 |
| Conv block 2 | Conv3×3(32→64) + BN + ReLU + MaxPool2×2 | 64×32×32 | ReLU | 18,560 |
| Conv block 3 | Conv3×3(64→128) + BN + ReLU + MaxPool2×2 | 128×16×16 | ReLU | 73,984 |
| Conv block 4 | Conv3×3(128→256) + BN + ReLU + MaxPool2×2 | 256×8×8 | ReLU | 295,424 |
| Pooling | AdaptiveAvgPool2d(1×1) | 256×1×1 | — | 0 |
| FC hidden | Linear(256→128) + ReLU + Dropout(0.3) | 128 | ReLU | 32,896 |
| Output | Linear(128→6) | 6 | raw logits | 774 |
| **Total** | | | | **422,566** |

- **Loss**: `CrossEntropyLoss`
- **Optimiser**: Adam, `lr=0.0002`, `weight_decay=0`
- **Regularisation**: BatchNorm after each conv, Dropout(0.3) before the output layer
- **Early stopping**: patience = 5, monitored on validation loss

---

## 5. Training and Experiments

| Exp | What changed | Val Acc | Val F1 | Conclusion |
|---|---|---|---|---|
| 0 | Baseline (lr=5e-4, drop=0.5, batch=32) | 77.15% | 0.7509 | Validation loss spiked to 3.06 → LR too high |
| 1 | lr 5e-4 → 2e-4 | 79.23% | 0.7780 | Smooth curves, better generalisation |
| **2** | **dropout 0.5 → 0.3** | **81.10%** | **0.7957** | **Best model** |
| 3 | weight_decay 0 → 1e-4 | 79.75% | 0.7835 | L2 did not address root cause |
| 4 | dropout 0.3 → 0.4 | 71.44% | 0.6893 | Over-regularised, underfitting |
| 5 | (invalid: two variables changed) | — | — | Excluded |
| 6 | batch 32 → 64 | 72.38% | 0.6780 | Fewer updates per epoch |

Per-epoch logs: `storage/logs/experiments.csv`. Curves: `storage/models/versions/<run>/curves/`.

---

## 6. Test-Set Results

The best model (Exp 2) was evaluated **once** on the held-out test set (963 images):

| Metric | Value |
|---|---|
| Accuracy | 0.7757 |
| Macro precision | 0.7591 |
| Macro recall | 0.7447 |
| **Macro F1** | **0.7434** |
| Weighted F1 | 0.7670 |
| Top-3 accuracy | 0.9647 |
| Inference time | 3.08 ms / image (CPU) |
| Majority-class baseline (macro F1) | 0.0798 |
| **Improvement over baseline** | **+0.6637** |

Per-class performance and the 6×6 confusion matrix are in `storage/test_results/`. `metal` had the lowest recall (0.461), mainly confused with `glass` (28 cases), because both are cylindrical, reflective containers that look similar at 128 × 128. Five misclassified examples with explanations are in `storage/test_results/misclassified/`.

---

## 7. Docker: Train and Test in Separate Environments

### 7.1 Prerequisites

- Docker Desktop (Compose v2)
- Raw images placed under `datasets/images/unprocessed/<class>/`

### 7.2 Build both images

```bash
docker compose build
```

### 7.3 Run training

```bash
docker compose run --rm train
```

Runs `python -m scripts.initialize` then `python -m scripts.train`. Artefacts are written to the host `storage/` via a bind mount.

### 7.4 Run testing

```bash
docker compose run --rm test
```

Lean test image with inference dependencies only (torch, numpy, pillow, matplotlib, scikit-learn). Loads `storage/models/release/model.pt`, evaluates on the test set, writes metrics, confusion matrix and misclassified images to `storage/test_results/`.

### 7.5 Run both in sequence

```bash
docker compose up --build
```

`test` waits for `train` to complete successfully (`depends_on: condition: service_completed_successfully`).

### 7.6 Manual `docker run` (equivalent)

```bash
docker build -f docker/model_train/Dockerfile.train -t nn-train .
docker build -f docker/model_test/Dockerfile.test  -t nn-test  .

docker run --rm -v "$(pwd)/datasets:/workspace/datasets" -v "$(pwd)/storage:/workspace/storage" nn-train
docker run --rm -v "$(pwd)/datasets:/workspace/datasets" -v "$(pwd)/storage:/workspace/storage" nn-test
```

PowerShell: replace `$(pwd)` with `${PWD}`.

### 7.7 Reproducibility

- Library versions pinned in `requirements-train.txt` and `requirements-test.txt`.
- Random seeds fixed (`seed = 42`; `random`, `numpy`, `torch`, CUDA all seeded).
- Test container reproduces the metrics in Section 6.
- Datasets and outputs are never copied into the images; they are bind-mounted at runtime.

### 7.8 Different-environment evidence

Training was run on the author's laptop; testing was run on a separate clean user account with no Python packages installed locally. Screenshots are in the technical report (Appendix: Docker Evidence).

---

## 8. Local Dashboard (Optional)

```bash
python -m scripts.host
```

Open `http://127.0.0.1:8000/` to run the evaluator on the full test set or upload a single image for prediction. The dashboard reads `storage/models/release/` and `datasets/images/split.json`; no retraining needed.

---

## 9. Reproducing from Scratch

```bash
git clone https://github.com/jordankuak-academic/nnb-project-BAI_A2009F-2605004.git
cd nnb-project-BAI_A2009F-2605004

# place raw images under datasets/images/unprocessed/{cardboard,glass,metal,paper,plastic,trash}/

docker compose build
docker compose run --rm train
docker compose run --rm test

# results:
#   storage/models/release/
#   storage/logs/experiments.csv
#   storage/models/versions/<run>/curves/
#   storage/test_results/
```

---

## 10. References

1. asdasdasasdas. *Garbage Classification*. Kaggle, 2018. https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification
2. Mostafa Mohamed. *Garbage Classification (12 classes)*. Kaggle, 2021. https://www.kaggle.com/datasets/mostafaabla/garbage-classification

---

## 11. Generative AI Declaration

Generative AI tools (ChatGPT / Claude) were used for learning and debugging only: to understand weight decay, batch-size effects and Docker volume semantics, and to review code structure. All architecture design, experiment decisions, result interpretation and written analysis are the author's own work. This declaration follows the academic-integrity rules stated in Section 6 of the assignment brief.

---

## 12. Author

**Student ID**: BAI_A2009F-2605004
**Course**: BAI13143 Neural Network Basics
**Institution**: Raffles University
**Submission**: Week 11, 11 November 2026
