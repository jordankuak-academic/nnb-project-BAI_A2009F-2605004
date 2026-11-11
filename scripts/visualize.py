import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_manifest(dataset_root: Path) -> dict:
    return json.loads((dataset_root / "split.json").read_text(encoding="utf-8-sig"))


def load_stats(dataset_root: Path) -> dict:
    return json.loads((dataset_root / "split_stat.json").read_text(encoding="utf-8-sig"))


def plot_class_distribution(stats: dict, output_path: Path) -> None:
    classes = sorted(stats["per_class"].keys())
    subsets = ["training", "validating", "testing"]
    colors = ["#2B47C0", "#617DFF", "#AEB3BD"]
    
    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(classes))
    for subset, color in zip(subsets, colors):
        values = [stats["per_class"][label][subset] for label in classes]
        ax.bar(classes, values, bottom=bottom, label=subset, color=color)
        bottom += np.array(values)
    
    ax.set_xlabel("class")
    ax.set_ylabel("number of images")
    ax.set_title("Per-class distribution across splits")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def plot_data_quality(stats: dict, output_path: Path) -> None:
    labels = ["raw candidates", "unique images", "duplicates removed", "rejected"]
    values = [
        stats["raw_candidates"],
        stats["unique_images"],
        stats["duplicates_removed"],
        stats["rejected_count"],
    ]
    colors = ["#AEB3BD", "#2B47C0", "#617DFF", "#FF6B6B"]
    
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(value),
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("number of images")
    ax.set_title("Data quality: raw, unique, duplicates and rejected")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def plot_sample_grid(dataset_root: Path, manifest: dict, class_names: list[str], output_path: Path, per_class: int = 5) -> None:
    samples = manifest["samples"]
    by_label: dict[str, list[dict]] = {label: [] for label in class_names}
    for sample in samples:
        by_label[sample["label"]].append(sample)
    
    fig, axes = plt.subplots(len(class_names), per_class, figsize=(per_class * 2, len(class_names) * 2))
    for row, label in enumerate(class_names):
        picks = by_label[label][:per_class]
        for col in range(per_class):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(picks):
                image_path = dataset_root / picks[col]["all_path"]
                with Image.open(image_path) as image:
                    ax.imshow(image)
            if col == 0:
                ax.set_ylabel(label, rotation=0, labelpad=50, fontsize=10)
    fig.suptitle("Sample images per class", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dataset_root = root / "datasets" / "images"
    processed_all = dataset_root / "processed" / "all"
    figures_dir = root / "storage" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = load_manifest(dataset_root)
    stats = load_stats(dataset_root)
    class_names = [label for label, _ in sorted(manifest["class_to_idx"].items(), key=lambda item: item[1])]
    
    plot_class_distribution(stats, figures_dir / "class_distribution.png")
    plot_data_quality(stats, figures_dir / "data_quality.png")
    plot_sample_grid(dataset_root, manifest, class_names, figures_dir / "sample_grid.png")
    
    print(json.dumps({
        "figures": {
            "class_distribution": str(figures_dir / "class_distribution.png"),
            "image_size_histogram": str(figures_dir / "image_size_histogram.png"),
            "sample_grid": str(figures_dir / "sample_grid.png"),
        }
    }, indent=2))


if __name__ == "__main__":
    main()
