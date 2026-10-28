
from pathlib import Path
import json
import zipfile
from collections import Counter

import matplotlib.pyplot as plt

DATA_ROOT = Path(r"G:\My Drive\datasets\Ai_Waste_Campaign")
ANNOTATIONS_ZIP = DATA_ROOT / "annotations.zip"
SPLIT_DIR = "annotations/4_class_splits"

OUTPUT_DIR = Path("reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_split(split):
    member = f"{SPLIT_DIR}/annotations_{split}.json"
    with zipfile.ZipFile(ANNOTATIONS_ZIP) as archive:
        with archive.open(member) as file:
            return json.load(file)


def collect_data():
    results = {}

    for split in ("train", "val", "test"):
        data = load_split(split)
        categories = {
            item["id"]: item["name"]
            for item in data["categories"]
        }

        class_counts = Counter(
            item["category_id"] for item in data["annotations"]
        )

        results[split] = {
            "images": data["images"],
            "annotations": data["annotations"],
            "class_counts": {
                categories[key]: class_counts.get(key, 0)
                for key in categories
            },
        }

    return results


def plot_class_distribution(results):
    splits = list(results)
    classes = list(results["train"]["class_counts"])

    x = range(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for index, split in enumerate(splits):
        counts = [
            results[split]["class_counts"][name]
            for name in classes
        ]
        positions = [
            value + (index - 1) * width for value in x
        ]
        ax.bar(positions, counts, width, label=split.capitalize())

    ax.set_xticks(list(x))
    ax.set_xticklabels(classes)
    ax.set_ylabel("Number of annotations")
    ax.set_title("EcoPulse AI — Class Distribution")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "class_distribution.png", dpi=200)
    plt.close(fig)


def plot_image_dimensions(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for split, data in results.items():
        images = data["images"]
        axes[0].hist(
            [item["width"] for item in images],
            bins=30, alpha=0.45, label=split.capitalize()
        )
        axes[1].hist(
            [item["height"] for item in images],
            bins=30, alpha=0.45, label=split.capitalize()
        )

    axes[0].set_title("Image Width Distribution")
    axes[0].set_xlabel("Width (pixels)")
    axes[1].set_title("Image Height Distribution")
    axes[1].set_xlabel("Height (pixels)")

    for ax in axes:
        ax.set_ylabel("Number of images")
        ax.legend()
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "image_dimensions.png", dpi=200)
    plt.close(fig)


def plot_data_quality(results):
    splits = list(results)
    image_counts = [
        len(results[split]["images"]) for split in splits
    ]
    annotation_counts = [
        len(results[split]["annotations"]) for split in splits
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(splits))
    width = 0.35

    ax.bar(
        [value - width / 2 for value in x],
        image_counts, width, label="Images"
    )
    ax.bar(
        [value + width / 2 for value in x],
        annotation_counts, width, label="Annotations"
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels([split.capitalize() for split in splits])
    ax.set_ylabel("Count")
    ax.set_title("Dataset Split Overview")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "data_quality.png", dpi=200)
    plt.close(fig)


def main():
    results = collect_data()

    plot_class_distribution(results)
    plot_image_dimensions(results)
    plot_data_quality(results)

    print("Visualizations generated:")
    for filename in (
        "class_distribution.png",
        "image_dimensions.png",
        "data_quality.png",
    ):
        print(OUTPUT_DIR / filename)


if __name__ == "__main__":
    main()
