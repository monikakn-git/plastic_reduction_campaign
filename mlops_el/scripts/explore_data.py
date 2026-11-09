"""Produce real Phase 1 charts and annotated samples from the acquired dataset."""
import collections
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".local/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from PIL import Image, ImageDraw, ImageOps


def main():
    out = ROOT / "reports/phase1"
    config = yaml.safe_load((ROOT / 'params.yaml').read_text())
    dataset_name = config['dataset']['name']
    data = json.loads((ROOT / f"data/raw/{dataset_name}/annotations.json").read_text())
    summary = json.loads((out / "dataset_summary.json").read_text())
    names = {x["id"]: x["name"] for x in data["categories"]}
    counts = summary["original_category_counts"]
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.barh(list(counts)[::-1], list(counts.values())[::-1])
    ax.set(title=f"{dataset_name}: source annotation distribution", xlabel="Annotated objects")
    fig.tight_layout()
    fig.savefig(out / "source_class_distribution.png", dpi=130)
    plt.close(fig)
    classes = summary["classes"]
    fig, ax = plt.subplots(figsize=(11, 5))
    for idx, split in enumerate(("train", "val", "test")):
        ax.bar([i + idx * .25 for i in range(len(classes))],
               [summary["split_target_counts"][split].get(c, 0) for c in classes],
               width=.25, label=split)
    ax.set_xticks([i + .25 for i in range(len(classes))], classes, rotation=15)
    ax.set(title="Detection classes: objects by source-grouped split", ylabel="Annotated objects")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "target_split_distribution.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist([i["width"] for i in data["images"]], bins=25, alpha=.6, label="width")
    axes[0].hist([i["height"] for i in data["images"]], bins=25, alpha=.6, label="height")
    axes[0].set(title="Original image dimensions", xlabel="Pixels")
    axes[0].legend()
    by_id = {i["id"]: i for i in data["images"]}
    areas = [a["bbox"][2] * a["bbox"][3] / (by_id[a["image_id"]]["width"] * by_id[a["image_id"]]["height"])
             for a in data["annotations"]]
    axes[1].hist(areas, bins=30)
    axes[1].set(title="Object bounding-box area / image area", xlabel="Relative area")
    fig.tight_layout()
    fig.savefig(out / "image_and_object_sizes.png", dpi=150)
    plt.close(fig)
    annotations = collections.defaultdict(list)
    for a in data["annotations"]:
        annotations[a["image_id"]].append(a)
    contact = Image.new("RGB", (1200, 900), "white")
    sample_images = sorted(data["images"], key=lambda x: x["id"])[::max(1, len(data["images"]) // 9)][:9]
    for index, record in enumerate(sample_images):
        with Image.open(ROOT / f"data/raw/{dataset_name}/images" / record["file_name"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((400, 270))
        draw = ImageDraw.Draw(image)
        sx, sy = image.width / record["width"], image.height / record["height"]
        for a in annotations[record["id"]]:
            x, y, w, h = a["bbox"]
            box = (x*sx, y*sy, (x+w)*sx, (y+h)*sy)
            draw.rectangle(box, outline="red", width=2)
            draw.text((x*sx, y*sy), names[a["category_id"]], fill="yellow")
        col, row = index % 3, index // 3
        contact.paste(image, (col*400, row*300))
        ImageDraw.Draw(contact).text((col*400+5, row*300+275), record["file_name"], fill="black")
    contact.save(out / "annotated_source_samples.jpg", quality=90)
    text = f"""# Phase 1 — Dataset exploration\n\nStatus: exploratory; faculty approval pending.\n\n- Original images verified: {summary['images']}\n- Original annotations: {summary['source_annotations']}\n- Proposed target annotations: {summary['target_annotations']}\n- Images with no selected target class (retained as negatives): {summary['negative_images']}\n- Exact duplicate images: {summary['exact_duplicate_images']}\n- Boxes clipped to image boundaries: {len(summary['clipped_annotation_ids'])}\n- Split image counts: {summary['split_image_counts']}\n\n## Findings and decisions\n\nAll 10,000 selected source images are retained at original resolution. The ten source detection classes are preserved through configs/plitter_classes.json. Images are not duplicated or augmented to reach the count. Negative images remain useful for evaluating false detections. Bottle and cup labels identify object types; they do not certify plastic material.\n\nClass imbalance and small litter objects make per-class precision, recall, and mAP important. No performance value has been assumed. Training-only augmentation changes input appearance and bounding boxes together; no augmented image copies are stored. Feature extraction is learned through the pretrained detector backbone.\n\nSplits keep source-video filename groups together and merge groups linked by exact image duplicates. Whole groups are balanced toward 70/15/15 by image count. This is a grouping proxy, not proof that every related scene is separated. Near-duplicate and visual annotation review are still required. No test-set predictions should guide model tuning.\n\nThe public dataset has no verified campus audit locations. It cannot establish campus hotspots or campaign outcomes. The official Zenodo record declares CC BY 4.0. Attribution and faculty approval remain required; approval is pending.\n\n## Figures\n\n![Original categories](source_class_distribution.png)\n\n![Target categories by split](target_split_distribution.png)\n\n![Image and object sizes](image_and_object_sizes.png)\n\n![Original annotated examples](annotated_source_samples.jpg)\n"""
    text += f"\n## Source annotation identity\n\nThe source contains {summary['source_annotation_id_collisions']} annotation-ID collisions across different images. The pipeline uses the pair (image_id, annotation_id) to preserve every annotation without silently dropping objects. Boundary clipping is recorded separately.\n"
    text += f"\n## JPEG preparation\n\n{len(summary['prepared_jpeg_marker_repairs'])} source JPEGs do not end with the expected end-of-image marker. The prepared copies append that marker without recompressing pixels or changing resolution. Raw originals and their recorded hashes remain unchanged. This prevents automatic image re-encoding by the training library.\n"
    (out / "exploration.md").write_text(text)
    print("Created exploration report, three plots, and original annotated samples.")


if __name__ == "__main__":
    main()
