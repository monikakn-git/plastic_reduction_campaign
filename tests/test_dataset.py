import unittest
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "artifacts" / "datasets" / "processed"
DATA_YAML = DATASET_ROOT / "data.yaml"

EXPECTED_CLASSES = {
    0: "litter",
    1: "pile",
    2: "face_mask",
    3: "trash_bin",
    4: "plastic_bag",
    5: "bottle",
    6: "cup",
    7: "rope",
    8: "sachet",
    9: "straw",
}


class TestDataset(unittest.TestCase):
    def test_data_yaml_exists(self):
        self.assertTrue(DATA_YAML.is_file(), f"data.yaml not found at {DATA_YAML}")

    def test_data_yaml_classes(self):
        with DATA_YAML.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self.assertIn("names", config, "names missing from data.yaml")
        names = config["names"]
        if isinstance(names, list):
            names_dict = {i: n for i, n in enumerate(names)}
        else:
            names_dict = {int(k): v for k, v in names.items()}

        self.assertEqual(names_dict, EXPECTED_CLASSES, "data.yaml classes do not match canonical classes")

    def test_dataset_splits_exist(self):
        for split in ("train", "val", "test"):
            img_dir = DATASET_ROOT / "images" / split
            lbl_dir = DATASET_ROOT / "labels" / split
            self.assertTrue(img_dir.is_dir(), f"Images dir missing: {img_dir}")
            self.assertTrue(lbl_dir.is_dir(), f"Labels dir missing: {lbl_dir}")
            self.assertGreater(len(list(img_dir.iterdir())), 0, f"No images in {split}")
            self.assertGreater(len(list(lbl_dir.iterdir())), 0, f"No labels in {split}")


if __name__ == "__main__":
    unittest.main()
