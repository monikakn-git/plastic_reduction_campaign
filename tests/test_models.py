import unittest
from pathlib import Path
from src.models.predict import DEFAULT_CHECKPOINTS, load_model, predict_image, detect_model_name

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGES_DIR = PROJECT_ROOT / "artifacts" / "datasets" / "processed" / "images" / "test"


class TestModels(unittest.TestCase):
    def test_checkpoints_exist(self):
        for name, path in DEFAULT_CHECKPOINTS.items():
            self.assertTrue(path.is_file(), f"Checkpoint {name} not found at {path}")

    def test_checkpoint_loading(self):
        for name, path in DEFAULT_CHECKPOINTS.items():
            model = load_model(path)
            self.assertIsNotNone(model)
            self.assertEqual(len(model.names), 10)
            self.assertIn(detect_model_name(path), ("YOLO11n", "YOLOv8n"))

    def test_inference_on_sample_image(self):
        test_images = list(TEST_IMAGES_DIR.glob("*.jpg"))
        self.assertGreater(len(test_images), 0, "No test images found")
        sample_img = test_images[0]

        for name, path in DEFAULT_CHECKPOINTS.items():
            res = predict_image(sample_img, confidence=0.25, checkpoint_path=path)
            self.assertEqual(res["image"], sample_img.name)
            self.assertIn("predictions", res)
            self.assertIn("total_detections", res)
            self.assertEqual(res["total_detections"], len(res["predictions"]))
            self.assertIn(res["model"], ("YOLO11n", "YOLOv8n"))


if __name__ == "__main__":
    unittest.main()
