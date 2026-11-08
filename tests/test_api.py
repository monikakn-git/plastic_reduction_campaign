import unittest
from src.api.app import home, health

class TestAPI(unittest.TestCase):
    def test_home(self):
        res = home()
        self.assertEqual(res["project"], "EcoPulse AI")
        self.assertIn("available_models", res)
        self.assertEqual(res["default_model"], "yolo11n")

    def test_health(self):
        res = health()
        self.assertEqual(res["status"], "healthy")
        self.assertTrue(res["model_loaded"])
        self.assertEqual(len(res["classes"]), 10)
        self.assertIn("yolo11n", res["available_checkpoints"])


if __name__ == "__main__":
    unittest.main()
