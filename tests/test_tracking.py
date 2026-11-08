import unittest
from pathlib import Path
from src.tracking.mlflow_utils import get_tracking_uri, setup_tracking, DEFAULT_EXPERIMENT

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestTracking(unittest.TestCase):
    def test_tracking_uri_resolution(self):
        uri = get_tracking_uri()
        self.assertTrue("mlflow.db" in uri or uri.startswith("http"))

    def test_setup_tracking(self):
        # Verify setup_tracking does not raise exception
        setup_tracking(DEFAULT_EXPERIMENT)


if __name__ == "__main__":
    unittest.main()
