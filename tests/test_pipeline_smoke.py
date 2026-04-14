import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from models.model_loader import load_model
from models.predict import predict_image
from utils.gradcam import generate_gradcam_overlay
from utils.preprocess import preprocess_image


class TestPipelineSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_path = Path(__file__).resolve().parents[1] / "models" / "best_model.pth"
        if not cls.model_path.exists():
            raise unittest.SkipTest("Model weights file models/best_model.pth is required for smoke test.")
        cls.model = load_model(cls.model_path, torch.device("cpu"))

    def test_inference_and_gradcam_overlay(self):
        sample_dir = Path(__file__).resolve().parents[1] / "assets" / "sample_images"
        sample_candidates = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            sample_candidates.extend(sample_dir.glob(ext))

        if not sample_candidates:
            raise unittest.SkipTest(
                "No real sample images found in assets/sample_images. "
                "Add at least one CT image (.jpg/.jpeg/.png)."
            )

        sample_path = sample_candidates[0]
        image = np.array(Image.open(sample_path).convert("RGB"), dtype=np.uint8)
        input_tensor = preprocess_image(image)

        prediction, confidence, probabilities = predict_image(self.model, input_tensor)
        overlay = generate_gradcam_overlay(self.model, input_tensor, image)

        self.assertIn(prediction, {"hemorrhage", "normal"})
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        self.assertEqual(probabilities.shape, (2,))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=4)

        self.assertEqual(overlay.shape, image.shape)
        self.assertEqual(overlay.dtype, np.uint8)
        self.assertTrue(np.isfinite(overlay).all())


if __name__ == "__main__":
    unittest.main()
