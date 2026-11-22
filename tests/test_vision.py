"""Unit Tests for MobileNetV2 Plant Disease Classifier."""

import unittest
from PIL import Image
import numpy as np

from src.vision.disease_classifier import PlantDiseaseClassifier


class TestPlantVision(unittest.TestCase):

    def setUp(self):
        self.classifier = PlantDiseaseClassifier()

    def test_classifier_initialization(self):
        self.assertEqual(len(self.classifier.classes), 38)
        self.assertIsNotNone(self.classifier.model)

    def test_image_prediction(self):
        # Create a synthetic test leaf image (green leaf simulation)
        img_array = np.zeros((224, 224, 3), dtype=np.uint8)
        img_array[:, :, 1] = 180  # green channel
        img_array[:, :, 0] = 40   # slight red
        img = Image.fromarray(img_array)

        result = self.classifier.predict(img, top_k=3)

        self.assertIn("crop", result)
        self.assertIn("disease", result)
        self.assertIn("confidence", result)
        self.assertIn("recommendation", result)
        self.assertEqual(len(result["top_k"]), 3)
        self.assertTrue(0.0 <= result["confidence"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
