"""End-to-End Tests for the AgriGPT Unified Pipeline."""

import unittest
from PIL import Image
import numpy as np

from src.pipeline import AgriGPTPipeline


class TestAgriGPTPipeline(unittest.TestCase):

    def setUp(self):
        self.pipeline = AgriGPTPipeline()
        self.pipeline.reset_conversation()

    def test_full_text_query_flow(self):
        result = self.pipeline.process_query("What fertilizer and NPK should I give to my wheat crop?")
        self.assertIn("response", result)
        self.assertTrue(len(result["response"]) > 20)
        self.assertIn("nlp_analysis", result)
        self.assertEqual(result["nlp_analysis"]["intent"]["intent"], "fertilizer_soil_management")

    def test_multimodal_vision_and_chat_flow(self):
        # 1. Simulate image upload
        img_array = np.zeros((224, 224, 3), dtype=np.uint8)
        img_array[:, :, 1] = 200
        test_img = Image.fromarray(img_array)

        diagnosis = self.pipeline.process_image(test_img)
        self.assertIsNotNone(diagnosis)
        self.assertIsNotNone(self.pipeline.conversation.active_crop)

        # 2. Subsequent follow-up question without naming the crop
        followup = self.pipeline.process_query("How often should I spray fungicides to cure this?")
        self.assertIn("response", followup)
        # Verify active crop context was retained
        self.assertIsNotNone(followup["active_context"]["active_crop"])


if __name__ == "__main__":
    unittest.main()
