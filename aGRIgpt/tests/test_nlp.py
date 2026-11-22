"""Unit and Integration Tests for AgriGPT NLP Modules."""

import unittest
from src.nlp.preprocessor import AgriTextPreprocessor
from src.nlp.intent_classifier import AgriIntentClassifier
from src.nlp.ner_extractor import AgriNERExtractor
from src.nlp.embeddings import AgriEmbeddingModel
from src.nlp.vector_store import AgriVectorStore
from src.nlp.conversation_manager import AgriConversationManager
from src.nlp.rag_engine import AgriRAGEngine


class TestAgriNLP(unittest.TestCase):

    def setUp(self):
        self.preprocessor = AgriTextPreprocessor()
        self.intent_classifier = AgriIntentClassifier()
        self.ner_extractor = AgriNERExtractor()
        self.vector_store = AgriVectorStore()
        self.rag_engine = AgriRAGEngine(
            preprocessor=self.preprocessor,
            intent_classifier=self.intent_classifier,
            ner_extractor=self.ner_extractor,
            vector_store=self.vector_store
        )
        self.conv_mgr = AgriConversationManager()

    def test_preprocessor(self):
        text = "What's the best dose of 19-19-19 and Mancozeb at 2.5 g/L for my tomato leaves?"
        res = self.preprocessor.preprocess(text)
        self.assertIn("tomato", res["cleaned"])
        self.assertTrue(len(res["tokens"]) > 0)
        self.assertTrue(any("2.5 g/L" in term or "2.5" in term for term in res["preserved_terms"] + res["tokens"]))

    def test_intent_classification(self):
        # Test disease intent
        res_disease = self.intent_classifier.predict("My potato leaves are rotting with late blight and brown lesions")
        self.assertEqual(res_disease["intent"], "disease_diagnosis")
        self.assertGreater(res_disease["confidence"], 0.25)

        # Test pest intent
        res_pest = self.intent_classifier.predict("Green aphids and whiteflies are attacking my cotton crop")
        self.assertEqual(res_pest["intent"], "pest_control")

        # Test fertilizer intent
        res_fert = self.intent_classifier.predict("How much urea and DAP fertilizer should I apply per acre?")
        self.assertEqual(res_fert["intent"], "fertilizer_soil_management")

    def test_ner_extraction(self):
        query = "How to treat Late Blight on Tomato using Mancozeb 75% WP @ 2.5 g/L in sandy loam soil?"
        entities = self.ner_extractor.extract_entities(query)
        grouped = entities["grouped"]

        self.assertIn("Tomato", grouped["CROPS"])
        self.assertIn("Late Blight", grouped["DISEASES"])
        self.assertIn("Mancozeb", grouped["FERTILIZERS_CHEMICALS"])
        self.assertIn("Alluvial Soil", grouped["SOIL_TYPES"])  # sandy loam alias
        self.assertTrue(len(grouped["METRIC_QUANTITIES"]) > 0)

    def test_vector_search(self):
        results = self.vector_store.search("late blight tomato control spray", top_k=2)
        self.assertGreater(len(results), 0)
        top_res = results[0]
        self.assertIn("Tomato", top_res["crop"])
        self.assertIn("Late Blight", top_res["title"])

    def test_rag_engine_and_context(self):
        rag_packet = self.rag_engine.retrieve_context(
            query="What chemical fungicide controls late blight in tomatoes?",
            conversation_manager=self.conv_mgr,
            top_k=2
        )
        self.assertEqual(rag_packet["active_crop"], "Tomato")
        self.assertGreater(len(rag_packet["retrieved_documents"]), 0)
        system_prompt = self.rag_engine.construct_system_prompt(rag_packet)
        self.assertIn("AgriGPT", system_prompt)
        self.assertIn("Late Blight", system_prompt)


if __name__ == "__main__":
    unittest.main()
