"""Unified AgriGPT Pipeline.
Integrates MobileNetV2 Vision, NLP Preprocessing, ML Intent Classification,
NER Entity Extraction, Hybrid RAG Retrieval, and Groq Llama 3.1 8B Generation.
"""

from __future__ import annotations

import io
from typing import Dict, Any, List, Optional, Union
from PIL import Image

from src.nlp.preprocessor import AgriTextPreprocessor, default_preprocessor
from src.nlp.intent_classifier import AgriIntentClassifier, default_intent_classifier
from src.nlp.ner_extractor import AgriNERExtractor, default_ner_extractor
from src.nlp.vector_store import AgriVectorStore, default_vector_store
from src.nlp.rag_engine import AgriRAGEngine, default_rag_engine
from src.nlp.conversation_manager import AgriConversationManager
from src.llm.groq_client import GroqAgriClient, default_groq_client
from src.vision.disease_classifier import PlantDiseaseClassifier, default_disease_classifier


class AgriGPTPipeline:
    """Master orchestrator combining Vision and NLP/RAG/LLM pipelines."""

    def __init__(
        self,
        rag_engine: Optional[AgriRAGEngine] = None,
        llm_client: Optional[GroqAgriClient] = None,
        vision_classifier: Optional[PlantDiseaseClassifier] = None,
        conversation_manager: Optional[AgriConversationManager] = None
    ):
        self.rag_engine = rag_engine or default_rag_engine
        self.llm_client = llm_client or default_groq_client
        self.vision_classifier = vision_classifier or default_disease_classifier
        self.conversation = conversation_manager or AgriConversationManager()

    def set_groq_api_key(self, api_key: str):
        """Configure or update Groq API key."""
        self.llm_client.set_api_key(api_key)

    def reset_conversation(self):
        """Clear all conversation turns and active context."""
        self.conversation.reset()

    def process_image(self, image_input: Union[str, Image.Image, bytes, io.BytesIO]) -> Dict[str, Any]:
        """Diagnose plant foliage image using MobileNetV2 and update conversation context."""
        diagnosis = self.vision_classifier.predict(image_input)
        self.conversation.update_vision_diagnosis(diagnosis)
        return diagnosis

    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Execute full NLP RAG and LLM response generation for text query."""
        # 1. Retrieve RAG context with active conversation state
        rag_packet = self.rag_engine.retrieve_context(
            query=user_query,
            conversation_manager=self.conversation,
            top_k=3
        )

        # 2. Add user message and NLP metadata to conversation memory
        self.conversation.add_user_message(
            text=user_query,
            nlp_meta={
                "intent": rag_packet["nlp_analysis"]["intent"],
                "entities": rag_packet["nlp_analysis"]["entities"]["grouped"],
                "tokens": rag_packet["nlp_analysis"]["preprocessed"]["tokens"],
                "retrieved_docs_count": len(rag_packet["retrieved_documents"])
            }
        )

        # 3. Formulate system prompt
        system_prompt = self.rag_engine.construct_system_prompt(rag_packet)

        # 4. Generate response via Groq Llama 3.1 8B Instant (or fallback)
        history_messages = self.conversation.get_recent_history()
        llm_result = self.llm_client.generate_response(
            system_prompt=system_prompt,
            messages=history_messages,
            rag_packet=rag_packet
        )

        assistant_text = llm_result["text"]
        source = llm_result["source"]

        # 5. Save assistant reply in conversation memory
        self.conversation.add_assistant_message(
            text=assistant_text,
            source=source,
            meta={"llm_status": llm_result["status"]}
        )

        return {
            "query": user_query,
            "response": assistant_text,
            "source": source,
            "llm_result": llm_result,
            "rag_packet": rag_packet,
            "nlp_analysis": rag_packet["nlp_analysis"],
            "retrieved_documents": rag_packet["retrieved_documents"],
            "active_context": self.conversation.get_context_summary()
        }

    def process_multimodal(
        self,
        image_input: Union[str, Image.Image, bytes, io.BytesIO],
        user_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process both image and optional user question simultaneously."""
        diagnosis = self.process_image(image_input)

        query = user_query.strip() if user_query and user_query.strip() else (
            f"I uploaded a leaf image diagnosed as {diagnosis['crop']} with {diagnosis['disease']}. "
            f"What should I do to treat it and prevent further spread?"
        )

        query_result = self.process_query(query)

        return {
            "diagnosis": diagnosis,
            "query_result": query_result,
            "response": query_result["response"],
            "source": query_result["source"]
        }


# Global default pipeline instance
default_pipeline = AgriGPTPipeline()
