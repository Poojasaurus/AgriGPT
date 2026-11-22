"""Agricultural Retrieval-Augmented Generation (RAG) Engine.
Integrates text preprocessing, intent classification, NER entity extraction,
hybrid vector retrieval, and context synthesis for accurate agricultural advice.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional

from src.nlp.preprocessor import AgriTextPreprocessor, default_preprocessor
from src.nlp.intent_classifier import AgriIntentClassifier, default_intent_classifier
from src.nlp.ner_extractor import AgriNERExtractor, default_ner_extractor
from src.nlp.vector_store import AgriVectorStore, default_vector_store
from src.nlp.conversation_manager import AgriConversationManager


class AgriRAGEngine:
    """Full NLP & RAG retrieval pipeline for AgriGPT."""

    def __init__(
        self,
        preprocessor: Optional[AgriTextPreprocessor] = None,
        intent_classifier: Optional[AgriIntentClassifier] = None,
        ner_extractor: Optional[AgriNERExtractor] = None,
        vector_store: Optional[AgriVectorStore] = None
    ):
        self.preprocessor = preprocessor or default_preprocessor
        self.intent_classifier = intent_classifier or default_intent_classifier
        self.ner_extractor = ner_extractor or default_ner_extractor
        self.vector_store = vector_store or default_vector_store

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Perform comprehensive NLP analysis on user query."""
        prep_res = self.preprocessor.preprocess(query)
        intent_res = self.intent_classifier.predict(query)
        ner_res = self.ner_extractor.extract_entities(query)

        return {
            "preprocessed": prep_res,
            "intent": intent_res,
            "entities": ner_res
        }

    def retrieve_context(
        self,
        query: str,
        conversation_manager: Optional[AgriConversationManager] = None,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """Retrieve relevant domain knowledge using NLP analysis and active conversation context."""
        nlp_analysis = self.analyze_query(query)

        detected_crops = nlp_analysis["entities"]["grouped"].get("CROPS", [])
        detected_diseases = nlp_analysis["entities"]["grouped"].get("DISEASES", [])
        detected_pests = nlp_analysis["entities"]["grouped"].get("PESTS", [])

        # Active conversation context fallback
        active_crop = detected_crops[0] if detected_crops else None
        active_disease = detected_diseases[0] if detected_diseases else None

        if conversation_manager:
            if not active_crop and conversation_manager.active_crop:
                active_crop = conversation_manager.active_crop
            if not active_disease and conversation_manager.active_disease:
                active_disease = conversation_manager.active_disease

            # Update conversation memory with new entities
            conversation_manager.update_context_from_entities(nlp_analysis["entities"])

        # Determine category filter based on intent
        intent_tag = nlp_analysis["intent"]["intent"]
        category_filter = None
        if intent_tag == "disease_diagnosis":
            category_filter = "disease"
        elif intent_tag == "pest_control":
            category_filter = "pest"
        elif intent_tag == "fertilizer_soil_management":
            category_filter = "fertilizer"
        elif intent_tag == "irrigation_water_management":
            category_filter = "irrigation"

        # Build keywords boost list
        keywords_boost = []
        if active_crop:
            keywords_boost.append(active_crop)
        if active_disease:
            keywords_boost.append(active_disease)
        keywords_boost.extend(detected_pests)

        # Retrieve documents
        retrieved_docs = self.vector_store.search(
            query=query,
            top_k=top_k,
            crop_filter=active_crop,
            category_filter=category_filter,
            keywords_boost=keywords_boost
        )

        # If zero results with strict filter, retry without category filter
        if not retrieved_docs and category_filter:
            retrieved_docs = self.vector_store.search(
                query=query,
                top_k=top_k,
                crop_filter=active_crop,
                category_filter=None,
                keywords_boost=keywords_boost
            )

        # Format context string
        context_snippets = []
        for doc in retrieved_docs:
            snippet = f"--- Document: {doc['title']} (Crop: {doc['crop']}, Score: {doc['score']}) ---\n{doc['content']}"
            context_snippets.append(snippet)

        formatted_context = "\n\n".join(context_snippets) if context_snippets else "No direct agricultural knowledge base entry matched."

        # Include vision diagnosis info if present in conversation
        vision_info = None
        if conversation_manager and conversation_manager.vision_diagnosis:
            vision_info = conversation_manager.vision_diagnosis

        return {
            "query": query,
            "nlp_analysis": nlp_analysis,
            "active_crop": active_crop,
            "active_disease": active_disease,
            "vision_info": vision_info,
            "retrieved_documents": retrieved_docs,
            "formatted_context": formatted_context
        }

    def construct_system_prompt(self, rag_packet: Dict[str, Any]) -> str:
        """Construct prompt for Groq Llama 3.1 8B Instant."""
        intent = rag_packet["nlp_analysis"]["intent"]["intent"]
        confidence = rag_packet["nlp_analysis"]["intent"]["confidence"]
        entities = rag_packet["nlp_analysis"]["entities"]["grouped"]
        active_crop = rag_packet.get("active_crop") or "Not explicitly specified"
        vision_info = rag_packet.get("vision_info")

        vision_section = ""
        if vision_info:
            vision_section = (
                f"\n[LEAF IMAGE DIAGNOSIS CONTEXT]\n"
                f"- Diagnosed Plant: {vision_info.get('crop', 'Unknown')}\n"
                f"- Diagnosed Condition: {vision_info.get('disease', 'Unknown')}\n"
                f"- Model Confidence: {vision_info.get('confidence', 0.0) * 100:.1f}%\n"
                f"- Severity: {vision_info.get('severity', 'Moderate')}\n"
                f"- Treatment Notes: {vision_info.get('recommendation', 'N/A')}\n"
            )

        prompt = f"""You are AgriGPT, an expert AI Agricultural Agronomist and Plant Pathologist assistant.
Your mission is to provide accurate, safe, scientifically grounded, and actionable farming advice.

[NLP ANALYSIS OF FARMER'S QUERY]
- Classified Intent: {intent} (Confidence: {confidence * 100:.1f}%)
- Extracted Crops: {', '.join(entities.get('CROPS', [])) or 'None'}
- Extracted Diseases: {', '.join(entities.get('DISEASES', [])) or 'None'}
- Extracted Pests: {', '.join(entities.get('PESTS', [])) or 'None'}
- Extracted Fertilizers/Chemicals: {', '.join(entities.get('FERTILIZERS_CHEMICALS', [])) or 'None'}
- Extracted Quantities/Dosages: {', '.join(entities.get('METRIC_QUANTITIES', [])) or 'None'}
- Active Crop Context: {active_crop}
{vision_section}
[DOMAIN KNOWLEDGE BASE (RAG CONTEXT)]
{rag_packet['formatted_context']}

[GUIDELINES FOR RESPONSE]
1. Answer clearly, empathetically, and professionally like an experienced agricultural extension officer.
2. If chemicals (fungicides/insecticides) are recommended, always provide exact dosages (e.g. g/L or ml/L) and safety precautions.
3. Suggest both Integrated Pest Management (chemical) AND Organic/Biological alternatives whenever applicable.
4. If a leaf image was diagnosed, reference the visual findings and validate them against the farmer's question.
5. Format key recommendations using bullet points, bold highlights, and clear action steps.
"""
        return prompt


# Singleton instance
default_rag_engine = AgriRAGEngine()
