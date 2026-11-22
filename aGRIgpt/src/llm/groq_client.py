"""Groq Llama 3.1 8B Instant LLM Client for AgriGPT.
Handles prompt generation, API calls, error recovery, and provides an expert local
knowledge-based response generator when offline or when GROQ_API_KEY is not set.
"""

from __future__ import annotations

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load .env if present
load_dotenv()


class GroqAgriClient:
    """Client for Groq LLM (llama-3.1-8b-instant) with integrated local expert fallback."""

    DEFAULT_MODEL = "llama-3.1-8b-instant"

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        self.client = None
        self._init_client()

    def _init_client(self):
        """Initialize Groq client if API key is configured."""
        if self.api_key and self.api_key.strip():
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key.strip())
            except Exception as e:
                print(f"[GroqAgriClient] Warning: Failed to initialize Groq client: {e}")
                self.client = None
        else:
            self.client = None

    def set_api_key(self, api_key: str):
        """Update API key dynamically from UI."""
        self.api_key = api_key.strip() if api_key else ""
        self._init_client()

    def is_available(self) -> bool:
        """Check if Groq API is configured and client initialized."""
        return self.client is not None and bool(self.api_key)

    def generate_response(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        rag_packet: Dict[str, Any],
        temperature: float = 0.3,
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        """Generate agricultural advisory response using Groq Llama 3.1 8B or local engine."""
        # 1. Try Groq Llama 3.1 8B Instant if available
        if self.is_available():
            try:
                formatted_messages = [{"role": "system", "content": system_prompt}]
                # Append prior conversation history
                for m in messages:
                    if m["role"] in ("user", "assistant"):
                        formatted_messages.append({"role": m["role"], "content": m["content"]})

                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=formatted_messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                content = completion.choices[0].message.content
                return {
                    "text": content,
                    "source": f"Groq ({self.model})",
                    "status": "success",
                    "usage": {
                        "prompt_tokens": getattr(completion.usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(completion.usage, "completion_tokens", 0)
                    }
                }
            except Exception as e:
                print(f"[GroqAgriClient] Error calling Groq API: {e}. Falling back to local engine...")
                # Fall through to local fallback

        # 2. Local Fallback Expert Generator based on RAG & NLP
        fallback_text = self._generate_local_fallback(rag_packet)
        return {
            "text": fallback_text,
            "source": "AgriGPT Local Knowledge Engine (Set GROQ_API_KEY for Llama 3.1 8B)",
            "status": "fallback",
            "usage": None
        }

    def _generate_local_fallback(self, rag_packet: Dict[str, Any]) -> str:
        """High quality local agricultural synthesis using retrieved documents and NLP metadata."""
        nlp = rag_packet.get("nlp_analysis", {})
        intent = nlp.get("intent", {}).get("intent", "general_greeting_help")
        entities = nlp.get("entities", {}).get("grouped", {})
        docs = rag_packet.get("retrieved_documents", [])
        vision = rag_packet.get("vision_info")
        query = rag_packet.get("query", "")

        crops = entities.get("CROPS", [])
        diseases = entities.get("DISEASES", [])
        pests = entities.get("PESTS", [])

        crop_str = crops[0] if crops else (vision.get("crop") if vision else "your crop")
        disease_str = diseases[0] if diseases else (vision.get("disease") if vision else None)

        response_parts = []

        # Vision context banner
        if vision:
            response_parts.append(
                f"### 🌱 Visual Inspection Diagnosis\n"
                f"- **Diagnosed Plant:** {vision.get('crop')}\n"
                f"- **Condition:** {vision.get('disease')} (Confidence: {vision.get('confidence', 0)*100:.1f}%)\n"
                f"- **Severity:** {vision.get('severity', 'Moderate')}\n"
                f"- **Primary Action:** {vision.get('recommendation', 'Inspect leaves and isolate affected plants.')}\n"
            )

        # Knowledge Base Content
        if docs:
            top_doc = docs[0]
            response_parts.append(f"### 📋 Agronomic Recommendation: {top_doc.get('title')}\n")
            response_parts.append(top_doc.get("content", ""))

            if len(docs) > 1:
                response_parts.append("\n\n#### 🔍 Additional Relevant Guidelines:")
                for d in docs[1:]:
                    response_parts.append(f"- **{d.get('title')}**: {d.get('content')[:180]}...")
        else:
            if intent == "general_greeting_help":
                response_parts.append(
                    "Hello! I am **AgriGPT**, your AI Agricultural Assistant.\n\n"
                    "I can help you with:\n"
                    "- 🌿 **Plant Disease Diagnosis & Vision Detection**\n"
                    "- 🐛 **Integrated Pest Management (IPM)**\n"
                    "- 🧪 **Fertilizer & Soil pH Recommendations**\n"
                    "- 💧 **Irrigation & Water Scheduling**\n"
                    "- 🌾 **Crop Cultivation & Varietal Advice**\n\n"
                    "How can I assist your farm today?"
                )
            else:
                response_parts.append(
                    f"Regarding your query on **{crop_str}**:\n"
                    "Please ensure optimal field sanitation, balanced fertilization, and adequate drainage. "
                    "For severe symptoms, consult your local agricultural extension service or state Krishi Vigyan Kendra (KVK)."
                )

        response_parts.append(
            "\n\n> 💡 *Note: Running in offline local knowledge mode. For full dynamic generative reasoning, provide a Groq API key in the sidebar.*"
        )

        return "\n\n".join(response_parts)


# Singleton instance
default_groq_client = GroqAgriClient()
