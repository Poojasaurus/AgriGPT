"""Context-Aware Agricultural Conversation Manager.
Maintains multi-turn conversational state, active crop context, diagnosis history,
and contextual memory across user queries and visual inspections.
"""

from __future__ import annotations

import time
from typing import List, Dict, Any, Optional


class AgriConversationManager:
    """Tracks dialog turns, user intent shifts, and active crop/disease focus."""

    def __init__(self, max_history_turns: int = 10):
        self.max_history_turns = max_history_turns
        self.messages: List[Dict[str, Any]] = []
        self.active_crop: Optional[str] = None
        self.active_disease: Optional[str] = None
        self.active_pest: Optional[str] = None
        self.vision_diagnosis: Optional[Dict[str, Any]] = None
        self.metadata_history: List[Dict[str, Any]] = []

    def reset(self):
        """Reset conversation state."""
        self.messages = []
        self.active_crop = None
        self.active_disease = None
        self.active_pest = None
        self.vision_diagnosis = None
        self.metadata_history = []

    def update_vision_diagnosis(self, diagnosis: Dict[str, Any]):
        """Inject leaf disease diagnosis from computer vision module into conversation state."""
        self.vision_diagnosis = diagnosis
        if diagnosis.get("crop"):
            self.active_crop = diagnosis["crop"]
        if diagnosis.get("disease"):
            self.active_disease = diagnosis["disease"]

    def update_context_from_entities(self, entities: Dict[str, Any]):
        """Update active focus based on NER extractions from the current turn."""
        crops = entities.get("grouped", {}).get("CROPS", [])
        diseases = entities.get("grouped", {}).get("DISEASES", [])
        pests = entities.get("grouped", {}).get("PESTS", [])

        if crops:
            self.active_crop = crops[0]
        if diseases:
            self.active_disease = diseases[0]
        if pests:
            self.active_pest = pests[0]

    def add_user_message(self, text: str, nlp_meta: Optional[Dict[str, Any]] = None):
        """Append user message with turn timestamp and NLP metadata."""
        msg = {
            "role": "user",
            "content": text,
            "timestamp": time.time(),
            "meta": nlp_meta or {}
        }
        self.messages.append(msg)
        if nlp_meta:
            self.metadata_history.append(nlp_meta)

    def add_assistant_message(self, text: str, source: str = "groq_llama", meta: Optional[Dict[str, Any]] = None):
        """Append assistant response to conversation history."""
        msg = {
            "role": "assistant",
            "content": text,
            "timestamp": time.time(),
            "source": source,
            "meta": meta or {}
        }
        self.messages.append(msg)

    def get_recent_history(self, num_turns: Optional[int] = None) -> List[Dict[str, str]]:
        """Return clean role/content history suitable for LLM prompting."""
        turns = num_turns or self.max_history_turns
        recent = self.messages[-turns * 2:] if self.messages else []
        return [{"role": m["role"], "content": m["content"]} for m in recent]

    def get_context_summary(self) -> Dict[str, Any]:
        """Summarize current state of the conversation context."""
        return {
            "active_crop": self.active_crop,
            "active_disease": self.active_disease,
            "active_pest": self.active_pest,
            "has_vision_diagnosis": self.vision_diagnosis is not None,
            "vision_diagnosis": self.vision_diagnosis,
            "total_messages": len(self.messages)
        }
