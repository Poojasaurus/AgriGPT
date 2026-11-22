"""Named Entity Recognition (NER) Module for Agriculture.
Extracts agricultural domain entities (Crops, Diseases, Pests, Chemicals/Fertilizers,
Soil Types, Growth Stages, and Dosage/Quantities) using hybrid gazetteer matching,
alias normalization, and regex quantity extractors.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, Any, List, Set, Optional


class AgriNERExtractor:
    """Extracts domain-specific agricultural entities from text."""

    DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "entities.json")

    QUANTITY_REGEXES = [
        # Dosages like 2.5 g/L, 2 ml/litre, 50 kg/ha, 3 kg/acre, 100 mg/L
        re.compile(r'\b(?:\d+(?:\.\d+)?)\s*(?:g|ml|kg|l|litre|litres|mg|tonnes?|quintals?)\s*(?:\/|\s+per\s+)\s*(?:l|litre|hectare|ha|acre|plant|tree|m2)\b', re.IGNORECASE),
        # Direct weights and concentrations: 19-19-19, 75% WP, 25% EC, 5% SG
        re.compile(r'\b\d+\s*%(?:\s*(?:WP|EC|SC|WG|SP|SL|SG))?\b', re.IGNORECASE),
        re.compile(r'\b\d+[\:\-]\d+[\:\-]\d+\b', re.IGNORECASE),
        # pH measurements: pH 6.5, pH 7
        re.compile(r'\bpH\s*(?:\d+(?:\.\d+)?)\b', re.IGNORECASE),
        # Duration/Time: 20-25 DAS, 30 DAT, 15 days
        re.compile(r'\b\d+(?:\s*-\s*\d+)?\s*(?:DAS|DAT|days|weeks|months|hours)\b', re.IGNORECASE)
    ]

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = os.path.abspath(data_path or self.DEFAULT_DATA_PATH)
        self.entity_gazetteer: Dict[str, Dict[str, str]] = {
            "CROP": {},
            "DISEASE": {},
            "PEST": {},
            "FERTILIZER_CHEMICAL": {},
            "SOIL_TYPE": {},
            "GROWTH_STAGE": {}
        }
        self.entity_canonical_names: Dict[str, Set[str]] = {
            "CROP": set(),
            "DISEASE": set(),
            "PEST": set(),
            "FERTILIZER_CHEMICAL": set(),
            "SOIL_TYPE": set(),
            "GROWTH_STAGE": set()
        }
        self._load_gazetteers()

    def _load_gazetteers(self):
        """Build alias-to-canonical lookup tables from entities.json."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Entities dataset not found at {self.data_path}")

        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        category_map = {
            "crops": "CROP",
            "diseases": "DISEASE",
            "pests": "PEST",
            "fertilizers_chemicals": "FERTILIZER_CHEMICAL",
            "soil_types": "SOIL_TYPE",
            "growth_stages": "GROWTH_STAGE"
        }

        for json_key, entity_type in category_map.items():
            for item in data.get(json_key, []):
                canonical = item["name"]
                self.entity_canonical_names[entity_type].add(canonical)
                # Map canonical name itself (case-folded)
                self.entity_gazetteer[entity_type][canonical.lower()] = canonical
                # Map synonyms/aliases
                for syn in item.get("synonyms", []):
                    self.entity_gazetteer[entity_type][syn.lower()] = canonical

    def extract_quantities(self, text: str) -> List[Dict[str, Any]]:
        """Extract agricultural metrics, dosages, and quantities."""
        quantities = []
        for regex in self.QUANTITY_REGEXES:
            for match in regex.finditer(text):
                matched_str = match.group(0).strip()
                quantities.append({
                    "text": matched_str,
                    "canonical": matched_str,
                    "type": "METRIC_QUANTITY",
                    "start": match.start(),
                    "end": match.end()
                })
        return quantities

    def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract all agricultural entities with offsets, canonical names, and category grouping."""
        if not text:
            return {
                "entities": [],
                "grouped": {
                    "CROPS": [],
                    "DISEASES": [],
                    "PESTS": [],
                    "FERTILIZERS_CHEMICALS": [],
                    "SOIL_TYPES": [],
                    "GROWTH_STAGES": [],
                    "METRIC_QUANTITIES": []
                },
                "summary": {}
            }

        text_lower = text.lower()
        extracted: List[Dict[str, Any]] = []
        occupied_spans: List[tuple[int, int]] = []

        # Helper to avoid overlapping spans
        def is_overlapping(start: int, end: int) -> bool:
            return any(not (end <= s or start >= e) for s, e in occupied_spans)

        # 1. First extract metric quantities
        for q in self.extract_quantities(text):
            if not is_overlapping(q["start"], q["end"]):
                extracted.append(q)
                occupied_spans.append((q["start"], q["end"]))

        # 2. Extract gazetteer entities (match longer phrases first)
        for entity_type, lookup in self.entity_gazetteer.items():
            # Sort aliases by length descending to match multi-word phrases first
            sorted_aliases = sorted(lookup.keys(), key=len, reverse=True)
            for alias in sorted_aliases:
                # Use word boundary matching for precise recognition
                pattern = r'(?<!\w)' + re.escape(alias) + r'(?!\w)'
                for match in re.finditer(pattern, text_lower):
                    start, end = match.start(), match.end()
                    if not is_overlapping(start, end):
                        canonical = lookup[alias]
                        extracted.append({
                            "text": text[start:end],
                            "canonical": canonical,
                            "type": entity_type,
                            "start": start,
                            "end": end
                        })
                        occupied_spans.append((start, end))

        # Sort extracted by character position
        extracted.sort(key=lambda x: x["start"])

        # Group by category
        grouped: Dict[str, List[str]] = {
            "CROPS": list(dict.fromkeys([e["canonical"] for e in extracted if e["type"] == "CROP"])),
            "DISEASES": list(dict.fromkeys([e["canonical"] for e in extracted if e["type"] == "DISEASE"])),
            "PESTS": list(dict.fromkeys([e["canonical"] for e in extracted if e["type"] == "PEST"])),
            "FERTILIZERS_CHEMICALS": list(dict.fromkeys([e["canonical"] for e in extracted if e["type"] == "FERTILIZER_CHEMICAL"])),
            "SOIL_TYPES": list(dict.fromkeys([e["canonical"] for e in extracted if e["type"] == "SOIL_TYPE"])),
            "GROWTH_STAGES": list(dict.fromkeys([e["canonical"] for e in extracted if e["type"] == "GROWTH_STAGE"])),
            "METRIC_QUANTITIES": list(dict.fromkeys([e["canonical"] for e in extracted if e["type"] == "METRIC_QUANTITY"]))
        }

        return {
            "entities": extracted,
            "grouped": grouped,
            "summary": {k: len(v) for k, v in grouped.items() if v}
        }


# Singleton instance
default_ner_extractor = AgriNERExtractor()
