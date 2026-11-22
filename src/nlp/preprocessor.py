"""Agricultural Text Preprocessing Module.
Handles normalization, agricultural entity preservation, contractions,
stopword filtering, and domain-aware tokenization.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Dict, Any


CONTRACTIONS: Dict[str, str] = {
    "can't": "cannot",
    "won't": "will not",
    "it's": "it is",
    "what's": "what is",
    "there's": "there is",
    "how's": "how is",
    "i'm": "i am",
    "i've": "i have",
    "we've": "we have",
    "they're": "they are",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "haven't": "have not",
    "hasn't": "has not",
    "shouldn't": "should not",
    "couldn't": "could not",
    "wouldn't": "would not"
}

# Stopwords tailored for agriculture - preserve questions, negations, quantities and conditions
COMMON_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "all", "any", "both", "each", "few",
    "more", "most", "other", "some", "such", "only", "own", "same", "so", "than", "too",
    "very", "s", "t", "just", "now", "please", "kindly", "tell", "me", "i", "my", "we", "our"
}

# Suffixes for rule-based stemming/lemmatization
IRREGULAR_LEMMAS = {
    "leaves": "leaf",
    "fungi": "fungus",
    "larvae": "larva",
    "pupae": "pupa",
    "caterpillars": "caterpillar",
    "maggots": "maggot",
    "blights": "blight",
    "lesions": "lesion",
    "spots": "spot",
    "rot": "rot",
    "rotting": "rot",
    "rotten": "rot",
    "curling": "curl",
    "wilting": "wilt",
    "wilted": "wilt",
    "yellowing": "yellow",
    "sprayed": "spray",
    "spraying": "spray",
    "sprays": "spray",
    "irrigated": "irrigate",
    "irrigating": "irrigate",
    "irrigation": "irrigate",
    "fertilized": "fertilize",
    "fertilizing": "fertilize",
    "fertilizers": "fertilizer",
    "fertilizer": "fertilizer",
    "pest": "pest",
    "pests": "pest",
    "insects": "insect",
    "aphid": "aphid",
    "aphids": "aphid",
    "borers": "borer"
}


class AgriTextPreprocessor:
    """Preprocesses raw agricultural queries into clean normalized tokens and text."""

    def __init__(self, remove_stopwords: bool = True, preserve_measurements: bool = True):
        self.remove_stopwords = remove_stopwords
        self.preserve_measurements = preserve_measurements

    def expand_contractions(self, text: str) -> str:
        """Expand English contractions."""
        lower_text = text.lower()
        for contraction, replacement in CONTRACTIONS.items():
            pattern = re.compile(re.escape(contraction), re.IGNORECASE)
            text = pattern.sub(replacement, text)
        return text

    def normalize_unicode(self, text: str) -> str:
        """Normalize accents and special characters."""
        return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")

    def preserve_units_and_formulas(self, text: str) -> str:
        """Standardize measurements like kg/ha, ml/L, 19-19-19, and pH values."""
        # e.g., "19:19:19" or "19-19-19"
        text = re.sub(r'(\d+)[\:\-](\d+)[\:\-](\d+)', r'\1-\2-\3', text)
        # e.g., "kg / ha" -> "kg/ha"
        text = re.sub(r'(\w+)\s*/\s*(\w+)', r'\1/\2', text)
        # e.g., "pH 6 . 5" -> "pH 6.5"
        text = re.sub(r'ph\s*(\d+)\s*\.\s*(\d+)', r'pH \1.\2', text, flags=re.IGNORECASE)
        return text

    def stem_or_lemmatize(self, word: str) -> str:
        """Domain-adapted rule-based lemmatizer/stemmer."""
        w = word.lower()
        if w in IRREGULAR_LEMMAS:
            return IRREGULAR_LEMMAS[w]
        
        # Simple suffix stripping
        if len(w) > 4:
            if w.endswith("ies") and not w.endswith("eies"):
                return w[:-3] + "y"
            if w.endswith("es") and not (w.endswith("tes") or w.endswith("ses")):
                return w[:-2]
            if w.endswith("s") and not w.endswith("ss") and not w.endswith("us") and not w.endswith("is"):
                return w[:-1]
            if w.endswith("ing") and len(w) > 5:
                # e.g. drying -> dry, cutting -> cut, spraying -> spray
                base = w[:-3]
                if len(base) > 2 and base[-1] == base[-2]:
                    return base[:-1]
                return base
            if w.endswith("ed") and len(w) > 4:
                base = w[:-2]
                if base.endswith("i"):
                    return base[:-1] + "y"
                return base
        return w

    def tokenize(self, text: str) -> List[str]:
        """Tokenize string while preserving decimals, hyphens in agricultural compound names and units."""
        # Pattern captures numbers with decimals, hyphenated compounds (e.g. late-blight, 19-19-19), and slash units (e.g. kg/ha, g/L)
        pattern = r'\b\d+(?:\.\d+)?(?:[\-\/][A-Za-z0-9]+)*\b|\b[A-Za-z0-9]+(?:[\-\/][A-Za-z0-9]+)*\b'
        return re.findall(pattern, text)

    def preprocess(self, text: str) -> Dict[str, Any]:
        """Full pipeline preprocessing."""
        if not text or not isinstance(text, str):
            return {
                "original": "",
                "cleaned": "",
                "tokens": [],
                "lemmas": [],
                "preserved_terms": []
            }

        original = text.strip()
        expanded = self.expand_contractions(original)
        normalized = self.normalize_unicode(expanded)
        preserved = self.preserve_units_and_formulas(normalized)

        raw_tokens = self.tokenize(preserved)
        cleaned_tokens = []
        lemmas = []
        preserved_terms = []

        for token in raw_tokens:
            lower_token = token.lower()
            
            # Record preserved terms like units, decimals, or hyphenated terms
            if "/" in token or ("-" in token and any(char.isdigit() for char in token)) or ("." in token and any(char.isdigit() for char in token)):
                preserved_terms.append(token)

            if self.remove_stopwords and lower_token in COMMON_STOPWORDS:
                continue

            cleaned_tokens.append(lower_token)
            lemmas.append(self.stem_or_lemmatize(lower_token))

        cleaned_text = " ".join(lemmas)

        return {
            "original": original,
            "cleaned": cleaned_text,
            "tokens": cleaned_tokens,
            "lemmas": lemmas,
            "preserved_terms": preserved_terms
        }


# Singleton instance for simple imports
default_preprocessor = AgriTextPreprocessor()
